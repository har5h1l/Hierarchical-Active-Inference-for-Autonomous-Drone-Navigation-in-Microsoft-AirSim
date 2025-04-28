import airsim
import numpy as np
import time
import os
import json
import subprocess
import sys
import zmq
from math import sqrt
from os import path

# Constants and hyperparameters
TARGET_LOCATION = [10.0, 20.0, -20.0]  # [x, y, z] in NED coordinates (east, 15m high)
MARGIN = 1.5  # Safety margin for waypoint generation (meters)
WAYPOINT_SAMPLE_COUNT = 75  # Number of waypoints to consider
POLICY_LENGTH = 3  # Number of steps in the policy
DENSITY_RADIUS = 5.0  # Radius for density evaluation
ARRIVAL_THRESHOLD = 1.0  # Distance in meters to consider target reached
MAX_ITERATIONS = 50  # Maximum number of iterations before stopping
INTERFACE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "interface"))
JULIA_PATH = "julia --project=."  # Path to Julia executable with project environment
ZMQ_SERVER_ADDRESS = "tcp://localhost:5555"  # ZeroMQ server address

# Add the airsim directory to the Python path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'airsim'))

# Import the scanner with the correct filename
try:
    from Sensory_Input_Processing import EnvironmentScanner
except ImportError:
    # Try with the original filename if renamed file not found
    from airsim.Sensory_Input_Processing import EnvironmentScanner


class ZMQInterface:
    """ZeroMQ-based interface for communication with Julia inference and planning server."""
    
    def __init__(self, server_address=ZMQ_SERVER_ADDRESS, timeout=8000, max_retries=3):
        """Initialize ZeroMQ client interface.
        
        Args:
            server_address: Address of the ZMQ server
            timeout: Socket timeout in milliseconds (reduced for faster retries)
            max_retries: Maximum number of connection retries
        """
        self.server_address = server_address
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._context = None
        self._socket = None
        self._connected = False
        self._server_process = None
        
        # Check if server is running and start it if needed
        if not self._is_server_running():
            print("ZMQ server not running. Starting server...")
            self._start_server()
        
        # Initialize connection
        self._setup_connection()
    
    def _is_server_running(self):
        """Check if the ZMQ server is running by looking for its status file."""
        status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zmq_server_running.status")
        return os.path.isfile(status_file)
    
    def _start_server(self):
        """Start the ZeroMQ server as a background process."""
        try:
            # Get the path to the Julia server script
            server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       "actinf", "zmq_server.jl")
            
            # Verify the script exists
            if not os.path.isfile(server_script):
                print(f"❌ Server script not found at: {server_script}")
                return False
                
            print(f"Found server script at: {server_script}")
            
            # First check if Julia is installed and available
            try:
                julia_version = subprocess.check_output(["julia", "--version"], 
                                                     encoding='utf-8',
                                                     errors='replace')
                print(f"Julia found: {julia_version.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error checking Julia version: {str(e)}")
                return False
            except FileNotFoundError:
                print("❌ Julia not found in PATH. Make sure Julia is installed and in your PATH.")
                return False
            
            # First develop the actinf package to ensure it's properly linked
            try:
                print("Developing actinf package before starting server...")
                workspace_root = os.path.dirname(os.path.abspath(__file__))
                dev_result = subprocess.run(
                    ["julia", "--project=.", "-e", 
                     "using Pkg; Pkg.develop(path=joinpath(pwd(), \"actinf\")); Pkg.instantiate(); println(\"✅ Package development complete\")"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding='utf-8',
                    errors='replace',
                    text=True,
                    cwd=workspace_root
                )
                print(dev_result.stdout)
                if dev_result.stderr.strip():
                    print(f"Stderr from package development: {dev_result.stderr}")
            except Exception as e:
                print(f"⚠️ Warning during package development: {e}")
                print("Continuing anyway, but ZMQ server might have issues...")
            
            # Explicitly kill any remnant zmq_server.jl processes first
            try:
                if os.name == 'nt':  # Windows
                    subprocess.call('taskkill /F /IM "julia.exe" /FI "WINDOWTITLE eq zmq_server.jl"', shell=True)
                else:  # Unix/Linux/Mac
                    subprocess.call("pkill -f 'julia.*zmq_server.jl'", shell=True)
                print("✅ Killed any existing Julia ZMQ server processes")
            except Exception as e:
                print(f"Note: Could not kill existing processes: {e}")
            
            # Remove status file if it exists (to ensure a clean start)
            status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zmq_server_running.status")
            if os.path.exists(status_file):
                try:
                    os.remove(status_file)
                    print(f"✅ Removed existing status file: {status_file}")
                except Exception as e:
                    print(f"⚠️ Could not remove existing status file: {e}")
            
            # Start the server process with UTF-8 encoding and detailed environment setup
            # Prepare Julia command using direct script execution instead of -e
            workspace_root = os.path.dirname(os.path.abspath(__file__))
            
            # Use a direct command execution instead of -e to avoid issues
            cmd = [
                "julia", 
                "--project=.",
                os.path.join(workspace_root, "actinf", "zmq_server.jl")
            ]
            
            print(f"Starting server with command: {' '.join(cmd)}")
            
            # Use Popen with UTF-8 encoding and line buffering
            self._server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                encoding='utf-8',  # Explicitly use UTF-8 encoding
                errors='replace',  # Replace invalid characters instead of failing
                bufsize=1,  # Line buffered
                universal_newlines=True,
                cwd=workspace_root  # Ensure correct working directory
            )
            
            print("Server process started with PID:", self._server_process.pid)
            
            # Wait for server to start up (check for status file)
            start_time = time.time()
            
            print(f"Waiting for status file: {status_file}")
            
            # Start reading process output immediately for better diagnostics
            from queue import Queue, Empty
            from threading import Thread
            
            stdout_queue = Queue()
            stderr_queue = Queue()
            
            # Store thread references to avoid premature garbage collection
            self._stdout_thread = None
            self._stderr_thread = None
            
            def read_output(pipe, queue, pipe_name):
                """Read output from pipe and put into queue."""
                try:
                    print(f"Started {pipe_name} reader thread")
                    for line in pipe:
                        if line:
                            queue.put(line)
                            # Print output in real-time for immediate feedback
                            print(f"Server {pipe_name}: {line.strip()}")
                except Exception as e:
                    error_msg = f"Error reading {pipe_name}: {e}"
                    queue.put(error_msg)
                    print(error_msg)
                finally:
                    print(f"Closing {pipe_name} pipe")
                    try:
                        pipe.close()
                    except Exception as e:
                        print(f"Error closing {pipe_name} pipe: {e}")
            
            # Create and start output reader threads
            self._stdout_thread = Thread(target=read_output, 
                                        args=(self._server_process.stdout, stdout_queue, "STDOUT"),
                                        daemon=True)
            self._stderr_thread = Thread(target=read_output, 
                                        args=(self._server_process.stderr, stderr_queue, "STDERR"),
                                        daemon=True)
            
            self._stdout_thread.start()
            self._stderr_thread.start()
            
            # Wait longer for the status file (60 seconds instead of 30)
            while not os.path.isfile(status_file) and time.time() - start_time < 60:
                time.sleep(0.5)
                
                # Check if process has terminated prematurely
                if self._server_process.poll() is not None:
                    # Process ended - get output
                    exit_code = self._server_process.returncode
                    print(f"❌ Server process terminated prematurely with exit code: {exit_code}")
                    
                    # Collect any output from queues
                    stdout_data = ""
                    stderr_data = ""
                    
                    try:
                        while not stdout_queue.empty():
                            stdout_data += stdout_queue.get_nowait()
                    except Exception:
                        pass
                    
                    try:
                        while not stderr_queue.empty():
                            stderr_data += stderr_queue.get_nowait()
                    except Exception:
                        pass
                    
                    if stdout_data:
                        print("STDOUT from terminated process:")
                        print(stdout_data)
                    
                    if stderr_data:
                        print("STDERR from terminated process:")
                        print(stderr_data)
                    
                    return False
            
            if os.path.isfile(status_file):
                print("✅ ZMQ server started successfully")
                # Wait a bit more to ensure the socket is bound
                time.sleep(1.5)
                return True
            else:
                # Process is still running but no status file - collect all available output
                stdout_data = ""
                stderr_data = ""
                
                try:
                    # Collect any remaining output from the queues
                    try:
                        while not stdout_queue.empty():
                            stdout_data += stdout_queue.get_nowait()
                    except Exception as e:
                        print(f"Error getting stdout data: {e}")
                        
                    try:
                        while not stderr_queue.empty():
                            stderr_data += stderr_queue.get_nowait()
                    except Exception as e:
                        print(f"Error getting stderr data: {e}")
                    
                    print("❌ Failed to start ZMQ server (status file not found)")
                    
                    if stdout_data:
                        print("Server STDOUT:")
                        print(stdout_data)
                    
                    if stderr_data:
                        print("Server STDERR:")
                        print(stderr_data)
                    
                    # Try to check if the process is still running
                    if self._server_process.poll() is None:
                        print("⚠️ Server process is still running, but no status file was created.")
                        print("Checking for port binding...")
                        
                        # Check if port 5555 is in use
                        try:
                            test_context = zmq.Context()
                            test_socket = test_context.socket(zmq.REQ)
                            test_socket.setsockopt(zmq.LINGER, 0)
                            test_socket.setsockopt(zmq.RCVTIMEO, 1000)
                            test_socket.setsockopt(zmq.SNDTIMEO, 1000)
                            test_socket.connect("tcp://localhost:5555")
                            test_socket.send_string("ping")
                            response = test_socket.recv_string()
                            if response == "pong":
                                print("✅ Port is bound and responding to pings, creating status file manually")
                                # Create status file manually
                                with open(status_file, "w") as f:
                                    f.write("running")
                                test_socket.close()
                                test_context.term()
                                return True
                            test_socket.close()
                            test_context.term()
                        except Exception:
                            pass
                        
                        print("❌ Server process is not responding, terminating")
                    
                    # Try to terminate the process since it didn't start correctly
                    try:
                        print("Terminating server process...")
                        self._server_process.terminate()
                        time.sleep(1)
                        if self._server_process.poll() is None:  # Still running
                            print("Killing server process forcefully...")
                            if os.name == 'nt':  # Windows
                                subprocess.call(f'taskkill /F /PID {self._server_process.pid}', shell=True)
                            else:  # Unix/Linux/Mac
                                self._server_process.kill()
                        self._server_process = None
                    except Exception as e:
                        print(f"Error terminating process: {e}")
                
                except Exception as e:
                    print(f"Error collecting process output: {e}")
                
                return False
                
        except Exception as e:
            print(f"❌ Error starting ZMQ server: {str(e)}")
            return False
    
    def _cleanup_socket(self):
        """Clean up ZMQ socket and context properly."""
        try:
            if self._socket is not None:
                try:
                    # Check if socket is still valid (not already closed)
                    if hasattr(self._socket, 'closed') and not self._socket.closed:
                        self._socket.close()
                        print("Socket closed")
                    else:
                        print("Socket was already closed")
                except Exception as e:
                    print(f"Warning during socket close: {str(e)}")
                finally:
                    self._socket = None
                
            # Only terminate context if we're creating a new one
            if self._context is not None:
                try:
                    # Give a small delay for pending messages to complete
                    time.sleep(0.2)
                    
                    # Check if context is not already terminated
                    if hasattr(self._context, 'closed') and not self._context.closed:
                        self._context.term()
                        print("ZMQ context terminated")
                    else:
                        print("ZMQ context was already terminated")
                except Exception as e:
                    print(f"Warning during context termination: {str(e)}")
                finally:
                    self._context = None
                
        except Exception as e:
            print(f"Warning during socket cleanup: {str(e)}")
    
    def _reset_socket(self):
        """Reset the ZMQ socket to recover from an inconsistent state."""
        print("🔄 Resetting ZMQ socket...")
        
        # Clean up existing socket and context
        self._cleanup_socket()
        
        # Wait a moment before reconnecting
        time.sleep(0.5)
        
        # Make sure server is running
        if not self._is_server_running():
            print("⚠️ Server not running during reset, attempting to start...")
            if not self._start_server():
                print("❌ Could not start server during reset")
                return False
        else:
            print("✅ ZMQ server is already running (status file exists)")
            # Verify the server process is actually running if we started it
            if self._server_process is not None:
                if self._server_process.poll() is not None:
                    print("⚠️ Server process has terminated but status file exists!")
                    # Remove stale status file
                    try:
                        status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zmq_server_running.status")
                        if os.path.exists(status_file):
                            os.remove(status_file)
                            print("✅ Removed stale status file")
                    except Exception as e:
                        print(f"⚠️ Failed to remove stale status file: {e}")
                    
                    # Restart server
                    print("🔄 Restarting ZMQ server...")
                    if not self._start_server():
                        print("❌ Failed to restart server")
                        return False
        
        # Create new context and socket with simpler settings
        try:
            print("Creating new ZMQ context and socket...")
            # Clean up any existing socket first
            if self._context is not None:
                try:
                    self._context.term()
                except Exception as e:
                    print(f"Error terminating context: {e}")
            
            # Create completely fresh context and socket
            self._context = zmq.Context.instance()
            self._socket = self._context.socket(zmq.REQ)
            
            # Set minimal socket options for better reliability
            self._socket.setsockopt(zmq.LINGER, 0)  # Don't wait for unsent messages
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self._socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            
            # Set additional socket options for reliability
            self._socket.setsockopt(zmq.TCP_KEEPALIVE, 1)  # Enable keep-alive
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)  # Seconds before sending keepalives
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 10)  # Interval between keepalives
            
            # Set appropriate high water marks
            self._socket.setsockopt(zmq.RCVHWM, 10)
            self._socket.setsockopt(zmq.SNDHWM, 10)
            
            # Reconnect
            print(f"Connecting to {self.server_address}...")
            self._socket.connect(self.server_address)
            self._connected = True
            print("✅ Socket reset and reconnected")
            
            # Use the improved ping method
            print("Verifying connection...")
            ping_success = self._simple_ping()
            
            if ping_success:
                print("✅ Connection verified with ping")
                return True
            else:
                print("❌ Ping failed after reset")
                self._connected = False
                
                # Try to restart the server if ping fails
                print("🔄 Attempting to restart server after failed ping...")
                # Remove stale status file if exists
                try:
                    status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zmq_server_running.status")
                    if os.path.exists(status_file):
                        os.remove(status_file)
                except Exception:
                    pass
                
                # Restart server
                if self._start_server():
                    # Try connecting again
                    time.sleep(2)  # Wait a bit longer
                    try:
                        # Close old socket first
                        if self._socket:
                            self._socket.close()
                        
                        # Create new socket
                        self._socket = self._context.socket(zmq.REQ)
                        self._socket.setsockopt(zmq.LINGER, 0)
                        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout)
                        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout)
                        self._socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
                        self._socket.connect(self.server_address)
                        self._connected = True
                        
                        # Try ping again
                        if self._simple_ping():
                            print("✅ Connection verified with ping after server restart")
                            return True
                    except Exception as e:
                        print(f"❌ Error reconnecting after server restart: {e}")
                
                return False
                
        except Exception as e:
            print(f"❌ Failed to reconnect after socket reset: {str(e)}")
            self._connected = False
            return False
    
    def _setup_connection(self):
        """Set up the ZMQ connection with proper context and socket."""
        return self._reset_socket()
    
    def _simple_ping(self):
        """Test the connection with a simple ping to the server."""
        if not self._connected or self._socket is None:
            print("Cannot ping: socket not connected or None")
            return False
        
        # Create a new socket specifically for the ping test
        # This avoids potential issues with the existing socket state
        try:
            ping_context = zmq.Context()
            ping_socket = ping_context.socket(zmq.REQ)
            ping_socket.setsockopt(zmq.LINGER, 0)
            ping_socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 seconds timeout (increased)
            ping_socket.setsockopt(zmq.SNDTIMEO, 3000)  # 3 seconds timeout (increased)
            ping_socket.connect(self.server_address)
            
            print("Sending ping to test connection with fresh socket...")
            ping_socket.send_string("ping")
            print("Ping sent, waiting for response...")
            
            # Add a short delay to let the server process the ping
            time.sleep(0.5)
            
            try:
                response = ping_socket.recv_string()
                print(f"Received response: {response}")
                
                # Clean up ping socket
                ping_socket.close()
                ping_context.term()
                
                return response == "pong"
            except zmq.ZMQError as e:
                print(f"Error receiving ping response: {e}")
                # Try once more after a longer delay
                time.sleep(1)
                try:
                    response = ping_socket.recv_string()
                    print(f"Received response on second attempt: {response}")
                    
                    # Clean up ping socket
                    ping_socket.close()
                    ping_context.term()
                    
                    return response == "pong"
                except zmq.ZMQError as e:
                    print(f"Error receiving ping response on second attempt: {e}")
                    
                    # Clean up ping socket
                    ping_socket.close()
                    ping_context.term()
                    
                    return False
                
        except zmq.ZMQError as e:
            print(f"Ping failed with ZMQ error: {e}")
            # Try to clean up ping socket if created
            try:
                if 'ping_socket' in locals():
                    ping_socket.close()
                if 'ping_context' in locals():
                    ping_context.term()
            except:
                pass
            return False
        except Exception as e:
            print(f"Ping failed with unexpected error: {e}")
            # Try to clean up ping socket if created
            try:
                if 'ping_socket' in locals():
                    ping_socket.close()
                if 'ping_context' in locals():
                    ping_context.term()
            except:
                pass
            return False
    
    def _reduce_voxel_grid(self, observation, max_voxels=1500):
        """Reduce the size of the voxel grid if it's too large."""
        if "voxel_grid" in observation and len(observation["voxel_grid"]) > max_voxels:
            original_count = len(observation["voxel_grid"])
            # Take random sample to reduce size but maintain coverage
            import random
            indices = random.sample(range(original_count), max_voxels)
            observation["voxel_grid"] = [observation["voxel_grid"][i] for i in indices]
            print(f"⚙️ Reduced voxel grid from {original_count} to {max_voxels} points")
        
        return observation
    
    def send_observation_and_receive_action(self, observation):
        """Send observation to Julia server and receive inferred state and planned action."""
        if not self._connected:
            # Try to reconnect if not connected
            print("ZMQ connection not established, attempting to connect...")
            if not self._setup_connection():
                print("❌ Not connected to ZMQ server and reconnection failed")
                return None, [0.0, 0.0, 0.0]
        
        # Verify server process is still running if we started it
        if self._server_process is not None and self._server_process.poll() is not None:
            print("❌ Server process has terminated unexpectedly!")
            print(f"Exit code: {self._server_process.returncode}")
            
            # Try to restart
            print("Attempting to restart server...")
            if not self._reset_socket():
                print("❌ Failed to restart server")
                return None, [0.0, 0.0, 0.0]
        
        # Default values in case of failure
        default_state = {
            "distance": 10.0,
            "azimuth": 0.0,
            "elevation": 0.0,
            "suitability": 0.5
        }
        default_action = [0.0, 0.0, 0.0]
        
        # Ensure target_position key is used instead of target_location
        if "target_location" in observation and "target_position" not in observation:
            observation["target_position"] = observation.pop("target_location")
        
        # Optimize message size by reducing voxel grid if necessary
        observation = self._reduce_voxel_grid(observation)
        
        # Convert observation to JSON string
        try:
            obs_json = json.dumps(observation)
            msg_size = len(obs_json) / 1024  # Size in KB
        except Exception as e:
            print(f"❌ Error serializing observation to JSON: {str(e)}")
            return default_state, default_action
        
        # Try to send observation and receive response
        for attempt in range(self.max_retries):
            try:
                # Send observation
                print(f"📤 Sending observation data ({round(msg_size, 1)} KB)...")
                start_time = time.time()  # Start timing the request
                
                # Set a timeout for the send operation
                self._socket.setsockopt(zmq.SNDTIMEO, self.timeout)
                try:
                    self._socket.send_string(obs_json)
                except zmq.ZMQError as e:
                    print(f"❌ Error sending data: {str(e)}")
                    if attempt < self.max_retries - 1:
                        print("Attempting to reset socket before retrying...")
                        self._reset_socket()
                        time.sleep(1)
                        continue
                    else:
                        return default_state, default_action
                
                # Receive response
                print(f"⏳ Waiting for response (timeout: {self.timeout/1000}s)...")
                
                # Set a timeout for the receive operation
                self._socket.setsockopt(zmq.RCVTIMEO, self.timeout)
                try:
                    response_json = self._socket.recv_string()
                except zmq.ZMQError as e:
                    print(f"❌ Error receiving data: {str(e)}")
                    if attempt < self.max_retries - 1:
                        print("Attempting to reset socket before retrying...")
                        self._reset_socket()
                        time.sleep(1)
                        continue
                    else:
                        return default_state, default_action
                
                # Calculate response time
                elapsed = time.time() - start_time
                
                resp_size = len(response_json) / 1024  # Size in KB
                print(f"📥 Received response ({round(resp_size, 1)} KB) in {round(elapsed, 2)}s")
                
                # Parse response
                try:
                    response = json.loads(response_json)
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing response JSON: {str(e)}")
                    print(f"Response content: {response_json[:100]}...")  # Show start of response
                    if attempt < self.max_retries - 1:
                        print("Retrying with fresh connection...")
                        self._reset_socket()
                        time.sleep(1)
                        continue
                    else:
                        return default_state, default_action
                
                # Check for errors in the response
                if "error" in response:
                    print(f"⚠️ Error from server: {response['error']}")
                    return default_state, response.get("action", default_action)
                
                # Extract expected state and action
                expected_state = response.get("expected_state", default_state)
                action = response.get("action", default_action)
                
                print(f"✅ Received action: {[round(x, 3) for x in action]}")
                return expected_state, action
                
            except zmq.ZMQError as e:
                err_num = getattr(e, 'errno', None)
                
                if attempt < self.max_retries - 1:
                    # Check if we lost the server
                    server_running = self._is_server_running()
                    
                    if not server_running:
                        print("⚠️ Server appears to have stopped. Attempting restart...")
                        self._start_server()
                        time.sleep(2)  # Give it time to start
                    
                    print(f"❌ ZMQ error during attempt {attempt+1}/{self.max_retries}: {str(e)}")
                    success = self._reset_socket()
                    
                    if not success:
                        print("❌ Failed to reset connection")
                    
                    # Wait before retrying
                    time.sleep(1)
                else:
                    print(f"❌ ZMQ error on final attempt: {str(e)}")
                
            except Exception as e:
                print(f"❌ Unexpected error during attempt {attempt+1}/{self.max_retries}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                
                if attempt < self.max_retries - 1:
                    print("🔄 Retrying with fresh connection...")
                    self._reset_socket()
                    time.sleep(1)
        
        # If all attempts failed
        print("❌ All attempts to communicate with the server failed")
        return default_state, default_action
    
    def close(self):
        """Close the ZMQ connection and clean up resources."""
        try:
            # First clean up the socket
            self._cleanup_socket()
            
            # Check if the server process is still running
            if self._server_process is not None:
                # We don't terminate the server process by default
                # This allows multiple clients to use the same server instance
                
                # But we should set the reference to None
                self._server_process = None
            
            # Clean up thread references
            self._stdout_thread = None
            self._stderr_thread = None
            
            # Reset connection state
            self._connected = False
            
            print("✅ ZMQ connection closed and resources cleaned up")
        except Exception as e:
            print(f"❌ Error closing ZMQ connection: {str(e)}")


class DroneController:
    def __init__(self):
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.scanner = EnvironmentScanner(self.client)
        
        # Initialize state
        self.current_position = None
        self.current_orientation = None
        self.target_location = TARGET_LOCATION
        
        print("DroneController initialized and connected to AirSim")
    
    def precompile_julia_components(self):
        """Run the Julia precompilation script to prepare all components"""
        print("⏳ Precompiling Julia components...")
        try:
            # First, ensure the actinf package is developed properly
            print("Developing actinf package...")
            develop_cmd = ["julia", "--project=.", "-e", 
                        "using Pkg; Pkg.develop(path=joinpath(pwd(), \"actinf\")); Pkg.instantiate()"]
            subprocess.run(develop_cmd, check=True, 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                         encoding='utf-8', errors='replace', text=True)
            
            # Run the actual precompilation script
            precompile_script = os.path.normpath("./precompile.jl")
            print(f"Running Julia precompilation: {precompile_script}")
            result = subprocess.run(
                ["julia", "--project=.", precompile_script],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8', 
                errors='replace',
                text=True
            )
            print("✅ Julia precompilation completed successfully")
            
            if "error" in result.stdout.lower() or "error" in result.stderr.lower():
                print("⚠️ Warnings or errors during precompilation:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during Julia precompilation: {str(e)}")
            print(f"Stdout: {e.stdout if hasattr(e, 'stdout') else 'No output'}")
            print(f"Stderr: {e.stderr if hasattr(e, 'stderr') else 'No error output'}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during Julia precompilation: {str(e)}")
            return False
    
    def reset_and_takeoff(self):
        """Reset AirSim, arm the drone and take off to default height"""
        print("Resetting AirSim...")
        self.client.reset()
        
        print("Enabling API control and arming drone...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        print("Taking off...")
        self.client.takeoffAsync().join()
        
        # Hover for a moment to stabilize
        time.sleep(2)
        
        # Get current position after takeoff
        self.update_drone_state()
        print(f"Drone hovering at position: {self.current_position}")
    
    def update_drone_state(self):
        """Update the current position and orientation of the drone"""
        state = self.client.getMultirotorState()
        self.current_position = [state.kinematics_estimated.position.x_val,
                                state.kinematics_estimated.position.y_val,
                                state.kinematics_estimated.position.z_val]
        
        self.current_orientation = [state.kinematics_estimated.orientation.w_val,
                                   state.kinematics_estimated.orientation.x_val,
                                   state.kinematics_estimated.orientation.y_val,
                                   state.kinematics_estimated.orientation.z_val]
        return self.current_position, self.current_orientation
    
    def collect_sensor_data(self):
        """Use the EnvironmentScanner to collect sensor data"""
        # Update drone state first
        self.update_drone_state()
        
        # Get obstacle information
        try:
            print("\n==== COLLECTING SENSOR DATA ====")
            print(f"Current position: {[round(p, 2) for p in self.current_position]}")
            
            num_obstacles, distances = self.scanner.fetch_density_distances()
            print(f"📊 Detected {num_obstacles} obstacles within {DENSITY_RADIUS}m radius")
            
            if distances:
                print(f"📊 Nearest obstacles: {[round(d, 2) for d in sorted(distances)[:5]]}m")
            
            # Get voxel coordinates (obstacle map)
            points = self.scanner.collect_sensor_data()
            if points is None:
                print("⚠️ No point cloud data received from sensors")
                voxel_grid = []
            else:
                print(f"📊 Collected {len(points)} raw point cloud points")
                voxel_grid = self.scanner.create_obstacle_voxel_grid(points)
                if voxel_grid is None:
                    print("⚠️ Failed to create voxel grid")
                    voxel_grid = []
                else:
                    # Convert numpy array to list for JSON serialization
                    voxel_grid = voxel_grid.tolist()
                    print(f"📊 Generated voxel grid with {len(voxel_grid)} points")
                    
                    # Print some sample voxels for debugging
                    if len(voxel_grid) > 0:
                        sample_size = min(3, len(voxel_grid))
                        print(f"📊 Sample voxels (first {sample_size}):")
                        for i in range(sample_size):
                            print(f"  → Voxel {i}: {[round(v, 2) for v in voxel_grid[i]]}")
            
            # Calculate obstacle density (number of obstacles within radius)
            density = num_obstacles / (4/3 * np.pi * DENSITY_RADIUS**3) if num_obstacles > 0 else 0
            print(f"📊 Obstacle density: {density:.4f}")
            
            # Create the observation dictionary
            observation = {
                "drone_position": self.current_position,
                "drone_orientation": self.current_orientation,
                "voxel_grid": voxel_grid,
                "nearest_obstacle_distances": distances[:2] if len(distances) >= 2 else 
                    (distances + [100.0] * (2 - len(distances)) if distances else [100.0, 100.0]),
                "obstacle_density": density,
                "target_location": self.target_location
            }
            
            print(f"📊 Target location: {[round(t, 2) for t in self.target_location]}")
            print(f"📊 Straight-line distance to target: {self.distance_to_target():.2f}m")
            print("==========================\n")
            
            return observation
            
        except Exception as e:
            print(f"Error collecting sensor data: {str(e)}")
            # Return a default observation if something fails
            return {
                "drone_position": self.current_position,
                "drone_orientation": self.current_orientation,
                "voxel_grid": [],
                "nearest_obstacle_distances": [100.0, 100.0],
                "obstacle_density": 0.0,
                "target_location": self.target_location
            }
    
    def save_observation_to_json(self, observation):
        """Save the observation dictionary to a JSON file for Julia"""
        # Ensure interface directory exists
        os.makedirs(INTERFACE_DIR, exist_ok=True)
        
        filepath = os.path.normpath(os.path.join(INTERFACE_DIR, "obs_input.json"))
        with open(filepath, 'w') as f:
            json.dump(observation, f, indent=2)
        print(f"Observation saved to {filepath}")
        return filepath
    
    def run_julia_inference_and_planning(self):
        """Execute Julia scripts for inference and planning"""
        try:
            # Run inference script
            inference_script = os.path.normpath("./run_inference.jl")
            print(f"Running Julia inference: {inference_script}")
            subprocess.run(["julia", "--project=.", inference_script], check=True)
            
            # Run planning script
            planning_script = os.path.normpath("./run_planning.jl")
            print(f"Running Julia planning: {planning_script}")
            subprocess.run(["julia", "--project=.", planning_script], check=True)
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running Julia scripts: {str(e)}")
            print(f"Command output: {e.output if hasattr(e, 'output') else 'No output available'}")
            return False
        except Exception as e:
            print(f"Unexpected error running Julia scripts: {str(e)}")
            return False
    
    def check_for_obstacles(self, safety_threshold=2.0):
        """Check if there are obstacles too close to the drone
        
        Args:
            safety_threshold: Minimum safe distance to obstacles (meters)
            
        Returns:
            bool: True if obstacle is detected within threshold, False otherwise
        """
        try:
            # Get updated obstacle information
            num_obstacles, distances = self.scanner.fetch_density_distances()
            
            # Debug: Print obstacle information
            print(f"🔍 OBSTACLE CHECK: Found {num_obstacles} objects within sensing range")
            if distances:
                print(f"🔍 Nearest obstacles: {[round(d, 2) for d in sorted(distances)[:3]]}m")
            
            # Check if any obstacle is within safety threshold
            if distances and min(distances) < safety_threshold:
                nearest_obstacle = min(distances)
                print(f"⚠️ OBSTACLE DETECTED at {nearest_obstacle:.2f}m! (threshold: {safety_threshold}m)")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking for obstacles: {str(e)}")
            # Assume there's an obstacle in case of error (safer)
            return True
    
    def move_to_waypoint(self, waypoint):
        """Command the drone to move to the specified waypoint with obstacle avoidance"""
        print(f"Moving to: {[round(w, 1) for w in waypoint]}")
        
        # Ensure waypoint is within simulation boundaries and round coordinates
        waypoint = [
            round(min(max(waypoint[0], -100), 100), 2),  # X between -100 and 100
            round(min(max(waypoint[1], -100), 100), 2),  # Y between -100 and 100
            round(min(max(waypoint[2], -20), 0), 2)      # Z between -20 and 0 (remember NED)
        ]
        
        try:
            # Start movement asynchronously
            movement_task = self.client.moveToPositionAsync(
                x=waypoint[0],
                y=waypoint[1],
                z=waypoint[2],
                velocity=2,
                timeout_sec=15,  # Timeout after 15 seconds
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False),
                lookahead=-1,
                adaptive_lookahead=1
            )
            
            # Monitor for obstacles while the drone is moving
            obstacle_detected = False
            check_interval = 0.1  # Check every 100ms
            
            # Monitor movement until completion or obstacle detection
            start_time = time.time()
            previous_position = self.current_position.copy()
            movement_complete = False
            
            while not movement_complete:
                # Sleep to avoid CPU overload
                time.sleep(check_interval)
                
                # Update current position
                self.update_drone_state()
                
                # Check if we've reached the destination
                current_distance_to_waypoint = sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, waypoint)))
                
                # Check for movement progress
                distance_moved = sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, previous_position)))
                previous_position = self.current_position.copy()
                
                # Determine if movement is complete by checking:
                # 1. If we're very close to the target
                # 2. Or if we've stopped moving for a while
                if current_distance_to_waypoint < 0.5:  # Within 0.5m of target
                    movement_complete = True
                elif time.time() - start_time > 15:  # Timeout after 15 seconds
                    print("Movement timed out")
                    movement_complete = True
                elif distance_moved < 0.01 and time.time() - start_time > 2:  # Stopped moving
                    # Only consider it complete if we've been trying for at least 2 seconds
                    # and haven't moved more than 1cm recently
                    movement_complete = True
                
                # Check for obstacles
                if self.check_for_obstacles(safety_threshold=2.5):
                    print("🛑 Obstacle detected! Stopping.")
                    self.client.cancelLastTask()  # Cancel the movement task
                    self.client.hoverAsync().join()  # Hover in place
                    obstacle_detected = True
                    break
            
            # Wait for any pending movement to settle
            if not obstacle_detected:
                self.client.hoverAsync().join()
            
            # Update state after movement
            self.update_drone_state()
            
            # If an obstacle was detected, trigger a new planning cycle
            if obstacle_detected:
                return "obstacle_detected"
            
            # Check if this is the final approach to target
            distance_to_target = self.distance_to_target()
            is_final_approach = distance_to_target < 2.0
            
            # Verify movement accuracy - more strict for final approach
            final_distance = sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, waypoint)))
            
            if final_distance > 0.3 or (is_final_approach and distance_to_target > ARRIVAL_THRESHOLD * 0.8):
                # Attempt final precision adjustment if close enough
                if final_distance < 2.0:
                    if is_final_approach:
                        print("Final approach in progress...")
                    
                    # Adjust velocity based on distance (slower when close to target)
                    approach_velocity = 0.5 if is_final_approach else 1.0
                    
                    # If this is final approach, aim directly at target instead of the waypoint
                    # This helps overcome any accumulated error from multiple movements
                    if is_final_approach and distance_to_target < 1.5:
                        final_waypoint = self.target_location
                    else:
                        final_waypoint = waypoint
                    
                    # Start precision movement
                    precision_task = self.client.moveToPositionAsync(
                        x=final_waypoint[0],
                        y=final_waypoint[1],
                        z=final_waypoint[2],
                        velocity=approach_velocity,  # Slower for precision
                        timeout_sec=15,
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=airsim.YawMode(is_rate=False),
                        lookahead=-1,
                        adaptive_lookahead=1
                    )
                    
                    # Monitor for obstacles during precision movement
                    start_time = time.time()
                    previous_position = self.current_position.copy()
                    movement_complete = False
                    
                    while not movement_complete:
                        # Sleep to avoid CPU overload
                        time.sleep(check_interval)
                        
                        # Update current position
                        self.update_drone_state()
                        
                        # Calculate distance to target and movement rate
                        current_distance = sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, final_waypoint)))
                        target_distance = self.distance_to_target()
                        distance_moved = sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, previous_position)))
                        previous_position = self.current_position.copy()
                        
                        # Check if movement is complete
                        precision_threshold = 0.2 if is_final_approach else 0.3
                        
                        if current_distance < precision_threshold:
                            movement_complete = True
                        elif is_final_approach and target_distance < ARRIVAL_THRESHOLD:
                            print(f"Target reached: {target_distance:.2f}m")
                            movement_complete = True
                        elif time.time() - start_time > 15:  # Timeout after 15 seconds
                            print("Precision movement timed out")
                            movement_complete = True
                        elif distance_moved < 0.005 and time.time() - start_time > 3:  # Stopped moving (smaller threshold)
                            movement_complete = True
                        
                        # Check for obstacles
                        if self.check_for_obstacles(safety_threshold=2.0):
                            print("🛑 Obstacle detected during precision movement!")
                            self.client.cancelLastTask()
                            self.client.hoverAsync().join()
                            return "obstacle_detected"
                    
                    # Ensure final stabilization
                    self.client.hoverAsync().join()
                    self.update_drone_state()
            
            # Final check to see if we've reached target
            target_distance = self.distance_to_target()
            if target_distance < ARRIVAL_THRESHOLD:
                print(f"✅ Target reached: {target_distance:.2f}m")
            
            return "success"
                
        except Exception as e:
            print(f"Error during movement: {str(e)}")
            # Try to stabilize the drone
            self.client.hoverAsync().join()
            self.update_drone_state()
            return "error"
    
    def distance_to_target(self):
        """Calculate the Euclidean distance from the current position to the target"""
        return sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, self.target_location)))
    
    def create_dummy_action_output(self):
        """Create a placeholder action file for testing without Julia"""
        os.makedirs(INTERFACE_DIR, exist_ok=True)
        
        # Simple action: move toward target
        direction = [
            self.target_location[0] - self.current_position[0],
            self.target_location[1] - self.current_position[1],
            self.target_location[2] - self.current_position[2]
        ]
        
        # Normalize to 1m step
        magnitude = sqrt(sum(d*d for d in direction))
        if magnitude > 0:
            direction = [d/magnitude for d in direction]
        
        action = {
            "next_waypoint": direction,
            "policy": [direction, direction, direction]  # Just repeat the same action
        }
        
        filepath = os.path.normpath(os.path.join(INTERFACE_DIR, "action_output.json"))
        with open(filepath, 'w') as f:
            json.dump(action, f, indent=2)
    
    def load_action_from_json(self):
        """Load the next waypoint and policy from Julia's output"""
        action_path = os.path.normpath(os.path.join(INTERFACE_DIR, "action_output.json"))
        
        try:
            with open(action_path, 'r') as f:
                action_data = json.load(f)
                
            next_waypoint = action_data.get("next_waypoint", [0, 0, 0])
            policy = action_data.get("policy", [[0, 0, 0]] * POLICY_LENGTH)
            
            return next_waypoint, policy
        except Exception as e:
            print(f"Error loading action from JSON: {str(e)}")
            # Return a small default movement in the direction of the target
            direction = [
                self.target_location[0] - self.current_position[0],
                self.target_location[1] - self.current_position[1],
                self.target_location[2] - self.current_position[2]
            ]
            
            # Normalize to 0.5m step (more cautious default)
            magnitude = sqrt(sum(d*d for d in direction))
            if magnitude > 0:
                direction = [d * 0.5 / magnitude for d in direction]
                
            return direction, [direction] * POLICY_LENGTH
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return [w, x, y, z]

    def _quaternion_conjugate(self, q):
        """Calculate quaternion conjugate"""
        w, x, y, z = q
        return [w, -x, -y, -z]

    def _rotate_vector_by_quaternion(self, vector, quaternion):
        """Rotate a vector using quaternion rotation
        
        Args:
            vector: 3D vector [x, y, z]
            quaternion: Quaternion [w, x, y, z]
            
        Returns:
            Rotated vector [x', y', z']
        """
        # Convert vector to quaternion representation (0 + xi + yj + zk)
        vq = [0.0] + vector
        
        # Get quaternion conjugate
        q_conj = self._quaternion_conjugate(quaternion)
        
        # Apply rotation: q * v * q^-1
        rotated = self._quaternion_multiply(
            self._quaternion_multiply(quaternion, vq),
            q_conj
        )
        
        # Extract vector part
        return rotated[1:4]

    def convert_to_global_waypoint(self, egocentric_waypoint):
        """Convert an egocentric waypoint to global coordinates
        
        The egocentric waypoint is relative to the drone's current position and orientation.
        We need to transform it to global coordinates.
        """
        # Print initial information for debugging
        print("\n==== COORDINATE TRANSFORMATION ====")
        print(f"Current position: {[round(p, 2) for p in self.current_position]}")
        print(f"Egocentric waypoint: {[round(w, 2) for w in egocentric_waypoint]}")
        
        # Calculate distance to target to adjust step size
        distance_to_target = self.distance_to_target()
        print(f"Distance to target: {distance_to_target:.2f}m")
        
        original_waypoint = egocentric_waypoint.copy()
        
        # For final approach, limit step size to avoid overshooting
        if distance_to_target < 2.0:
            # Scale down the step size as we get closer to target
            scale_factor = min(distance_to_target / 2.0, 0.8)
            print(f"Final approach: Scaling movement by {scale_factor:.2f}")
            egocentric_waypoint = [
                egocentric_waypoint[0] * scale_factor,
                egocentric_waypoint[1] * scale_factor,
                egocentric_waypoint[2] * scale_factor
            ]
        
        # Apply quaternion-based rotation to account for drone orientation
        # Note: AirSim uses NED coordinate system with quaternions in [w,x,y,z] format
        print(f"Drone orientation (quaternion): {[round(q, 3) for q in self.current_orientation]}")
        
        # Enable/disable orientation adjustment for testing
        apply_orientation = True  # Set to False to test without orientation adjustment
        
        if apply_orientation:
            # Rotate the egocentric vector using quaternion rotation
            rotated_vector = self._rotate_vector_by_quaternion(egocentric_waypoint, self.current_orientation)
            print(f"Orientation-adjusted vector: {[round(v, 2) for v in rotated_vector]}")
            
            # Add the rotated vector to current position
            global_waypoint = [
                self.current_position[0] + rotated_vector[0],
                self.current_position[1] + rotated_vector[1],
                self.current_position[2] + rotated_vector[2]
            ]
        else:
            print(f"WARNING: Orientation adjustment disabled for testing")
            
            # Add the egocentric vector to current position without rotation
            global_waypoint = [
                self.current_position[0] + egocentric_waypoint[0],
                self.current_position[1] + egocentric_waypoint[1],
                self.current_position[2] + egocentric_waypoint[2]
            ]
        
        # Print the result before boundary enforcement
        print(f"Transformed waypoint (before limits): {[round(w, 2) for w in global_waypoint]}")
        
        # Ensure waypoint is within simulation boundaries
        original_global = global_waypoint.copy()
        global_waypoint = [
            min(max(global_waypoint[0], -100), 100),  # X between -100 and 100
            min(max(global_waypoint[1], -100), 100),  # Y between -100 and 100
            min(max(global_waypoint[2], -20), 0)      # Z between -20 and 0 (remember NED)
        ]
        
        # Check if boundaries were enforced
        if original_global != global_waypoint:
            print(f"⚠️ Waypoint was outside boundaries and has been constrained!")
        
        print(f"Final global waypoint: {[round(w, 2) for w in global_waypoint]}")
        print("=================================\n")
        
        return global_waypoint

    def inspect_suitability_metric(self, expected_state):
        """Analyze the suitability metric to understand path planning decisions
        
        Args:
            expected_state: The state belief received from the Julia inference engine
        """
        if not expected_state:
            print("No expected state available to analyze")
            return
        
        # Extract suitability value
        suitability = expected_state.get('suitability', None)
        
        if suitability is None:
            print("No suitability metric found in expected state")
            return
        
        print("\n==== SUITABILITY ANALYSIS ====")
        print(f"Current suitability: {suitability:.4f}")
        
        # Interpret suitability value
        if suitability > 0.8:
            print("🟢 HIGH SUITABILITY: Path is considered very safe")
        elif suitability > 0.5:
            print("🟡 MEDIUM SUITABILITY: Path has some obstacles but is generally navigable")
        else:
            print("🔴 LOW SUITABILITY: Path is considered risky with significant obstacles")
        
        # Calculate factors that might affect suitability
        distance_to_target = self.distance_to_target()
        
        # Get nearest obstacle distance
        _, distances = self.scanner.fetch_density_distances()
        nearest_obstacle = min(distances) if distances else 100.0
        
        print(f"Factors affecting suitability:")
        print(f"→ Distance to target: {distance_to_target:.2f}m")
        print(f"→ Nearest obstacle: {nearest_obstacle:.2f}m")
        
        # Suggest potential path adjustments
        print("\nPotential path improvements:")
        
        if nearest_obstacle < 3.0:
            print("→ Move away from nearby obstacles")
        
        if suitability < 0.5 and nearest_obstacle > 5.0:
            print("→ Suitability is low despite no very close obstacles - may be influenced by voxel grid obstacles")
            print("→ Consider adjusting suitability calculation weights in Julia code")
        
        print("=============================\n")


def main():
    print("Starting autonomous drone navigation")
    print(f"Target location: {TARGET_LOCATION}")
    
    # Initialize drone controller
    controller = DroneController()
    
    # Initialize ZMQ Interface
    zmq_interface = None
    
    try:
        # First, precompile the Julia components to ensure actinf package is ready
        print("Precompiling Julia components...")
        if not controller.precompile_julia_components():
            print("Failed to precompile Julia components. Cannot continue.")
            return
        
        # Reset AirSim and take off
        controller.reset_and_takeoff()
        
        # Initialize ZMQ interface for communication with Julia server
        print("Initializing ZeroMQ interface...")
        zmq_interface = ZMQInterface()
        
        iteration = 0
        while iteration < MAX_ITERATIONS:
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Check if we've reached the target
            distance_to_target = controller.distance_to_target()
            print(f"Distance to target: {distance_to_target:.2f}m")
            if distance_to_target < ARRIVAL_THRESHOLD:
                print("🎯 Target reached! Mission complete.")
                break
            
            # 1. Collect sensor data
            observation = controller.collect_sensor_data()
            
            # 2. Send observation to Julia server via ZMQ and receive expected state and action
            print("Processing observation...")
            expected_state, egocentric_waypoint = zmq_interface.send_observation_and_receive_action(observation)
            
            # Debug: Print detailed belief state information
            if expected_state:
                print("\n==== DRONE BELIEF STATE ====")
                print(f"🧠 Distance belief: {expected_state.get('distance', 'N/A'):.2f}m")
                print(f"🧠 Azimuth: {expected_state.get('azimuth', 'N/A'):.2f} rad ({round(np.rad2deg(expected_state.get('azimuth', 0)), 1)}°)")
                print(f"🧠 Elevation: {expected_state.get('elevation', 'N/A'):.2f} rad ({round(np.rad2deg(expected_state.get('elevation', 0)), 1)}°)")
                print(f"🧠 Path suitability: {expected_state.get('suitability', 'N/A'):.2f}")
                print("==========================\n")
                
                # Analyze suitability metric to understand path planning decisions
                controller.inspect_suitability_metric(expected_state)
            
            # Debug: Print waypoint decision
            print(f"🧭 Planned egocentric waypoint: {[round(coord, 2) for coord in egocentric_waypoint]}")
            
            # If ZMQ communication failed, use local fallback (create a minimal action toward target)
            if expected_state is None:
                print("⚠️ Communication failed, using fallback")
                # Generate a simple direct vector to target as fallback
                direction = [
                    controller.target_location[0] - controller.current_position[0],
                    controller.target_location[1] - controller.current_position[1],
                    controller.target_location[2] - controller.current_position[2]
                ]
                
                # Normalize to 0.5m step (more cautious default)
                magnitude = sqrt(sum(d*d for d in direction))
                if magnitude > 0:
                    egocentric_waypoint = [d * 0.5 / magnitude for d in direction]
                else:
                    egocentric_waypoint = [0.0, 0.0, 0.0]  # Hover in place
            
            # 3. Convert to global coordinates and move the drone
            global_waypoint = controller.convert_to_global_waypoint(egocentric_waypoint)
            
            # Save the planned waypoint for post-movement analysis
            planned_waypoint = global_waypoint.copy()
            
            movement_result = controller.move_to_waypoint(global_waypoint)
            
            # Analyze movement accuracy
            print("\n==== MOVEMENT ANALYSIS ====")
            actual_position = controller.current_position
            print(f"Planned waypoint: {[round(w, 2) for w in planned_waypoint]}")
            print(f"Actual position: {[round(p, 2) for p in actual_position]}")
            
            # Calculate movement error
            movement_error = sqrt(sum((a - b) ** 2 for a, b in zip(actual_position, planned_waypoint)))
            print(f"Movement error: {movement_error:.2f}m")
            
            if movement_error > 1.0:
                print("⚠️ High movement error - potential navigation issue")
            
            # Analyze target progress
            distance_to_target = controller.distance_to_target()
            print(f"Current distance to target: {distance_to_target:.2f}m")
            print("===========================\n")
            
            # Check if an obstacle was detected during movement
            if movement_result == "obstacle_detected":
                print("Replanning due to obstacle...")
                # No need to increment iteration - we'll retry with a fresh plan
                continue
            elif movement_result == "error":
                print("Error during movement. Replanning...")
                # Brief pause to stabilize before replanning
                time.sleep(1)
                continue
            
            iteration += 1
        
        if iteration >= MAX_ITERATIONS:
            print("Maximum iterations reached without finding target.")
        
    finally:
        # Cleanup ZMQ connection if it was initialized
        if zmq_interface is not None:
            zmq_interface.close()
        
        # Land the drone
        print("Landing drone...")
        controller.client.landAsync().join()
        controller.client.armDisarm(False)
        controller.client.enableApiControl(False)
        print("Test completed.")


if __name__ == "__main__":
    main()
