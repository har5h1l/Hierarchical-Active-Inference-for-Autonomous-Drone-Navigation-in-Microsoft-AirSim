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
import socket
import traceback
import math
import random
from matplotlib import pyplot as plt
import platform

# Custom NumPy-aware JSON encoder
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types"""
    def default(self, obj):
        # Handle specific NumPy types (avoiding deprecated type names)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle any other NumPy type not explicitly listed above
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            return obj.item()
        
        # Let the parent class handle it or raise TypeError
        return super(NumpyJSONEncoder, self).default(obj)

# Constants and hyperparameters
TARGET_LOCATION = [-20.0, -20.0, -30.0]  # [x, y, z] in NED coordinates
MARGIN = 1.5  # Safety margin for waypoint generation (meters)
WAYPOINT_SAMPLE_COUNT = 75  # Number of waypoints to consider
POLICY_LENGTH = 3  # Number of steps in the policy
DENSITY_RADIUS = 5.0  # Radius for density evaluation
ARRIVAL_THRESHOLD = 1.2  # meters
MAX_ITERATIONS = 100
INTERFACE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "interface"))
JULIA_PATH = "julia --project=."  # Path to Julia executable with project environment
# Alternative ports in case the primary port is unavailable
ZMQ_PORTS = [5555, 5556, 5557, 5558]  # Port 5555 is default in zmq_server.jl
ZMQ_SERVER_ADDRESS = f"tcp://localhost:{5555}"  # ZeroMQ server address with the first port
ZMQ_TIMEOUT = 10000  # ZMQ socket timeout in milliseconds (15 seconds)
ZMQ_MAX_RETRIES = 3  # Maximum number of retries for ZMQ communication
DEFAULT_ZMQ_PORT = 5555  # Use a single port instead of trying multiple ports

# Add the airsim directory to the Python path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'airsim'))

# Import the scanner with the correct filename
try:
    from Sensory_Input_Processing import EnvironmentScanner
except ImportError:
    # Try with the original filename if renamed file not found
    from airsim.Sensory_Input_Processing import EnvironmentScanner

# ZMQInterface class definition
class ZMQInterface:
    """Interface for communicating with the Julia Active Inference server via ZMQ"""
    
    def __init__(self, server_address=None, defer_connection=False):
        """Initialize ZMQ connection to Julia server
        
        Args:
            server_address: ZMQ server address (default: None, will use DEFAULT_ZMQ_PORT)
            defer_connection: If True, don't attempt to establish connection immediately
        """
        # Use the default port for consistent connection
        self.server_address = server_address if server_address else f"tcp://localhost:{DEFAULT_ZMQ_PORT}"
        print(f"Initializing ZMQ interface for: {self.server_address}")
        
        # Initialize context and socket variables
        self.context = None
        self.socket = None
        
        # Set up the connection if not deferred
        if not defer_connection:
            self._setup_zmq_connection()
        else:
            print("Connection deferred - call _setup_zmq_connection() when ready")
    
    def _setup_zmq_connection(self):
        """Set up the ZMQ connection with appropriate timeouts and configuration"""
        try:
            # Close any existing connections first
            if hasattr(self, 'socket') and self.socket:
                self.socket.close()
            if hasattr(self, 'context') and self.context:
                self.context.term()
            
            # Create new context and socket
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 1000)  # Increased linger time
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT)  # Use global timeout
            self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT)  # Use global timeout
            
            # Connect with proper error handling
            try:
                print(f"Connecting to ZMQ server at {self.server_address}...")
                self.socket.connect(self.server_address)
                print(f"Connected to ZMQ server at {self.server_address}")
                
                # Test connection (simplified)
                connection_result = self._test_connection()
                if connection_result:
                    print("✅ ZMQ socket connected successfully")
                else:
                    print("⚠️ ZMQ connection test failed, but continuing anyway")
                
                return True
            except zmq.ZMQError as e:
                print(f"Failed to connect to ZMQ server at {self.server_address}: {str(e)}")
                if self.socket:
                    self.socket.close()
                    self.socket = None
                return False
        
        except Exception as e:
            print(f"Failed to setup ZMQ connection: {str(e)}")
            traceback.print_exc()
            self.socket = None
            self.context = None
            return False
    
    def _test_connection(self):
        """Test the ZMQ connection with a simple ping-pong exchange
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            # Try to send a simple ping message and receive a pong response
            print("Testing ZMQ connection with ping...")
            
            # Set a short timeout just for the ping test
            self.socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout for ping
            
            # Send ping message
            try:
                self.socket.send_string("ping", flags=zmq.NOBLOCK)
            except zmq.ZMQError as e:
                print(f"Failed to send ping: {str(e)}")
                # Restore original timeout
                self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT)
                return False
                
            # Try to receive response with polling
            poller = zmq.Poller()
            poller.register(self.socket, zmq.POLLIN)
            
            if poller.poll(2000):  # 2 second timeout
                try:
                    response = self.socket.recv_string()
                    # Restore original timeout
                    self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT)
                    
                    if response == "pong":
                        print("✅ ZMQ connection test successful (ping-pong)")
                        return True
                    else:
                        print(f"⚠️ Unexpected response to ping: {response}")
                        return False
                except zmq.ZMQError as e:
                    print(f"Failed to receive pong: {str(e)}")
                    return False
            else:
                print("⚠️ Timeout waiting for pong response")
                # Restore original timeout
                self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT)
                return False
                
        except Exception as e:
            print(f"⚠️ ZMQ connection test failed: {str(e)}")
            return False
    
    def _reset_socket(self):
        """Reset and reconnect the ZMQ socket"""
        print("Resetting ZMQ socket...")
        
        try:
            # Close existing socket if any
            if hasattr(self, 'socket') and self.socket is not None:
                try:
                    self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait when closing
                    self.socket.close()
                except Exception as e:
                    print(f"Warning when closing socket: {str(e)}")
                self.socket = None
                print("Closed existing socket")
            
            # Terminate existing context if any
            if hasattr(self, 'context') and self.context is not None:
                try:
                    self.context.term()
                except Exception as e:
                    print(f"Warning when terminating context: {str(e)}")
                self.context = None
                print("Terminated ZMQ context")
            
            # Brief pause to ensure resources are released
            time.sleep(0.5)
            
            # Create new context and socket
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            
            # Configure socket options
            self.socket.setsockopt(zmq.LINGER, 1000)   # Wait up to 1000ms on close
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT)  # Set receive timeout
            self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT)  # Set send timeout
            self.socket.setsockopt(zmq.REQ_RELAXED, 1)  # More relaxed REQ socket behavior
            self.socket.setsockopt(zmq.REQ_CORRELATE, 1)  # Correlate replies with requests
            
            # Connect to server
            print(f"Connecting to server at {self.server_address}")
            self.socket.connect(self.server_address)
            
            print("Socket reset complete")
            return True
            
        except Exception as e:
            print(f"❌ Error during socket reset: {str(e)}")
            traceback.print_exc()
            # Make sure we don't have dangling references
            self.socket = None
            self.context = None
            return False

    def _sanitize_for_json(self, obj):
        """
        Recursively sanitize an object for JSON serialization, converting NumPy types to native Python types.
        
        Args:
            obj: Any Python object to sanitize
            
        Returns:
            JSON-serializable version of the object
        """
        # Handle None
        if obj is None:
            return None
            
        # Handle NumPy scalars (convert to native Python types)
        # Use exact type names instead of np.float_ which is deprecated in NumPy 2.0
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._sanitize_for_json(obj.tolist())
            
        # Handle lists, tuples and sets
        elif isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(item) for item in obj]
            
        # Handle dictionaries
        elif isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
            
        # Handle NumPy arrays not caught above
        elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
            return self._sanitize_for_json(obj.tolist())
            
        # Let JSON serializer handle other types or fail with clear error
        return obj

    def _is_server_running(self):
        """Check if the Julia ZMQ server is actually running and listening
        
        Returns:
            bool: True if server is confirmed running, False otherwise
        """
        # First check if there's a Julia process running the ZMQ server
        julia_running = False
        
        try:
            # Platform-specific process checking
            if platform.system() == "Windows":
                # On Windows, use more reliable methods to check for Julia processes
                try:
                    # First try with wmic for more detailed information
                    wmic_cmd = "wmic process where name='julia.exe' get commandline"
                    result = subprocess.run(wmic_cmd, shell=True, capture_output=True, text=True)
                    
                    if "zmq_server.jl" in result.stdout:
                        print("✅ Julia ZMQ server process found running on Windows (wmic)")
                        julia_running = True
                    elif "julia.exe" in result.stdout:
                        print("⚠️ Julia process found but can't confirm if it's running ZMQ server (wmic)")
                        julia_running = True  # Assume it might be our server
                    else:
                        # Fallback to tasklist which is more reliable on some Windows versions
                        result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq julia.exe"], 
                                               capture_output=True, text=True)
                        if "julia.exe" in result.stdout.lower():
                            print("✅ Julia process found running on Windows (tasklist)")
                            julia_running = True
                        else:
                            # Additional check with netstat to see if something is using our port
                            result = subprocess.run(
                                ["netstat", "-ano", "|", "findstr", ":5555"],
                                shell=True, capture_output=True, text=True)
                            if "LISTENING" in result.stdout:
                                print("⚠️ Something is listening on port 5555 but couldn't confirm if it's Julia")
                                julia_running = True  # Assume it might be our server
                            else:
                                print("❌ No Julia process found running on Windows")
                                return False
                except Exception as e:
                    print(f"Error checking for process on Windows: {str(e)}")
                    # Continue with socket check as fallback
            else:
                # macOS/Linux approach using ps
                result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
                if "julia" in result.stdout and "zmq_server.jl" in result.stdout:
                    print("✅ Julia ZMQ server process found running")
                    julia_running = True
                else:
                    # Check for just Julia - it might be running the server without explicit cmdline
                    if "julia" in result.stdout:
                        print("⚠️ Julia process found but can't confirm if it's running ZMQ server")
                        julia_running = True  # Assume it might be our server
                    else:
                        print("❌ No Julia process found running")
                        return False
        except Exception as e:
            print(f"Error checking for Julia process: {str(e)}")
            # Continue with socket check as fallback
        
        # Next check if there's a status file indicating server is running - platform independent
        status_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zmq_server_running.status")
        if os.path.exists(status_file_path):
            # Check if the file was created recently (within last 5 minutes)
            try:
                file_age = time.time() - os.path.getmtime(status_file_path)
                if file_age < 300:  # 5 minutes in seconds
                    print(f"✅ Found recent ZMQ server status file (age: {file_age:.1f}s)")
                    return True
                else:
                    print(f"⚠️ Found stale ZMQ server status file (age: {file_age:.1f}s)")
                    # Don't return yet, continue with other checks
            except Exception:
                print("✅ Found ZMQ server status file but couldn't check age")
                return True
                
        # If Julia is running but we need to verify server responsiveness, try a socket connection
        if julia_running:
            if hasattr(self, 'socket') and self.socket is not None:
                try:
                    # Check socket status in platform-independent way
                    events = self.socket.getsockopt(zmq.EVENTS)
                    if events & zmq.POLLOUT:
                        print("✅ ZMQ socket appears to be in valid state")
                        return True
                    else:
                        print("⚠️ ZMQ socket exists but may not be usable")
                except zmq.ZMQError:
                    print("⚠️ Error checking socket state")
            
            # Try to create a test socket to see if the server is listening
            try:
                test_context = zmq.Context()
                test_socket = test_context.socket(zmq.REQ)
                test_socket.setsockopt(zmq.LINGER, 0)
                test_socket.setsockopt(zmq.RCVTIMEO, 500)  # Short timeout for quick check
                test_socket.connect(self.server_address)
                
                # On Windows, try more aggressive connection test
                if platform.system() == "Windows":
                    try:
                        # Send a ping with short timeout
                        test_socket.send_string("ping", flags=zmq.NOBLOCK)
                        poller = zmq.Poller()
                        poller.register(test_socket, zmq.POLLIN)
                        if poller.poll(1000):  # 1 second timeout
                            response = test_socket.recv_string()
                            if response == "pong":
                                print("✅ Successfully received pong from server")
                                test_socket.close()
                                test_context.term()
                                return True
                    except Exception:
                        # Fall through to basic connection success
                        pass
                
                # Just connecting succeeded
                test_socket.close()
                test_context.term()
                print("✅ Successfully connected test socket to server")
                return True
            except Exception as e:
                print(f"⚠️ Could not connect test socket: {e}")
                if 'test_socket' in locals():
                    test_socket.close()
                if 'test_context' in locals():
                    test_context.term()
            
            # If we get here with Julia running, give it the benefit of doubt
            # The server might still be initializing
            print("⚠️ Julia is running but server might not be ready yet")
            return julia_running
            
        # All checks failed
        return False

    def _start_server(self):
        """Start the Julia ZMQ server with improved reliability and platform compatibility
        
        Returns:
            bool: True if server was started successfully, False otherwise
        """
        print("\n==== Starting Julia ZMQ Server ====")
        
        # First verify if server is already running (don't restart unnecessarily)
        if self._is_server_running():
            print("Julia server appears to be already running")
            return True
        
        # Find the server script path - ensure Windows compatibility with path handling
        server_script_path = None
        possible_paths = [
            "zmq_server.jl",  # Current directory
            os.path.join("actinf", "zmq_server.jl")  # actinf subdirectory
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                server_script_path = path
                print(f"Found server script at: {server_script_path}")
                break
                
        if server_script_path is None:
            print("❌ Could not find zmq_server.jl in any expected location")
            return False
        
        # Ensure we're using the correct port
        port = self.server_address.split(":")[-1]
        if port != "5555":
            print(f"Warning: Julia server is configured to use port 5555, but current address uses {port}")
            print("Switching to port 5555 for compatibility")
            self.server_address = "tcp://localhost:5555"
        
        # Kill any existing Julia processes that might block the port - platform-specific
        try:
            if platform.system() == "Windows":
                # Windows-specific commands for process termination
                # First try to kill any Julia process specifically running zmq_server.jl
                subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq *zmq_server*"], 
                              capture_output=True, text=True)
                # Then kill any remaining Julia processes
                subprocess.run(["taskkill", "/F", "/IM", "julia.exe"], 
                              capture_output=True, text=True)
                print("Cleaned up any existing Julia processes on Windows")
            else:
                # Unix/macOS-specific commands
                subprocess.run(["pkill", "-f", "julia"], 
                              capture_output=True, text=True)
                print("Cleaned up any existing Julia processes on macOS/Linux")
            time.sleep(2)  # Wait for processes to terminate
        except Exception as e:
            print(f"Note: Could not kill existing Julia processes: {str(e)}")
        
        # Find Julia executable with platform-specific paths
        julia_path = "julia"  # Default command in PATH
        
        if platform.system() == "Windows":
            # Windows-specific paths - check more common installation locations
            possible_julia_paths = [
                r"C:\Julia-1.9.3\bin\julia.exe",
                r"C:\Julia-1.9.2\bin\julia.exe",
                r"C:\Julia-1.9.1\bin\julia.exe",
                r"C:\Julia-1.9.0\bin\julia.exe",
                r"C:\Julia-1.8.5\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.3\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.2\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.1\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.0\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.8.5\bin\julia.exe"
            ]
            
            # Expand %USERNAME% environment variable in Windows paths
            username = os.environ.get('USERNAME', '')
            possible_julia_paths = [path.replace('%USERNAME%', username) for path in possible_julia_paths]
            
            for path in possible_julia_paths:
                if os.path.exists(path):
                    julia_path = path
                    print(f"Found Julia at: {julia_path}")
                    break
        elif platform.system() == "Darwin":  # macOS
            # Check common macOS Julia locations
            possible_julia_paths = [
                "/Applications/Julia-1.9.app/Contents/Resources/julia/bin/julia",
                "/Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia",
                "/usr/local/bin/julia"
            ]
            for path in possible_julia_paths:
                if os.path.exists(path):
                    julia_path = path
                    print(f"Found Julia at: {julia_path}")
                    break
        
        # Check if Julia is available
        try:
            version_result = subprocess.run([julia_path, "--version"], 
                                           capture_output=True, text=True, timeout=5)
            if version_result.returncode == 0:
                julia_version = version_result.stdout.strip()
                print(f"Using Julia: {julia_version}")
            else:
                print("⚠️ Julia version check failed but continuing anyway")
        except Exception as e:
            print(f"⚠️ Could not verify Julia installation: {str(e)}")
            print("Make sure Julia is installed and in your PATH")
            return False
        
        # Build the command with project flag - consistent across platforms
        cmd = [julia_path, "--project=.", server_script_path]
        print(f"Executing: {' '.join(cmd)}")
        
        # Get current working directory for subprocess
        cwd = os.path.dirname(os.path.abspath(__file__))
        print(f"Working directory: {cwd}")
        
        # Create a status file path to check if server is running - platform independent
        status_file_path = os.path.join(cwd, "zmq_server_running.status")
        
        # Remove any existing status file
        if os.path.exists(status_file_path):
            try:
                os.remove(status_file_path)
                print("Removed existing status file")
            except Exception as e:
                print(f"Could not remove existing status file: {str(e)}")
        
        # Start the server based on platform
        if platform.system() == "Windows":
            try:
                # On Windows, try to create a new console window
                # Use CREATE_NEW_CONSOLE flag to create a separate window that stays open
                print("Starting Julia server in a new console window (Windows)...")
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 1  # SW_SHOWNORMAL
                
                server_process = subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    cwd=cwd,
                    startupinfo=startupinfo
                )
                print("Windows server process started with PID:", server_process.pid)
            except Exception as e:
                print(f"Error starting with new console: {str(e)}")
                print("Falling back to standard process method...")
                # Fallback to standard process but still detached
                try:
                    # Use DETACHED_PROCESS to keep the process running independently
                    server_process = subprocess.Popen(
                        cmd,
                        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                        stdout=open(os.path.join(cwd, "julia_zmq_server.log"), "w"),
                        stderr=open(os.path.join(cwd, "julia_zmq_server_error.log"), "w"),
                        cwd=cwd
                    )
                    print(f"Windows server started with detached process, PID: {server_process.pid}")
                except Exception as e2:
                    print(f"Second attempt failed: {str(e2)}")
                    # Last resort: basic process
                    server_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=cwd
                    )
                    print("Windows server started with basic subprocess")
        else:
            # For macOS/Linux, use nohup to keep server running even if Python exits
            try:
                # First try with nohup to ensure server stays running
                nohup_cmd = ["nohup"] + cmd
                with open(os.path.join(cwd, "julia_zmq_server.log"), "w") as log_file:
                    server_process = subprocess.Popen(
                        nohup_cmd,
                        stdout=log_file,
                        stderr=log_file,
                        cwd=cwd,
                        preexec_fn=os.setpgrp  # Create new process group to avoid signals
                    )
                print(f"Started Julia server with nohup, logging to julia_zmq_server.log")
            except Exception as e:
                print(f"Error starting with nohup: {str(e)}")
                # Fallback to standard approach
                server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd
                )
                print("Started Julia server with standard subprocess")
        
        # Wait for server initialization with progress indicator - platform independent
        print(f"Waiting up to 45 seconds for server to initialize...")
        for i in range(45):  # Increased timeout for Windows
            print(f"Starting Julia server: {i+1}/45 seconds elapsed...", end="\r")
            
            # Check for the status file which indicates server is running
            if os.path.exists(status_file_path):
                print(f"\n✅ Server status file found at {status_file_path}")
                print("ZMQ server is confirmed running")
                return True
                
            # Check if process has exited early (error)
            if server_process.poll() is not None:
                print("\n❌ Server process exited prematurely")
                break
                
            # Check if socket becomes available
            if i > 15:  # Give at least 15 seconds before checking socket (Windows might be slower)
                try:
                    # Try creating a test socket to see if server is listening
                    test_context = zmq.Context()
                    test_socket = test_context.socket(zmq.REQ)
                    test_socket.setsockopt(zmq.LINGER, 0)
                    test_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
                    test_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second timeout
                    test_socket.connect(self.server_address)
                    
                    # Try sending a simple message - don't wait for response
                    try:
                        test_socket.send_string("ping", flags=zmq.NOBLOCK)
                    except:
                        pass
                        
                    # If we get here, the socket connected
                    print("\n✅ Successfully connected test socket - server is listening")
                    test_socket.close()
                    test_context.term()
                    return True
                except Exception:
                    # Still not ready, continue waiting
                    if 'test_socket' in locals():
                        test_socket.close()
                    if 'test_context' in locals():
                        test_context.term()
            
            time.sleep(1)
        
        print("\nChecking final server status...")
        
        # Final check if process is still running
        if server_process.poll() is None:
            print("✅ Server process is still running")
            
            # Try to connect one more time
            try:
                test_context = zmq.Context()
                test_socket = test_context.socket(zmq.REQ)
                test_socket.setsockopt(zmq.LINGER, 0)
                test_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
                test_socket.connect(self.server_address)
                test_socket.close()
                test_context.term()
                print("✅ Final connection test successful - server is listening")
                return True
            except Exception:
                print("⚠️ Process is running but socket connection failed")
                # Assume server might still be initializing
                return True
        else:
            # Process has terminated, get output if available
            try:
                stdout, stderr = server_process.communicate(timeout=1)
                print(f"❌ Server process failed to start (exit code: {server_process.returncode})")
                if stdout:
                    stdout_str = stdout.decode('utf-8', errors='replace')
                    print(f"Server stdout excerpt: {stdout_str[:500]}")
                if stderr:
                    stderr_str = stderr.decode('utf-8', errors='replace')
                    print(f"Server stderr excerpt: {stderr_str[:500]}")
                    
                    # Check for common errors
                    if "cannot bind" in stderr_str and "Address already in use" in stderr_str:
                        print("⚠️ Port 5555 is already in use. Trying to kill existing processes...")
                        # Make a more aggressive attempt to kill processes using the port
                        if platform.system() == "Windows":
                            # Check ports on Windows and kill processes
                            try:
                                # Find PID using the port with netstat
                                netstat = subprocess.run(
                                    ["netstat", "-ano", "|", "findstr", ":5555"], 
                                    shell=True, capture_output=True, text=True)
                                lines = netstat.stdout.strip().split("\n")
                                for line in lines:
                                    if "LISTENING" in line:
                                        parts = line.split()
                                        if len(parts) > 4:
                                            pid = parts[-1]
                                            print(f"Found process {pid} using port 5555, killing...")
                                            subprocess.run(["taskkill", "/F", "/PID", pid], 
                                                           capture_output=True, text=True)
                            except Exception as e:
                                print(f"Failed to find/kill process using port: {str(e)}")
                                
                            # Kill all julia.exe processes as a last resort
                            subprocess.run(["taskkill", "/F", "/IM", "julia.exe"], 
                                           capture_output=True, text=True)
                        else:
                            # Linux/macOS approach
                            subprocess.run(["lsof", "-i", ":5555"], capture_output=True, text=True)
                            subprocess.run(["pkill", "-f", "julia"], capture_output=True, text=True)
                        
                        time.sleep(2)
                        # Try starting again after killing
                        return self._start_server()
            except Exception as e:
                print(f"Error getting server output: {str(e)}")
        
        print("❌ Failed to start ZMQ server")
        return False

# Function to initialize ZMQ with Julia precompilation
def initialize_zmq_with_precompilation():
    """Initialize the ZMQ interface after running Julia precompilation
    
    This ensures Julia components are properly compiled before any communication
    starts, preventing timeouts and connection issues.
    
    Returns:
        ZMQInterface: The initialized ZMQ interface
    """
    print("\n==== Running Julia Precompilation ====")
    precompile_success = False
    
    # Get current working directory for consistent path handling on all platforms
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Find Julia executable with better platform-specific handling
        julia_path = "julia"  # Default command in PATH
        
        if platform.system() == "Windows":
            # Windows-specific Julia paths
            possible_julia_paths = [
                r"C:\Julia-1.9.3\bin\julia.exe",
                r"C:\Julia-1.9.2\bin\julia.exe",
                r"C:\Julia-1.9.1\bin\julia.exe",
                r"C:\Julia-1.9.0\bin\julia.exe",
                r"C:\Julia-1.8.5\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.3\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.2\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.1\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.0\bin\julia.exe"
            ]
            
            # Expand %USERNAME% environment variable in Windows paths
            username = os.environ.get('USERNAME', '')
            possible_julia_paths = [path.replace('%USERNAME%', username) for path in possible_julia_paths]
            
            # Try each path
            for path in possible_julia_paths:
                if os.path.exists(path):
                    julia_path = path
                    print(f"Found Julia at: {julia_path}")
                    break
        
        # Run precompilation script to prepare Julia environment with timeout
        print("Starting Julia precompilation (this may take several minutes)...")
        print("Precompiling RxInfer and Active Inference package dependencies...")
        
        # Verify precompile.jl exists
        precompile_script = os.path.join(cwd, "precompile.jl")
        if not os.path.exists(precompile_script):
            print(f"⚠️ Precompilation script not found at {precompile_script}")
            print("Will proceed without precompilation")
        else:
            # Add a reasonable timeout for precompilation (10 minutes)
            precompile_timeout = 600  # seconds
            
            try:
                # Use platform-specific precompilation approach
                if platform.system() == "Windows":
                    print(f"Running Windows-optimized precompilation with {precompile_timeout}s timeout...")
                    
                    # On Windows, create a log file for capturing output
                    precompile_log = os.path.join(cwd, "julia_precompile.log")
                    
                    # First kill any existing Julia processes that might interfere
                    try:
                        subprocess.run(["taskkill", "/F", "/IM", "julia.exe"], 
                                     capture_output=True, text=True)
                        time.sleep(1)  # Give time for processes to terminate
                    except Exception as e:
                        print(f"Note: Could not kill existing Julia processes: {str(e)}")
                    
                    # Use different approach on Windows since timeout handling works differently
                    try:
                        # Create a detached process with window showing status
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = 1  # SW_SHOWNORMAL
                        
                        # Start with separate window to make it visible
                        julia_cmd = [julia_path, precompile_script]
                        print(f"Executing: {' '.join(julia_cmd)}")
                        
                        with open(precompile_log, "w") as log_file:
                            precompile_process = subprocess.Popen(
                                julia_cmd,
                                stdout=log_file,
                                stderr=log_file,
                                cwd=cwd,
                                creationflags=subprocess.CREATE_NEW_CONSOLE
                            )
                        
                        # Wait for precompilation with timeout
                        print(f"Waiting for precompilation to complete (timeout: {precompile_timeout}s)...")
                        
                        # Wait with timeout
                        try:
                            return_code = precompile_process.wait(timeout=precompile_timeout)
                            if return_code == 0:
                                print("✅ Julia precompilation completed successfully")
                                precompile_success = True
                            else:
                                print(f"⚠️ Precompilation returned non-zero exit code: {return_code}")
                        except subprocess.TimeoutExpired:
                            print(f"⚠️ Precompilation timed out after {precompile_timeout} seconds")
                            # Try to terminate the process
                            try:
                                precompile_process.terminate()
                                time.sleep(1)
                                if precompile_process.poll() is None:
                                    precompile_process.kill()
                            except Exception as e:
                                print(f"Error terminating precompilation process: {e}")
                        
                        # Check precompile log file
                        if os.path.exists(precompile_log):
                            try:
                                with open(precompile_log, 'r') as f:
                                    log_content = f.read()
                                    last_lines = log_content.strip().split('\n')[-5:]
                                    print("Last few lines of precompilation log:")
                                    for line in last_lines:
                                        print(f"  {line}")
                            except Exception as e:
                                print(f"Error reading precompile log: {e}")
                    except Exception as e:
                        print(f"Error during Windows precompilation: {e}")
                        traceback.print_exc()
                else:
                    # macOS/Linux approach with simpler subprocess.run
                    julia_cmd = [julia_path, precompile_script]
                    print(f"Running precompilation with {precompile_timeout}s timeout...")
                    result = subprocess.run(
                        julia_cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=precompile_timeout
                    )
                    
                    if result.returncode == 0:
                        print("✅ Julia precompilation completed successfully:")
                        # Get the last few lines for better confirmation
                        output_lines = result.stdout.strip().split("\n")
                        for line in output_lines[-3:]:  # Print last 3 lines
                            if line.strip():  # Only print non-empty lines
                                print(f"  {line}")
                        precompile_success = True
                    else:
                        print(f"⚠️ Julia precompilation returned code {result.returncode}:")
                        print(result.stderr[:500])  # Print first 500 chars of error
                        print("Attempting to continue despite precompilation issues...")
            except subprocess.TimeoutExpired:
                print(f"⚠️ Julia precompilation timed out after {precompile_timeout} seconds")
                print("Proceeding without completed precompilation")
                # Try to terminate any hanging Julia process
                if platform.system() != "Windows":
                    # For macOS
                    subprocess.run(["pkill", "-f", "julia precompile.jl"], 
                                  capture_output=True, text=True)
                else:
                    # For Windows - try to kill any julia.exe processes
                    subprocess.run(["taskkill", "/F", "/IM", "julia.exe"], 
                                   capture_output=True, text=True)

    except FileNotFoundError:
        print("⚠️ Julia executable not found in PATH")
        print("Make sure Julia is installed and accessible in the system PATH")
        print("Continuing without precompilation, connectivity may be affected")
    except Exception as e:
        print(f"⚠️ Error during precompilation setup: {str(e)}")
        traceback.print_exc()
        print("Continuing without precompilation")
        
    if not precompile_success:
        print("\nAttempting to start ZMQ server directly...")
        # Try to start ZMQ server directly if precompilation failed
        server_script_path = os.path.join("actinf", "zmq_server.jl")
        if os.path.exists(server_script_path):
            try:
                print("Starting Julia ZMQ server...")
                cmd = [julia_path, "--project=.", server_script_path]
                
                # Platform-specific server startup
                if platform.system() == "Windows":
                    print("Starting ZMQ server with Windows-specific configuration...")
                    
                    # On Windows, use CREATE_NEW_CONSOLE for visibility and proper detachment
                    try:
                        # Use startupinfo to control window visibility
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = 1  # SW_SHOWNORMAL
                        
                        server_process = subprocess.Popen(
                            cmd,
                            creationflags=subprocess.CREATE_NEW_CONSOLE,
                            cwd=cwd,
                            startupinfo=startupinfo
                        )
                        print(f"Windows ZMQ server started, PID: {server_process.pid}")
                    except Exception as e:
                        print(f"Error starting server with new console: {e}")
                        # Fallback to hidden process
                        server_logfile = os.path.join(cwd, "julia_zmq_server.log")
                        server_errfile = os.path.join(cwd, "julia_zmq_server_error.log")
                        
                        with open(server_logfile, "w") as log_file, open(server_errfile, "w") as err_file:
                            server_process = subprocess.Popen(
                                cmd,
                                stdout=log_file,
                                stderr=err_file,
                                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                                cwd=cwd
                            )
                        print(f"ZMQ server started with detached process, logging to {server_logfile}")
                else:
                    # For macOS/Linux use nohup for proper detachment
                    nohup_cmd = ["nohup"] + cmd
                    with open(os.path.join(cwd, "julia_zmq_server.log"), "w") as log_file:
                        server_process = subprocess.Popen(
                            nohup_cmd,
                            stdout=log_file,
                            stderr=log_file,
                            cwd=cwd,
                            preexec_fn=os.setpgrp  # Create new process group to avoid signals
                        )
                    
                print("ZMQ server started - waiting for initialization")
                print("Giving the server time to initialize and load packages...")
                time.sleep(20 if platform.system() == "Windows" else 15)  # Longer wait time for Windows
                print("Proceeding with ZMQ interface creation")
            except Exception as e:
                print(f"Error starting server: {str(e)}")
                traceback.print_exc()
    else:
        print("\nPrecompilation successful - Ready to connect to ZMQ server")
    
    # Create and return ZMQ interface with proper platform-specific settings
    return ZMQInterface()

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Scanner for obstacle detection
class Scanner:
    def __init__(self, client, scan_range=20.0, voxel_size=1.0):
        self.client = client
        self.scan_range = scan_range
        self.voxel_size = voxel_size

    def fetch_density_distances(self):
        """Get obstacle positions and distances with orientation-aware transformation"""
        try:
            # Initialize empty lists for return values
            obstacle_positions = []
            obstacle_distances = []

            # Get drone state including position and orientation
            drone_state = self.client.getMultirotorState().kinematics_estimated
            drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
            drone_orientation = [
                drone_state.orientation.w_val,
                drone_state.orientation.x_val, 
                drone_state.orientation.y_val,
                drone_state.orientation.z_val
            ]
            
            # Validate drone position
            if any(np.isnan(x) or np.isinf(x) for x in drone_pos):
                print("⚠️ Invalid drone position detected, using default [0,0,0]")
                drone_pos = [0.0, 0.0, 0.0]
            
            # Get lidar data
            lidar_data = self.client.getLidarData()
            
            if len(lidar_data.point_cloud) < 3:
                print("No lidar points detected")
                return [], []  # Always return empty lists
            
            # Convert point cloud to positions
            try:
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error reshaping point cloud: {str(e)}")
                return [], []  # Always return empty lists

            # Create quaternion rotation functions for obstacle position transformation
            def quaternion_multiply(q1, q2):
                w1, x1, y1, z1 = q1
                w2, x2, y2, z2 = q2
                w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                return [w, x, y, z]
            
            def quaternion_conjugate(q):
                w, x, y, z = q
                return [w, -x, -y, -z]
            
            def rotate_point_by_quaternion(point, quaternion):
                # Convert point to quaternion form (0 + xi + yj + zk)
                point_quat = [0.0, point[0], point[1], point[2]]
                
                # Get quaternion conjugate
                q_conj = quaternion_conjugate(quaternion)
                
                # Rotate using q * p * q^-1
                rotated = quaternion_multiply(
                    quaternion_multiply(quaternion, point_quat),
                    q_conj
                )
                
                # Return the vector part
                return rotated[1:4]

            # Process each point in the point cloud
            for point in points:
                # Skip invalid points
                if np.isnan(point).any() or np.isinf(point).any():
                    continue
                
                try:
                    # Transform point to global reference frame based on drone orientation
                    global_point = rotate_point_by_quaternion(point, drone_orientation)
                    
                    # Add drone position to get absolute position in world
                    global_point = [
                        global_point[0] + drone_pos[0],
                        global_point[1] + drone_pos[1],
                        global_point[2] + drone_pos[2]
                    ]
                    
                    # Calculate distance to drone
                    dist = np.sqrt(np.sum((np.array(drone_pos) - np.array(global_point)) ** 2))
                    
                    # Skip invalid distance calculations
                    if np.isnan(dist) or np.isinf(dist):
                        continue
                    
                    # Only include obstacles within scan range
                    if dist <= self.scan_range and dist > 0.0:
                        obstacle_positions.append(global_point)
                        obstacle_distances.append(float(dist))  # Convert to Python float
                except Exception as point_error:
                    # Skip this point if there was an error
                    continue
            
            return obstacle_positions, obstacle_distances  # Always returns lists
        
        except Exception as e:
            print(f"Error in fetch_density_distances: {str(e)}")
            traceback.print_exc()
            return [], []  # Always return empty lists

# Initialize scanner
scanner = Scanner(client)

# Initialize ZMQ interface with proper precompilation
zmq_interface = initialize_zmq_with_precompilation()

def move_to_waypoint(client, current_pos, waypoint, velocity=2):
    """Move to waypoint with calculated yaw to ensure sensors face the direction of travel
    
    Args:
        client: AirSim client
        current_pos: Current drone position [x, y, z]
        waypoint: Target waypoint [x, y, z]
        velocity: Movement velocity in m/s
    """
    # Calculate movement vector and distance
    movement_vector = np.array(waypoint) - np.array(current_pos)
    distance = np.linalg.norm(movement_vector)
    
    # Don't calculate yaw for very small movements
    if distance < 0.1:
        print("Movement distance too small, using current orientation")
        client.moveToPositionAsync(
            waypoint[0], waypoint[1], waypoint[2],
            velocity
        ).join()
        return
    
    # Calculate yaw angle (in radians) to face the direction of movement
    # In NED coordinates, yaw is measured from North (x-axis) positive clockwise
    yaw = math.atan2(movement_vector[1], movement_vector[0])
    
    # Convert to degrees for logging
    yaw_degrees = math.degrees(yaw)
    print(f"Calculated yaw for movement: {yaw_degrees:.1f} degrees")
    
    # Use moveToPositionAsync with yaw control
    client.moveToPositionAsync(
        waypoint[0], waypoint[1], waypoint[2],
        velocity,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_degrees)
    ).join()

# Main navigation process
def main():
    print("\n==== Autonomous Drone Navigation with Active Inference ====")
    print(f"Target location: {TARGET_LOCATION}")
    
    # Use AirSim's built-in takeoff API
    print("Taking off...")
    client.takeoffAsync().join()
    
    # Get post-takeoff position
    drone_state = client.getMultirotorState().kinematics_estimated
    drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
    print(f"Position after takeoff: {[round(p, 2) for p in drone_pos]}")
    
    # If needed, adjust to specific mission altitude
    start_height = -5.0  # 5 meters above ground in NED coordinates (negative Z = up)
    print(f"Adjusting to mission altitude: {-start_height} meters")
    client.moveToZAsync(start_height, 2).join()
    time.sleep(1)
    
    # Get initial drone state after altitude adjustment
    drone_state = client.getMultirotorState().kinematics_estimated
    drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
    print(f"Initial position: {[round(p, 2)

