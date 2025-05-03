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
            
            # Configure socket options
            self.socket.setsockopt(zmq.LINGER, 1000)  # Wait up to 1 second when closing
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT)  # Use global timeout
            self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT)  # Use global timeout
            self.socket.setsockopt(zmq.REQ_RELAXED, 1)  # More relaxed REQ socket behavior
            self.socket.setsockopt(zmq.REQ_CORRELATE, 1)  # Correlate replies with requests
            
            # Connect with proper error handling
            try:
                print(f"Connecting to ZMQ server at {self.server_address}...")
                self.socket.connect(self.server_address)
                print(f"Connected to ZMQ server at {self.server_address}")
                
                # Test connection with a simple ping
                try:
                    self.socket.send_string("ping")
                    poller = zmq.Poller()
                    poller.register(self.socket, zmq.POLLIN)
                    if poller.poll(ZMQ_TIMEOUT):
                        response = self.socket.recv_string()
                        if response == "pong":
                            print("✅ ZMQ socket connected and responding to pings")
                            return True
                        else:
                            print(f"⚠️ Unexpected ping response: {response}")
                    else:
                        print("⚠️ No response to ping within timeout")
                except Exception as e:
                    print(f"⚠️ Error during ping test: {str(e)}")
                
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

    def send_observation_and_receive_action(self, observation):
        """Send an observation to the Julia ZMQ server and receive the next action (waypoint)
        
        Args:
            observation: Dictionary containing environmental data including:
                         - drone_position
                         - target_position
                         - obstacle_positions
                         - obstacle_distances
                         - other parameters needed by the active inference model
        
        Returns:
            tuple: (next_waypoint, policy) where:
                   - next_waypoint is a list [x, y, z] in NED coordinates
                   - policy is a list of future waypoints (may be empty if not returned by server)
        """
        # Ensure socket is connected
        if not hasattr(self, 'socket') or self.socket is None:
            print("ZMQ socket not initialized, attempting to connect...")
            success = self._setup_zmq_connection()
            if not success:
                print("❌ Failed to establish ZMQ connection")
                return None, []
        
        # Transform observations to match server expectations
        transformed_observation = self._sanitize_for_json(observation)
        
        # Convert obstacle points to voxel grid format
        if 'obstacle_positions' in transformed_observation and len(transformed_observation['obstacle_positions']) > 0:
            transformed_observation['voxel_grid'] = transformed_observation.pop('obstacle_positions')
            
            if 'obstacle_distances' in transformed_observation and len(transformed_observation['obstacle_distances']) > 0:
                transformed_observation['nearest_obstacle_dist'] = min(transformed_observation['obstacle_distances'])
                transformed_observation['nearest_obstacle_distances'] = transformed_observation.pop('obstacle_distances')
            
            obstacle_count = len(transformed_observation['voxel_grid'])
            if obstacle_count > 0 and 'density_radius' in transformed_observation:
                density = min(1.0, obstacle_count / (100.0 * transformed_observation['density_radius']))
                transformed_observation['obstacle_density'] = density
                print(f"Calculated obstacle density: {density:.3f} based on {obstacle_count} obstacles")
            else:
                transformed_observation['obstacle_density'] = 0.0
        else:
            transformed_observation['voxel_grid'] = []
            transformed_observation['nearest_obstacle_distances'] = [100.0, 100.0]
            transformed_observation['obstacle_density'] = 0.0
        
        # Add suitability preferences
        transformed_observation['target_preference_weight'] = 0.7
        transformed_observation['obstacle_repulsion_weight'] = 0.6
        
        # Add suitability weights dictionary for planning
        transformed_observation['suitability_weights'] = {
            'nearest_obstacle_weight': 0.5,
            'obstacle_density_weight': 0.5,
            'min_suitability_threshold': 0.3
        }
        
        # Convert to JSON string with extra validation
        try:
            json_data = json.dumps(transformed_observation, cls=NumpyJSONEncoder)
            # Validate JSON can be parsed back
            test = json.loads(json_data)
            if not isinstance(test, dict):
                raise ValueError("JSON did not serialize to a dictionary")
        except Exception as e:
            print(f"❌ Error serializing observation to JSON: {str(e)}")
            traceback.print_exc()
            return None, []
        
        # Set up variables for retry logic
        max_retries = ZMQ_MAX_RETRIES
        retry_count = 0
        success = False
        
        # Try sending and receiving with retries
        while retry_count <= max_retries and not success:
            try:
                # Send observation
                print(f"Sending observation to Julia server (attempt {retry_count + 1}/{max_retries + 1})...")
                print(f"Observation size: {len(json_data)/1024:.1f} KB")
                
                # First check if socket is in valid state
                try:
                    events = self.socket.getsockopt(zmq.EVENTS)
                    if not (events & zmq.POLLOUT):
                        print("Socket not ready for sending, resetting...")
                        self._reset_socket()
                except zmq.ZMQError:
                    print("Error checking socket state, resetting...")
                    self._reset_socket()
                
                # Use send with a timeout poll to avoid blocking forever
                self.socket.send_string(json_data, flags=zmq.NOBLOCK)
                
                # Receive response with polling to handle timeout
                print("Waiting for response from Julia server...")
                poller = zmq.Poller()
                poller.register(self.socket, zmq.POLLIN)
                
                # Use poller with timeout
                if poller.poll(ZMQ_TIMEOUT):  # Timeout in milliseconds
                    # Receive the response
                    response_str = self.socket.recv_string()
                    success = True
                    print(f"Successfully received response: {len(response_str)/1024:.1f} KB")
                else:
                    print(f"⚠️ Timeout waiting for response (attempt {retry_count + 1}/{max_retries + 1})")
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        print("Resetting socket and retrying...")
                        self._reset_socket()
                        time.sleep(2)  # Longer pause before retry
                    else:
                        print("❌ Maximum retries reached. Using fallback strategy.")
                        return None, []
            
            except zmq.ZMQError as e:
                print(f"❌ ZMQ error communicating with server: {str(e)}")
                retry_count += 1
                
                if retry_count <= max_retries:
                    print(f"Resetting socket and retrying (attempt {retry_count}/{max_retries})...")
                    self._reset_socket()
                    time.sleep(2)  # Longer pause before retry
                else:
                    print("❌ Maximum retries reached. Using fallback strategy.")
                    return None, []
            
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
                traceback.print_exc()
                return None, []
        
        # Process the received response
        if not success:
            return None, []
        
        try:
            # Parse the response JSON
            response = json.loads(response_str)
            print(f"Response keys: {list(response.keys())}")
            
            # Check for any error message from the server
            if 'error' in response:
                print(f"⚠️ Server reported an error: {response['error']}")
            
            # Try to extract the waypoint from different possible keys
            next_waypoint = None
            for key in ['waypoint', 'action', 'next_waypoint']:
                if key in response and isinstance(response[key], list) and len(response[key]) == 3:
                    next_waypoint = response[key]
                    break
            
            # If no valid waypoint found, look for more detailed info
            if next_waypoint is None:
                print(f"❌ Could not find valid waypoint in response")
                print(f"Response content: {response}")
                return None, []
                
            # Extract policy if present
            policy = response.get('policy', [])
            
            # Validate next_waypoint
            if isinstance(next_waypoint, list) and len(next_waypoint) == 3:
                # Check for invalid values
                if any(math.isnan(x) or math.isinf(x) for x in next_waypoint):
                    print(f"⚠️ Invalid values in waypoint: {next_waypoint}")
                    return None, []
                
                print(f"Successfully extracted waypoint: {[round(p, 2) for p in next_waypoint]}")
                return next_waypoint, policy
            else:
                print(f"❌ Invalid waypoint format: {next_waypoint}")
                return None, []
                
        except json.JSONDecodeError:
            print(f"❌ Failed to decode JSON response from server")
            print(f"Raw response: {response_str[:200]}...")  # Print first 200 chars
            return None, []
            
        except Exception as e:
            print(f"❌ Error processing server response: {str(e)}")
            traceback.print_exc()
            return None, []

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
                            port = self.server_address.split(":")[-1]
                            result = subprocess.run(
                                ["netstat", "-ano", "|", "findstr", f":{port}"],
                                shell=True, capture_output=True, text=True)
                            if "LISTENING" in result.stdout:
                                print(f"⚠️ Something is listening on port {port} but couldn't confirm if it's Julia")
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
        """Start the Julia ZMQ server as a separate process
        
        Returns:
            bool: True if server was successfully started, False otherwise
        """
        print("Starting Julia ZMQ server...")
        
        # Get the path to the zmq_server.jl script
        cwd = os.path.dirname(os.path.abspath(__file__))
        
        # First check in the "actinf" subdirectory (preferred location)
        zmq_server_path = os.path.join(cwd, "actinf", "zmq_server.jl")
        if not os.path.exists(zmq_server_path):
            # Try parent directory as fallback
            zmq_server_path = os.path.join(cwd, "zmq_server.jl")
            if not os.path.exists(zmq_server_path):
                print(f"❌ zmq_server.jl not found at {zmq_server_path}")
                return False
        
        print(f"Found zmq_server.jl at: {zmq_server_path}")
        
        # Find Julia executable
        julia_path = "julia"  # Default to system path
        
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
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.0\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.8.5\bin\julia.exe"
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
        
        try:
            # First make sure any existing julia process with zmq_server.jl is terminated
            if platform.system() == "Windows":
                try:
                    # On Windows, we need to use different commands
                    find_cmd = "wmic process where \"commandline like '%zmq_server.jl%' and name='julia.exe'\" get processid"
                    result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
                    
                    # Parse output to find PIDs
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:  # First line is header
                        for line in lines[1:]:
                            try:
                                pid = int(line.strip())
                                if pid > 0:
                                    kill_cmd = f"taskkill /F /PID {pid}"
                                    subprocess.run(kill_cmd, shell=True, capture_output=True, text=True)
                                    print(f"Terminated existing Julia ZMQ server process with PID {pid}")
                            except (ValueError, TypeError):
                                pass
                except Exception as e:
                    print(f"Warning when trying to kill existing Julia processes: {e}")
            else:
                # On Unix-like systems, use pkill
                try:
                    subprocess.run(["pkill", "-f", "zmq_server.jl"], 
                                  capture_output=True, text=True)
                except Exception as e:
                    print(f"Warning when trying to kill existing Julia processes: {e}")
            
            # Brief pause to ensure process termination
            time.sleep(1)
            
            # Start the ZMQ server with platform-specific approach
            if platform.system() == "Windows":
                print("Starting ZMQ server on Windows...")
                
                # Create log file path
                log_path = os.path.join(cwd, "zmq_server.log")
                
                # Run with visible window, captured output, and logged to file
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 1  # SW_NORMAL
                
                # For Windows, use Popen to create a detached process
                with open(log_path, "w") as log_file:
                    subprocess.Popen(
                        [julia_path, zmq_server_path],
                        stdout=log_file,
                        stderr=log_file,
                        cwd=cwd,
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                        startupinfo=startupinfo
                    )
                
                print(f"ZMQ server started - logs will be written to {log_path}")
            else:
                # For Unix-like systems, use a background process
                print("Starting ZMQ server on Unix-like system...")
                log_path = os.path.join(cwd, "zmq_server.log")
                
                # Use nohup to detach the process properly
                with open(log_path, "w") as log_file:
                    # Start Julia process in background
                    subprocess.Popen(
                        [julia_path, zmq_server_path],
                        stdout=log_file,
                        stderr=log_file,
                        cwd=cwd
                    )
                
                print(f"ZMQ server started - logs will be written to {log_path}")
            
            # Wait for the server to initialize
            print("Waiting for ZMQ server to initialize...")
            server_ready = False
            max_wait = 60  # Maximum seconds to wait
            start_time = time.time()
            
            while not server_ready and (time.time() - start_time) < max_wait:
                # Check if the server is running
                if self._is_server_running():
                    server_ready = True
                    print(f"ZMQ server is ready after {time.time() - start_time:.1f} seconds")
                    break
                
                # Progress indicator
                if (time.time() - start_time) % 5 < 0.1:  # Show message roughly every 5 seconds
                    elapsed = time.time() - start_time
                    print(f"Still waiting for ZMQ server after {elapsed:.1f}s...")
                
                time.sleep(1)
            
            if not server_ready:
                print(f"❌ Timed out waiting for ZMQ server to initialize after {max_wait} seconds")
                print("Please check the ZMQ server log file for errors")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting ZMQ server: {str(e)}")
            traceback.print_exc()
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
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.0\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.8.5\bin\julia.exe"
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
        
        # Create ZMQ interface first to initialize properties
        print("Creating ZMQ interface...")
        zmq_interface = ZMQInterface(defer_connection=True)
        
        # Attempt to check if server is already running
        if zmq_interface._is_server_running():
            print("ZMQ server is already running - proceeding with the existing server")
            return zmq_interface
            
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
    
    # Explicitly start ZMQ server now - this is the critical change
    print("\n==== Explicitly Starting Julia ZMQ Server ====")
    
    # If we don't have an interface yet, create one
    if 'zmq_interface' not in locals():
        zmq_interface = ZMQInterface(defer_connection=True)
    
    # Call the explicit server start method, which will start the server if not already running
    server_started = zmq_interface._start_server()
    
    if server_started:
        print("✅ ZMQ server started successfully")
        # Now we can set up the ZMQ connection
        connection_success = zmq_interface._setup_zmq_connection()
        if connection_success:
            print("✅ ZMQ connection established")
        else:
            print("❌ Failed to establish ZMQ connection after starting server")
            print("Please check Julia installation and dependencies")
    else:
        print("❌ Failed to start ZMQ server")
        print("Navigation will not work without the ZMQ server")
    
    return zmq_interface

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
    
    # Use AirSim's built-in takeoff API only
    print("Taking off...")
    client.takeoffAsync().join()
    
    # Get post-takeoff position
    drone_state = client.getMultirotorState().kinematics_estimated
    drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
    print(f"Position after takeoff: {[round(p, 2) for p in drone_pos]}")
    
    # Initialize iteration counter and distance to target
    iteration = 0
    distance_to_target = np.linalg.norm(np.array(TARGET_LOCATION) - np.array(drone_pos))
    
    # Store trajectory for visualization
    trajectory = [drone_pos]
    
    print(f"Initial distance to target: {distance_to_target:.2f} meters")
    print(f"Starting navigation with Active Inference...")
    
    # Main navigation loop
    while distance_to_target > ARRIVAL_THRESHOLD and iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n=== Iteration {iteration}/{MAX_ITERATIONS} ===")
        
        # Get current drone state
        drone_state = client.getMultirotorState().kinematics_estimated
        drone_pos = [
            drone_state.position.x_val,
            drone_state.position.y_val,
            drone_state.position.z_val
        ]
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(np.array(TARGET_LOCATION) - np.array(drone_pos))
        print(f"Current position: {[round(p, 2) for p in drone_pos]}")
        print(f"Distance to target: {distance_to_target:.2f} meters")
        
        # Get environmental information (obstacles)
        obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
        
        # Create observation for active inference
        observation = {
            "drone_position": drone_pos,
            "target_position": TARGET_LOCATION,
            "obstacle_positions": obstacle_positions,  # Will be empty list if no obstacles detected
            "obstacle_distances": obstacle_distances,  # Will be empty list if no obstacles detected
            "waypoint_count": WAYPOINT_SAMPLE_COUNT,
            "safety_margin": MARGIN,
            "policy_length": POLICY_LENGTH,
            "density_radius": DENSITY_RADIUS
        }
        
        # Print diagnostic info
        print(f"Detected {len(obstacle_positions)} obstacles")
        
        # Send observation to Julia server and get next waypoint via ZMQ
        print("Computing next waypoint with Active Inference...")
        next_waypoint, policy = zmq_interface.send_observation_and_receive_action(observation)
        
        # Check if we received a valid waypoint - if not, terminate navigation
        if next_waypoint is None:
            print("❌ Failed to get valid waypoint from Active Inference model")
            print("ZMQ communication failed. Terminating navigation.")
            break
        
        print(f"Next waypoint: {[round(p, 2) for p in next_waypoint]}")
        
        # Move drone to the next waypoint
        move_to_waypoint(client, drone_pos, next_waypoint)
        
        # Record trajectory
        trajectory.append(next_waypoint)
        
        # Check if we've reached the target
        if distance_to_target <= ARRIVAL_THRESHOLD:
            print("\n🎉 Target reached successfully!")
            break
        
        # Small delay to allow for state updates
        time.sleep(0.5)
    
    # Final status report
    if distance_to_target <= ARRIVAL_THRESHOLD:
        print(f"\n✅ Navigation completed successfully in {iteration} iterations!")
        print(f"Final position: {[round(p, 2) for p in drone_pos]}")
        print(f"Distance to target: {distance_to_target:.2f} meters")
    elif next_waypoint is None:
        print("\n❌ Navigation terminated due to ZMQ communication failure")
        print(f"Final position: {[round(p, 2) for p in drone_pos]}")
        print(f"Remaining distance to target: {distance_to_target:.2f} meters")
    else:
        print(f"\n⚠️ Maximum iterations ({MAX_ITERATIONS}) reached without arriving at target")
        print(f"Final position: {[round(p, 2) for p in drone_pos]}")
        print(f"Remaining distance to target: {distance_to_target:.2f} meters")
    
    # Visualize the trajectory
    try:
        print("\nGenerating trajectory visualization...")
        trajectory_array = np.array(trajectory)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 'b-', label='Drone Path')
        
        # Mark start and end points
        ax.scatter(trajectory_array[0, 0], trajectory_array[0, 1], trajectory_array[0, 2], 
                   c='green', marker='o', s=100, label='Start')
        ax.scatter(trajectory_array[-1, 0], trajectory_array[-1, 1], trajectory_array[-1, 2], 
                   c='red', marker='x', s=100, label='End')
        
        # Mark target
        ax.scatter(TARGET_LOCATION[0], TARGET_LOCATION[1], TARGET_LOCATION[2], 
                   c='purple', marker='*', s=200, label='Target')
        
        # Plot obstacles if any were detected
        if len(obstacle_positions) > 0:
            obstacle_array = np.array(obstacle_positions)
            ax.scatter(obstacle_array[:, 0], obstacle_array[:, 1], obstacle_array[:, 2], 
                       c='orange', marker='^', s=50, alpha=0.5, label='Obstacles')
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('Drone Navigation Trajectory with Active Inference')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Add legend
        ax.legend()
        
        # Save figure
        plt.savefig('drone_path.png')
        print("Trajectory visualization saved to 'drone_path.png'")
        
        # Show plot (comment this out if running headless)
        plt.show()
        
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
    
    print("\nNavigation sequence complete")
    
    # Land the drone at the end
    print("Landing the drone...")
    client.landAsync().join()
    
    # Disarm
    client.armDisarm(False)

main()


