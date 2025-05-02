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
        """Test the ZMQ connection - make ping optional and focus on connection only"""
        try:
            # Just check that the socket is connected - don't rely on ping
            print("ZMQ socket connected, skipping ping test.")
            return True
            
        except Exception as e:
            print(f"⚠️ ZMQ connection test check failed: {str(e)}")
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
    
    def _is_server_running(self):
        """Check if the ZMQ socket is connected without relying on ping"""
        # Simply check if the socket exists
        if hasattr(self, 'socket') and self.socket is not None:
            return True
        return False
    
    def _start_server(self):
        """Start the Julia ZMQ server automatically using the existing zmq_server.jl script
        
        Returns:
            bool: True if server was started successfully, False otherwise
        """
        print("\n==== Starting Julia ZMQ Server ====")
        
        try:
            # Determine the correct path to zmq_server.jl - check both root and actinf directory
            server_script_path = "zmq_server.jl"  # First try root directory
            if not os.path.exists(server_script_path):
                # Try actinf directory as fallback
                server_script_path = os.path.join("actinf", "zmq_server.jl")
                if not os.path.exists(server_script_path):
                    print(f"⚠️ Server script not found at root or in actinf directory")
                    print("❌ Could not find zmq_server.jl")
                    return False
            
            print(f"Found server script at: {server_script_path}")
            
            # Extract the port from the current server address
            port = self.server_address.split(":")[-1]
            
            # Ensure port is set to 5555 (default for Julia server)
            if port != "5555":
                print(f"Warning: Julia server is configured to use port 5555, but current address uses {port}")
                print("Switching to port 5555 for compatibility")
                self.server_address = "tcp://localhost:5555"
                port = "5555"
            
            # First check if Julia process is already running with ZMQ server
            print("Checking if Julia ZMQ server is already running...")
            if platform.system() == "Windows":
                result = subprocess.run(["tasklist"], capture_output=True, text=True)
                if "julia" in result.stdout.lower():
                    print("✅ Julia process found running - will attempt to connect to it")
                    time.sleep(1)
                    return True
            else:  # Linux/Mac
                result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
                if "julia" in result.stdout and "zmq_server" in result.stdout:
                    print("✅ Julia process with ZMQ server found running")
                    time.sleep(1)
                    return True
            
            # Kill any potentially stuck/zombie Julia processes first
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/IM", "julia.exe"], 
                                  capture_output=True, text=True)
                else:
                    subprocess.run(["pkill", "-f", "julia"], 
                                  capture_output=True, text=True)
                print("Cleaned up any existing Julia processes")
                time.sleep(1)  # Wait for processes to terminate
            except Exception as e:
                print(f"Note: Could not kill existing Julia processes: {str(e)}")
            
            # Prepare the command to start Julia with full paths
            julia_path = "julia"
            
            # Try to find julia executable if it's not in the PATH
            if platform.system() == "Windows":
                possible_paths = [
                    r"C:\Julia-1.9.3\bin\julia.exe",
                    r"C:\Julia-1.9.2\bin\julia.exe",
                    r"C:\Julia-1.9.1\bin\julia.exe",
                    r"C:\Julia-1.9.0\bin\julia.exe",
                    r"C:\Julia-1.8.5\bin\julia.exe",
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        julia_path = path
                        print(f"Found Julia at: {julia_path}")
                        break
            
            # Build the full command
            cmd = [julia_path, "--project=.", server_script_path]
            print(f"Executing: {' '.join(cmd)}")
            
            # Get current working directory for subprocess
            cwd = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
            print(f"Working directory: {cwd}")
            
            # Start the server as a background process
            if platform.system() == "Windows":
                try:
                    # On Windows, first try with CREATE_NEW_CONSOLE
                    print("Starting Julia server in a new console window...")
                    server_process = subprocess.Popen(
                        cmd,
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                        cwd=cwd
                    )
                except Exception as e:
                    print(f"Error starting with new console: {str(e)}")
                    # Fallback to regular background process
                    print("Falling back to standard process...")
                    server_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=cwd
                    )
            else:
                # On Unix-like systems, start in the background
                server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd
                )
            
            # Wait longer for server to start
            print("Waiting for server to initialize (10 seconds)...")
            for i in range(10):
                print(f"Starting Julia server: {i+1}/10 seconds elapsed...", end="\r")
                time.sleep(1)
                
                # Check if process has exited early (error)
                if server_process.poll() is not None:
                    break
            
            print("\nChecking server status...")
            
            # Check if process is still running
            if server_process.poll() is None:
                print("✅ Server process started successfully and is still running")
                return True
            else:
                # Process has terminated, get output
                try:
                    stdout, stderr = server_process.communicate(timeout=1)
                    print(f"❌ Server process failed to start (exit code: {server_process.returncode})")
                    if stdout:
                        print(f"Server stdout: {stdout.decode('utf-8', errors='replace')}")
                    if stderr:
                        print(f"Server stderr: {stderr.decode('utf-8', errors='replace')}")
                except Exception as e:
                    print(f"Error retrieving server output: {str(e)}")
                
                # Try a different approach - use run instead of Popen
                print("\nAttempting to start server with subprocess.run as a fallback...")
                try:
                    # Use run with a timeout, just to check if Julia can start at all
                    result = subprocess.run(
                        [julia_path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    print(f"Julia version check: {result.stdout.strip()}")
                    
                    # Now try to run the actual server script
                    print("Starting ZMQ server as a detached process...")
                    
                    # On Windows, start the process detached
                    if platform.system() == "Windows":
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = 1  # SW_SHOWNORMAL
                        
                        server_process = subprocess.Popen(
                            cmd,
                            startupinfo=startupinfo,
                            creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS,
                            cwd=cwd
                        )
                    else:
                        # On Unix, use setsid to detach
                        server_process = subprocess.Popen(
                            cmd,
                            stdout=open(os.devnull, 'w'),
                            stderr=open(os.devnull, 'w'),
                            preexec_fn=os.setsid,
                            cwd=cwd
                        )
                    
                    print("Server process started with alternative method")
                    time.sleep(5)  # Give it time to initialize
                    return True
                    
                except Exception as e:
                    print(f"❌ Failed to start server with alternative method: {str(e)}")
                    return False
                
        except Exception as e:
            print(f"❌ Error starting ZMQ server: {str(e)}")
            traceback.print_exc()
            return False
    
    def _diagnose_zmq_server_issues(self):
        """Run simple diagnostics on ZMQ server connection issues"""
        print("\n==== ZMQ SERVER DIAGNOSTICS ====")
        
        # Check socket status
        if not hasattr(self, 'socket') or self.socket is None:
            print("⚠️ ZMQ socket is not initialized")
            return False
        
        # Check if Julia process is running
        julia_running = False
        if platform.system() == "Windows":
            result = subprocess.run(["tasklist"], capture_output=True, text=True)
            if "julia" in result.stdout.lower():
                print("✅ Julia process found running")
                julia_running = True
            else:
                print("❌ No Julia process found running")
                print("Please start the Julia ZMQ server with:")
                print("   julia --project=. actinf/zmq_server.jl")
        else:  # Linux/Mac
            result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
            if "julia" in result.stdout:
                print("✅ Julia process found running")
                julia_running = True
            else:
                print("❌ No Julia process found running")
                print("Please start the Julia ZMQ server with:")
                print("   julia --project=. actinf/zmq_server.jl")
        
        print("================================\n")
        return julia_running
    
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
        """Send observation to Julia server and receive action via ZMQ
        
        Args:
            observation: Dictionary containing observation data
            
        Returns:
            tuple: (next_waypoint, policy) if successful, otherwise (None, None)
            
        Notes:
            The waypoint received from Julia is expected to be in the drone's local coordinate 
            frame (egocentric). This means it represents a relative movement from the drone's 
            current position, with the drone's orientation determining the forward direction.
        """
        # Check if ZMQ interface is initialized
        if not hasattr(self, 'socket') or self.socket is None:
            print("⚠️ ZMQ socket not initialized, attempting to connect")
            if not self._setup_zmq_connection():
                print("❌ Failed to initialize ZMQ socket")
                # If connection setup fails, check if server is running - attempt to start if not
                if not self._is_server_running():
                    print("Julia server not detected, attempting to start...")
                    self._start_server()
                    time.sleep(2)  # Wait for server to initialize
                    if not self._setup_zmq_connection():
                        print("❌ Still unable to connect after starting server")
                        return None, None
                else:
                    return None, None
        
        # Maximum number of retry attempts
        max_retries = ZMQ_MAX_RETRIES
        current_retry = 0
        
        while current_retry < max_retries:
            try:
                print(f"Sending request to ZMQ server at {self.server_address}...")
                
                # Check socket state - recreate if needed
                if hasattr(self, 'socket') and self.socket is not None:
                    socket_state = 0
                    try:
                        socket_state = self.socket.getsockopt(zmq.EVENTS)
                        if not (socket_state & zmq.POLLOUT):
                            print("⚠️ Socket not ready for sending, resetting...")
                            self._reset_socket()
                    except zmq.ZMQError:
                        print("⚠️ Error checking socket state, resetting...")
                        self._reset_socket()
                        
                    # Verify socket after reset
                    if not hasattr(self, 'socket') or self.socket is None:
                        print("Socket still invalid after reset, attempting to reconnect")
                        self._setup_zmq_connection()
                        if not hasattr(self, 'socket') or self.socket is None:
                            current_retry += 1
                            continue
                else:
                    print("⚠️ Socket object missing or None, recreating...")
                    self._reset_socket()
                    
                    # Verify socket after reset
                    if not hasattr(self, 'socket') or self.socket is None:
                        print("Socket still invalid after reset, attempting to reconnect")
                        self._setup_zmq_connection()
                        if not hasattr(self, 'socket') or self.socket is None:
                            current_retry += 1
                            continue
                
                # Sanitize and serialize observation to JSON using custom NumPy encoder
                sanitized_observation = self._sanitize_for_json(observation)
                json_data = json.dumps(sanitized_observation, cls=NumpyJSONEncoder)
                
                # Send data to Julia server with timeout handling
                try:
                    self.socket.send_string(json_data, flags=zmq.NOBLOCK)
                    print("Request sent, waiting for response...")
                except zmq.ZMQError as e:
                    print(f"Error sending data: {str(e)}")
                    self._reset_socket()
                    current_retry += 1
                    continue
                
                # Receive response with timeout handling
                try:
                    # Use polling to implement a more reliable timeout
                    poller = zmq.Poller()
                    poller.register(self.socket, zmq.POLLIN)
                    
                    # Wait for response with timeout
                    if poller.poll(ZMQ_TIMEOUT):
                        response = self.socket.recv_string()
                        print("Response received from ZMQ server")
                    else:
                        print("⚠️ Timeout waiting for response")
                        # Check if server is still running
                        if not self._is_server_running():
                            print("Julia server appears to have stopped, attempting to restart...")
                            self._start_server()
                            time.sleep(2)  # Wait for server to initialize
                        self._reset_socket()
                        current_retry += 1
                        continue
                except zmq.ZMQError as e:
                    print(f"Error receiving data: {str(e)}")
                    self._reset_socket()
                    current_retry += 1
                    continue
                
                # Verbose debugging for response
                print(f"Raw response: {response[:200]}...")  # Print first 200 chars to avoid flooding console
                
                # Safely parse JSON response
                try:
                    parsed_response = json.loads(response)
                    print(f"Response keys: {list(parsed_response.keys())}")
                except json.JSONDecodeError as je:
                    print(f"❌ JSON parsing error: {str(je)}")
                    print(f"Response data (truncated): {response[:100]}...")
                    # If we can't parse the response at all, there may be server issues
                    if "julia" in response.lower() and "error" in response.lower():
                        print("Julia server error detected in response")
                        # Try to restart the server after 2 failed parsing attempts
                        if current_retry >= 1:
                            print("Multiple JSON parsing failures, attempting to restart Julia server...")
                            self._start_server()
                            time.sleep(3)  # Give more time for restart
                    current_retry += 1
                    continue
                
                # Extract waypoint from response - handle all possible formats
                waypoint = None
                
                # Check for error messages
                if 'error' in parsed_response:
                    print(f"Server error: {parsed_response['error']}")
                    if 'restart_recommended' in parsed_response and parsed_response['restart_recommended']:
                        print("Server recommended restart - restarting Julia server...")
                        self._start_server()
                        time.sleep(3)
                    current_retry += 1
                    continue
                
                # IMPROVEMENT: Check waypoint and action keys first (most reliable)
                if "waypoint" in parsed_response and isinstance(parsed_response["waypoint"], list) and len(parsed_response["waypoint"]) >= 3:
                    waypoint = [float(x) for x in parsed_response["waypoint"][:3]]
                    print(f"Extracted waypoint from 'waypoint' key: {waypoint}")
                # Fall back to action key if waypoint not found
                elif "action" in parsed_response and isinstance(parsed_response["action"], list) and len(parsed_response["action"]) >= 3:
                    waypoint = [float(x) for x in parsed_response["action"][:3]]
                    print(f"Extracted waypoint from 'action' key: {waypoint}")
                # Fall back to checking other possible keys if neither waypoint nor action is found
                else:
                    # Check all possible key names for waypoint/action data
                    for key in ['nextState', 'next_state', 'state']:
                        if key in parsed_response:
                            value = parsed_response[key]
                            # Handle case where value might be nested
                            if isinstance(value, dict) and 'position' in value:
                                waypoint = value['position']
                            elif isinstance(value, list) and len(value) >= 3:
                                # Ensure numeric values
                                if all(isinstance(v, (int, float)) for v in value[:3]):
                                    waypoint = value[:3]
                                else:
                                    print(f"Found key '{key}' but values are not all numeric: {value[:3]}")
                            elif isinstance(value, (list, tuple)) and all(isinstance(item, (int, float)) for item in value[:3]):
                                # If it's a list/tuple of numbers
                                waypoint = list(value[:3])
                            else:
                                print(f"Found key '{key}' but value is not in expected format: {value}")
                    
                    # If still no waypoint, check if there's a position key directly
                    if waypoint is None and 'position' in parsed_response:
                        position_value = parsed_response['position']
                        if isinstance(position_value, (list, tuple)) and len(position_value) >= 3:
                            if all(isinstance(v, (int, float)) for v in position_value[:3]):
                                waypoint = list(position_value[:3])
                            else:
                                print(f"Found 'position' but values are not all numeric: {position_value[:3]}")
                        else:
                            print(f"Found 'position' but format is invalid: {position_value}")
                    
                    # If still no waypoint, try to extract from policy if it exists
                    if waypoint is None and 'policy' in parsed_response:
                        policy_value = parsed_response['policy']
                        if isinstance(policy_value, list) and len(policy_value) > 0:
                            first_policy_element = policy_value[0]
                            if isinstance(first_policy_element, (list, tuple)) and len(first_policy_element) >= 3:
                                if all(isinstance(v, (int, float)) for v in first_policy_element[:3]):
                                    waypoint = list(first_policy_element[:3])
                                    print("Extracted waypoint from first policy element")
                
                # If still no waypoint, log detailed info
                if waypoint is None:
                    print(f"❌ No valid waypoint found in response. Keys: {list(parsed_response.keys())}")
                    # Show the first level of nested structure
                    for key, value in parsed_response.items():
                        if isinstance(value, dict):
                            print(f"  '{key}' contains keys: {list(value.keys())}")
                        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            print(f"  '{key}' is a list of dicts with first item keys: {list(value[0].keys())}")
                    
                    # Debug output the complete response for analysis
                    print("Full response structure:")
                    print(json.dumps(parsed_response, indent=2)[:500] + "...")  # Limit output length
                    
                    # If we still can't find a waypoint, try a fallback strategy
                    if current_retry >= max_retries - 1:
                        print("Fallback: Using [0,0,-5] as safe hover waypoint")
                        waypoint = [0.0, 0.0, -5.0]  # Safe waypoint that hovers in place
                    else:
                        current_retry += 1
                        continue
                
                # Extract policy if available
                policy = parsed_response.get('policy')
                
                # If policy not available, try to construct a simple one from the waypoint
                if policy is None:
                    # Create a default policy if not provided - repeat the waypoint 3 times
                    policy = [waypoint] * 3
                
                print(f"Response processed: waypoint={waypoint}, policy length={len(policy) if policy else 0}")
                
                # Validate waypoint format
                if waypoint and isinstance(waypoint, (list, tuple)) and len(waypoint) >= 3:
                    # Ensure all waypoint values are valid numbers (not NaN or Inf)
                    try:
                        for i in range(3):
                            if not isinstance(waypoint[i], (int, float)) or \
                               (isinstance(waypoint[i], float) and (np.isnan(waypoint[i]) or np.isinf(waypoint[i]))):
                                print(f"⚠️ Invalid value in waypoint at index {i}: {waypoint[i]}")
                                waypoint[i] = 0.0  # Replace invalid value with safe default
                        
                        # Ensure waypoint is a list of 3 floats
                        waypoint = [float(waypoint[0]), float(waypoint[1]), float(waypoint[2])]
                        print(f"Final validated waypoint: {waypoint}")
                        return waypoint, policy
                    except Exception as vex:
                        print(f"⚠️ Error validating waypoint: {str(vex)}")
                        if current_retry >= max_retries - 1:
                            print("Fallback: Using [0,0,-5] as safe hover waypoint")
                            return [0.0, 0.0, -5.0], [[0.0, 0.0, -5.0]] * 3
                else:
                    print(f"⚠️ Invalid waypoint format received: {waypoint}")
                    current_retry += 1
                    continue
                
            except zmq.ZMQError as e:
                print(f"⚠️ ZMQ communication error: {str(e)}")
                
                if current_retry < max_retries:
                    # Try to reset the connection
                    print("Resetting socket and retrying...")
                    self._reset_socket()
                    time.sleep(1)  # Add a small delay between retries
                    current_retry += 1
                else:
                    break
            
            except Exception as e:
                print(f"⚠️ Unexpected error in ZMQ communication: {str(e)}")
                traceback.print_exc()
                self._reset_socket()
                time.sleep(1)
                current_retry += 1
        
        print("Exceeded maximum retry attempts")
        print("⚠️ Error in ZMQ communication: Invalid waypoint received from ZMQ: None")
        
        # Last resort fallback - return a safe hovering waypoint after all retries fail
        print("⚠️ Using emergency fallback waypoint [0,0,-5]")
        return [0.0, 0.0, -5.0], [[0.0, 0.0, -5.0]] * 3

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
    try:
        # Run precompilation script to prepare Julia environment
        julia_cmd = ["julia", "precompile.jl"]
        result = subprocess.run(julia_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Julia precompilation completed successfully:")
            print(result.stdout.split("\n")[-2])  # Print confirmation message
            precompile_success = True
        else:
            print(f"⚠️ Julia precompilation returned code {result.returncode}:")
            print(result.stderr[:500])  # Print first 500 chars of error
            print("Continuing anyway, but there may be performance issues")
    except subprocess.TimeoutExpired:
        print("⚠️ Julia precompilation timed out after 60 seconds")
        print("Continuing anyway, but there may be performance issues")
    except FileNotFoundError:
        print("⚠️ Julia executable not found in PATH")
        print("Continuing without precompilation, connectivity may be affected")
    except Exception as e:
        print(f"⚠️ Error during precompilation: {str(e)}")
        print("Continuing without precompilation")
        
    if not precompile_success:
        print("\nAttempting to start ZMQ server directly...")
        # Try to start ZMQ server directly if precompilation failed
        server_script_path = os.path.join("actinf", "zmq_server.jl")
        if os.path.exists(server_script_path):
            try:
                cmd = ["julia", "--project=.", server_script_path]
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE if platform.system() == "Windows" else 0)
                print("ZMQ server started - waiting for initialization")
                time.sleep(10)  # Give server time to start
            except Exception as e:
                print(f"Error starting server: {str(e)}")
    
    # Create and return ZMQ interface
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
    
    # Take off and hover at starting height
    start_height = -5.0  # 5 meters above ground in NED coordinates (negative Z = up)
    print(f"Taking off to initial altitude: {-start_height} meters")
    client.takeoffAsync().join()
    client.moveToZAsync(start_height, 2).join()
    time.sleep(1)
    
    # Get initial drone state
    drone_state = client.getMultirotorState().kinematics_estimated
    drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
    print(f"Initial position: {[round(p, 2) for p in drone_pos]}")
    
    # Initialize visualization if matplotlib is available
    try:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Navigation Path')
        visualization_enabled = True
    except Exception as e:
        print(f"Visualization disabled: {str(e)}")
        visualization_enabled = False
    
    # Store path for visualization
    path_history = [drone_pos]
    
    # Main navigation loop
    iteration = 0
    arrived = False
    
    while iteration < MAX_ITERATIONS and not arrived:
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Get current drone position and orientation
        drone_state = client.getMultirotorState().kinematics_estimated
        drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
        drone_orientation = [
            drone_state.orientation.w_val,
            drone_state.orientation.x_val,
            drone_state.orientation.y_val,
            drone_state.orientation.z_val
        ]
        
        print(f"Current position: {[round(p, 2) for p in drone_pos]}")
        
        # Calculate distance to target
        distance_to_target = np.sqrt(sum((np.array(drone_pos) - np.array(TARGET_LOCATION))**2))
        print(f"Distance to target: {distance_to_target:.2f} meters")
        
        # Check if arrived at target
        if distance_to_target < ARRIVAL_THRESHOLD:
            print("\n🎯 Target reached! Mission complete.")
            arrived = True
            break
        
        # Get obstacle information from scanner
        obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
        
        # Calculate obstacle density within a radius
        density_radius = DENSITY_RADIUS  # meters
        points_in_radius = sum(1 for dist in obstacle_distances if dist <= density_radius)
        obstacle_density = points_in_radius / max(1, len(obstacle_distances)) if obstacle_distances else 0
        
        nearest_obstacle_dist = min(obstacle_distances) if obstacle_distances else 100.0
        print(f"Nearest obstacle: {nearest_obstacle_dist:.2f}m, Obstacle density: {obstacle_density:.3f}")
        
        # Prepare observation for Julia server
        observation = {
            "drone_position": drone_pos,
            "drone_orientation": drone_orientation,
            "target_location": TARGET_LOCATION,
            "nearest_obstacle_distances": obstacle_distances[:10] if obstacle_distances else [],  # Send top 10 distances
            "obstacle_density": obstacle_density,
            "voxel_grid": [] if not obstacle_positions else obstacle_positions,  # Explicitly send empty list when no obstacles
            "iteration": iteration,
            "obstacle_repulsion_weight": 0.7,  # Can be adjusted based on scenario
            "target_preference_weight": 0.5    # Can be adjusted based on scenario
        }
        
        # Send observation to Julia server and get next waypoint
        next_waypoint, policy = zmq_interface.send_observation_and_receive_action(observation)
        
        if next_waypoint is None:
            print("❌ Failed to get valid waypoint from server. Hovering in place.")
            next_waypoint = [0, 0, 0]  # Hover in place as fallback
        
        print(f"Next waypoint (global): {[round(w, 2) for w in next_waypoint]}")
        
        # Add safety check - don't proceed if obstacles are too close
        if nearest_obstacle_dist < 1.5:  # Safety threshold
            print("⚠️ Obstacle too close! Pausing movement for safety.")
            time.sleep(1)
            continue
        
        # Move to the next waypoint using the orientation-aware function
        move_to_waypoint(client, drone_pos, next_waypoint, velocity=2)
        
        # Update path history
        path_history.append(drone_pos)
        
        # Visualize path if enabled
        if visualization_enabled and iteration % 5 == 0:  # Update visualization every 5 iterations
            try:
                ax.clear()
                path_array = np.array(path_history)
                ax.plot3D(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'b-o')
                ax.plot3D([TARGET_LOCATION[0]], [TARGET_LOCATION[1]], [TARGET_LOCATION[2]], 'r*', markersize=10)
                
                # Plot obstacles if available
                if obstacle_positions:
                    obstacle_array = np.array(obstacle_positions)
                    ax.scatter(obstacle_array[:, 0], obstacle_array[:, 1], obstacle_array[:, 2], c='gray', alpha=0.5, s=10)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'Drone Navigation Path - Iteration {iteration + 1}')
                plt.draw()
                plt.pause(0.001)
            except Exception as viz_error:
                print(f"Visualization error: {str(viz_error)}")
        
        iteration += 1
        time.sleep(0.5)  # Small delay between iterations
    
    # End of navigation
    if not arrived:
        print("\n⚠️ Maximum iterations reached without arriving at target.")
    
    # Return to home position
    print("\nReturning to starting position...")
    client.moveToPositionAsync(0, 0, start_height, 2).join()
    
    # Land the drone
    print("Landing...")
    client.landAsync().join()
    
    # Disarm
    client.armDisarm(False)
    
    # Release control
    client.enableApiControl(False)
    
    print("\n==== Navigation session complete ====")
    
    # Show final visualization
    if visualization_enabled:
        plt.show()  # Keep the plot window open

# If running as main script, execute the navigation process
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nNavigation interrupted by user.")
        # Safely stop the drone
        client.moveByVelocityAsync(0, 0, 0, 1).join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception as e:
        print(f"\n\nNavigation error: {str(e)}")
        traceback.print_exc()
        # Try to land safely
        try:
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except:
            print("Failed to execute emergency landing.")

