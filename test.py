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

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# No need to redefine target since we have TARGET_LOCATION constant

# Scanner for obstacle detection
class Scanner:
    def __init__(self, client, scan_range=20.0, voxel_size=1.0):
        self.client = client
        self.scan_range = scan_range
        self.voxel_size = voxel_size
        self.obstacle_positions = []
        self.obstacle_distances = []
    
    def fetch_density_distances(self):
        """Get obstacle positions and distances with orientation-aware transformation"""
        try:
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
                # Always return empty lists, never integers or other types
                return [], []
            
            # Convert point cloud to positions
            try:
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error reshaping point cloud: {str(e)}")
                # Always return empty lists, never integers or other types
                return [], []
            
            # Filter out points beyond scan range
            obstacle_positions = []
            obstacle_distances = []
            
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
            
            for point in points:
                # Skip invalid points
                if np.isnan(point).any() or np.isinf(point).any():
                    continue
                
                try:
                    # Transform point to global reference frame based on drone orientation
                    # Note: LiDAR points are already in the drone's local frame, so we need to
                    # transform them to the global frame using the drone's orientation
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
                        obstacle_positions.append(global_point)  # Use globally transformed points
                        obstacle_distances.append(float(dist))  # Convert to Python float
                except Exception as point_error:
                    # Skip this point if there was an error
                    continue
            
            return obstacle_positions, obstacle_distances
        
        except Exception as e:
            print(f"Error in fetch_density_distances: {str(e)}")
            traceback.print_exc()
            # Always return empty lists, never integers or other types
            return [], []

# Initialize scanner
scanner = Scanner(client)

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
            
            # The ping test is now optional - uncomment if you want to use it
            """
            # Store current timeout and set shorter timeout for test
            current_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
            self.socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 second timeout for ping test
            
            # Send ping request
            print("Testing ZMQ connection with ping...")
            self.socket.send_string("ping")
            response = self.socket.recv_string()
            
            # Reset timeout to original value
            self.socket.setsockopt(zmq.RCVTIMEO, current_timeout)
            
            # Check response
            if response == "pong":
                print("✅ ZMQ connection test successful")
                return True
            else:
                print(f"⚠️ Unexpected response from server: {response}")
                return True  # Continue anyway
            """
                
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
                
                # Check all possible key names for waypoint/action data
                for key in ['waypoint', 'action', 'nextState', 'next_state', 'state']:
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
    
    def close(self):
        """Close ZMQ connection"""
        try:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()
            print("ZMQ connection closed")
        except Exception as e:
            print(f"Error closing ZMQ connection: {str(e)}")


class DroneController:
    def __init__(self):
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        try:
            # Initialize the environment scanner
            self.scanner = EnvironmentScanner(self.client)
            print("Environment scanner initialized")
        except Exception as e:
            print(f"Warning: Could not initialize environment scanner: {str(e)}")
            print("Falling back to basic scanner")
            self.scanner = Scanner(self.client)
        
        # Remove scanner test that causes errors
        # Initialize state with default values
        self.current_position = [0.0, 0.0, 0.0]
        self.current_orientation = [1.0, 0.0, 0.0, 0.0]  # Default identity quaternion
        self.target_location = TARGET_LOCATION
        
        # Update with actual values
        self.update_drone_state()
        
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
    
    def collect_sensor_data(self, drone, scanner, current_position, target_location, iteration=0):
        """Collect all necessary sensor data for path planning with orientation awareness"""
        try:
            # Update drone state to get fresh orientation data
            self.update_drone_state()
            current_orientation = self.current_orientation
            
            # Validate input positions - ensure they are valid numeric values
            if not all(isinstance(pos, (int, float)) and np.isfinite(pos) for pos in current_position):
                print(f"⚠️ Invalid current_position: {current_position}, using default [0,0,0]")
                current_position = [0.0, 0.0, 0.0]
            
            if not all(isinstance(pos, (int, float)) and np.isfinite(pos) for pos in target_location):
                print(f"⚠️ Invalid target_location: {target_location}, using default [10,0,-3]")
                target_location = [10.0, 0.0, -3.0]
            
            # Calculate target direction and distance - this is essential for navigation
            target_vector = np.array(target_location) - np.array(current_position)
            target_distance = np.linalg.norm(target_vector)
            
            # Validate target_distance is finite
            if not np.isfinite(target_distance) or target_distance < 0:
                target_distance = 0.1  # Default minimum distance
            
            # Normalize target direction
            if target_distance > 0:
                target_direction = target_vector / target_distance
            else:
                # Default direction if target is at current position
                target_direction = np.array([1.0, 0.0, 0.0])
            
            # Convert to list for JSON serialization
            target_direction = target_direction.tolist()
            
            # Get point clouds and calculate obstacle density and distances
            try:
                obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
                
                # Fix for 'int' is not iterable error - ensure we have proper lists
                if isinstance(obstacle_positions, int) or not hasattr(obstacle_positions, '__iter__'):
                    print(f"Warning: obstacle_positions is not iterable (type: {type(obstacle_positions)})")
                    obstacle_positions = []
                    
                if isinstance(obstacle_distances, int) or not hasattr(obstacle_distances, '__iter__'):
                    print(f"Warning: obstacle_distances is not iterable (type: {type(obstacle_distances)})")
                    obstacle_distances = []
            except Exception as e:
                print(f"Error fetching density distances: {str(e)}")
                print("Using empty obstacle lists")
                obstacle_positions = []
                obstacle_distances = []
            
            # Calculate drone's forward vector from quaternion orientation
            w, x, y, z = current_orientation
            forward_x = 2 * (x*z + w*y)
            forward_y = 2 * (y*z - w*x)
            forward_z = 1 - 2 * (x*x + y*y)
            forward_vector = np.array([forward_x, forward_y, forward_z])
            if np.linalg.norm(forward_vector) > 0:
                forward_vector = forward_vector / np.linalg.norm(forward_vector)
            else:
                forward_vector = np.array([1.0, 0.0, 0.0])  # Default to forward if invalid
            
            # Calculate how aligned the drone is with the target direction
            target_alignment = np.dot(forward_vector, target_direction)
            print(f"Target alignment: {target_alignment:.2f} (-1 to 1, 1 is perfectly aligned)")
            
            # IMPORTANT: For open areas with no obstacles, set default safe values
            # These ensure the drone moves directly toward the target when no obstacles are present
            if not obstacle_distances or len(obstacle_distances) == 0:
                print("No obstacles detected - using direct path to target")
                observation = {
                    'drone_position': current_position,
                    'drone_orientation': current_orientation,
                    'forward_vector': forward_vector.tolist(),
                    'target_position': target_location,
                    'target_direction': target_direction,
                    'target_distance': float(target_distance),
                    'target_alignment': float(target_alignment),
                    'obstacle_density': 0.0,  # No obstacles
                    'obstacle_repulsion_weight': 0.0,  # No repulsion
                    'nearest_obstacle_distance': 100.0,  # Far away (no obstacle)
                    'direct_path_clear': True,  # Path is clear
                    'direct_path_suitability': 1.0,  # Perfect suitability
                    'path_suitability': 1.0,  # Perfect suitability
                    'obstacle_positions': [],
                    'obstacle_distances': [],
                    'waypoint_radius': min(target_distance * 0.5, 15.0),  # Allow longer steps when no obstacles
                    'iteration': iteration,
                    'close_obstacle_count_front': 0,
                    'close_obstacle_count_other': 0,
                    'medium_obstacle_count_front': 0,
                    'medium_obstacle_count_other': 0
                }
                return observation
            
            # Process obstacle data with orientation awareness (continues only if obstacles exist)
            # Calculate obstacle density based on the number of obstacles in different ranges
            # with orientation awareness (giving higher weight to obstacles in front)
            close_range = 5.0
            medium_range = 10.0
            far_range = 20.0
            
            # Lists to store obstacles by range and direction
            close_obstacles_front = []
            close_obstacles_other = []
            medium_obstacles_front = []
            medium_obstacles_other = []
            far_obstacles_front = []
            far_obstacles_other = []
            
            # Process each obstacle with directional awareness
            for i, (position, distance) in enumerate(zip(obstacle_positions, obstacle_distances)):
                # Determine if obstacle is in front of the drone
                obstacle_vector = np.array(position) - np.array(current_position)
                if np.linalg.norm(obstacle_vector) > 0:
                    obstacle_vector = obstacle_vector / np.linalg.norm(obstacle_vector)
                else:
                    # Skip obstacles at same position as drone (should be rare/impossible)
                    continue
                
                # Calculate alignment with forward vector (-1 to 1, where 1 is directly in front)
                alignment = np.dot(forward_vector, obstacle_vector)
                is_frontal = alignment > 0.0  # Obstacle is in front if dot product is positive
                
                # Categorize by distance and direction
                if distance < close_range:
                    if is_frontal:
                        close_obstacles_front.append((distance, alignment))
                    else:
                        close_obstacles_other.append(distance)
                elif distance < medium_range:
                    if is_frontal:
                        medium_obstacles_front.append((distance, alignment))
                    else:
                        medium_obstacles_other.append(distance)
                elif distance < far_range:
                    if is_frontal:
                        far_obstacles_front.append((distance, alignment))
                    else:
                        far_obstacles_other.append(distance)
            
            # Calculate directionally-weighted density metrics
            # Frontal obstacles are weighted by their alignment (more weight if directly in front)
            # and by proximity (closer obstacles have more weight)
            close_density_front = sum([(1.0 + 0.5 * align) * (1.0 - dist/close_range) for dist, align in close_obstacles_front]) * 1.2
            close_density_other = len(close_obstacles_other) * 0.8
            medium_density_front = sum([(1.0 + 0.3 * align) * (1.0 - dist/medium_range) for dist, align in medium_obstacles_front]) * 0.6
            medium_density_other = len(medium_obstacles_other) * 0.3
            far_density_front = sum([(1.0 + 0.1 * align) * (1.0 - dist/far_range) for dist, align in far_obstacles_front]) * 0.3
            far_density_other = len(far_obstacles_other) * 0.1
            
            # Combined density with directional weighting
            total_density = (
                close_density_front + close_density_other + 
                medium_density_front + medium_density_other + 
                far_density_front + far_density_other
            )
            
            # Scale the density to a [0,1] range
            obstacle_count = len(obstacle_distances)
            if obstacle_count > 0:
                # Normalize density by obstacle count, but with a cap to avoid too low density for few obstacles
                obstacle_density = min(1.0, total_density / max(obstacle_count, 5))
            else:
                obstacle_density = 0.0
            
            # IMPORTANT: Cap the maximum density to ensure it doesn't prevent movement
            # This prevents the density from being too high and blocking all paths
            obstacle_density = min(0.7, float(obstacle_density))
            
            # Find the nearest obstacle distance with a safe minimum value
            # Give priority to frontal obstacles by decreasing their effective distance
            nearest_obstacle_dist = 100.0
            
            for i, (position, distance) in enumerate(zip(obstacle_positions, obstacle_distances)):
                # Check if obstacle is in front
                obstacle_vector = np.array(position) - np.array(current_position)
                if np.linalg.norm(obstacle_vector) > 0:
                    obstacle_vector = obstacle_vector / np.linalg.norm(obstacle_vector)
                    alignment = np.dot(forward_vector, obstacle_vector)
                    
                    # For obstacles that block the path to target, give them higher priority
                    target_blocking_factor = np.dot(obstacle_vector, target_direction)
                    # Higher factor means the obstacle is more in line with the path to target
                    
                    # Adjust effective distance based on direction (frontal obstacles seem closer)
                    if alignment > 0:
                        # Scale by alignment: directly in front (alignment=1) reduces distance by up to 30%
                        # If the obstacle also blocks path to target, reduce distance further
                        blocking_factor = max(0, target_blocking_factor)
                        effective_distance = distance * (1.0 - 0.3 * alignment - 0.2 * blocking_factor)
                        if effective_distance < nearest_obstacle_dist:
                            nearest_obstacle_dist = effective_distance
                    else:
                        # Side/rear obstacles use actual distance
                        if distance < nearest_obstacle_dist:
                            nearest_obstacle_dist = distance
            
            # IMPORTANT: Ensure nearest_obstacle_dist is a valid value and not too low
            # This prevents obstacle avoidance from being too aggressive
            if np.isnan(nearest_obstacle_dist) or np.isinf(nearest_obstacle_dist) or nearest_obstacle_dist <= 0:
                nearest_obstacle_dist = 100.0  # Default to a safe value
            
            # IMPORTANT: For obstacle-free paths, ensure a high minimum distance
            # This ensures the drone prioritizes reaching the target when obstacles are far away
            if nearest_obstacle_dist > 20.0:
                nearest_obstacle_dist = 100.0
                
            # Calculate obstacle repulsion with improved exponential decay function
            # Higher repulsion when obstacles are closer
            base_repulsion = 0.0
            if nearest_obstacle_dist < medium_range:
                # Exponential decay: stronger effect for closer obstacles
                decay_factor = 0.3  # Controls how quickly repulsion decreases with distance
                base_repulsion = math.exp(-decay_factor * nearest_obstacle_dist)
                
                # Amplify repulsion when very close to obstacles
                if nearest_obstacle_dist < close_range:
                    # Additional scaling for close obstacles
                    base_repulsion *= (1.0 + (close_range - nearest_obstacle_dist) / close_range * 2.0)
            
            # Adjust repulsion based on frontal obstacle density
            # More weight for obstacles in front of the drone
            frontal_factor = 1.0
            if close_obstacles_front:
                # Calculate average alignment of close frontal obstacles
                avg_alignment = sum([align for _, align in close_obstacles_front]) / len(close_obstacles_front)
                # Higher factor for well-aligned obstacles (directly in path)
                frontal_factor = 1.0 + avg_alignment
            
            density_factor = 1.0 + (len(close_obstacles_front) * 0.3) * frontal_factor
            obstacle_repulsion_weight = base_repulsion * density_factor
            
            # IMPORTANT: Cap repulsion weight to prevent it from blocking progress completely
            # This ensures the drone can always find a path to the target
            obstacle_repulsion_weight = max(0.0, min(3.0, float(obstacle_repulsion_weight)))
            
            # Enhanced target proximity bonus - more gradual reduction based on distance to target
            # The closer to target, the less repulsed by obstacles (unless obstacles are very close)
            proximity_factor = 1.0  # Default - no reduction
            if target_distance < 10.0:  # Apply bonus within 10m of target (increased from 5m)
                # Gradual scaling factor based on distance to target
                # 1.0 (no reduction) at 10m, decreasing to 0.3 (70% reduction) at 1m
                proximity_factor = 0.3 + (0.7 * (target_distance / 10.0))
                
                # Safety check - maintain higher repulsion if obstacles are very close
                safety_threshold = 1.5  # Ensure full repulsion within this distance
                if nearest_obstacle_dist <= safety_threshold:
                    # No reduction when obstacles are very close
                    proximity_factor = 1.0
                elif nearest_obstacle_dist < 3.0:
                    # Gradual increase in repulsion as obstacles get closer to safety threshold
                    # Linear interpolation between full factor and calculated factor
                    safety_blend = (3.0 - nearest_obstacle_dist) / (3.0 - safety_threshold)
                    proximity_factor = safety_blend + proximity_factor * (1.0 - safety_blend)
                
                # Apply target proximity bonus
                obstacle_repulsion_weight *= proximity_factor
                
                # Debug output
                print(f"Target proximity bonus: {proximity_factor:.2f} at distance {target_distance:.2f}m")
            
            # Calculate direct path suitability when near target with orientation awareness
            direct_path_clear = True
            direct_path_suitability = 1.0
            
            # IMPROVED: When close to target (< 20m instead of 10m), always check if direct path is clear
            # This gives more priority to direct paths at longer ranges
            if target_distance < 20.0 and obstacle_positions:
                # Calculate target direction in global frame
                global_target_dir = np.array(target_direction)
                
                # Project obstacles onto the direct path to target
                direct_path_obstacles = []
                
                # Check each obstacle
                for i, (obs_pos, obs_dist) in enumerate(zip(obstacle_positions, obstacle_distances)):
                    # Vector from current position to obstacle
                    to_obstacle = np.array(obs_pos) - np.array(current_position)
                    
                    # Only consider obstacles closer than the target
                    if obs_dist < target_distance:
                        # Project obstacle onto target direction vector
                        dot_product = np.dot(to_obstacle, global_target_dir)
                        
                        # Projection distance along target vector
                        proj_dist = dot_product
                        
                        # If projection is positive and less than distance to target,
                        # obstacle might be in the way
                        if 0 < proj_dist < target_distance:
                            # Calculate perpendicular distance from obstacle to direct path
                            proj_vector = global_target_dir * proj_dist
                            perp_vector = to_obstacle - proj_vector
                            perp_dist = np.linalg.norm(perp_vector)
                            
                            # Adjust perception radius based on distance
                            # Obstacles further away need to be closer to the line to count as "blocking"
                            # IMPROVED: Smaller perception radius to give more weight to direct paths
                            perception_radius = 1.5 * (1.0 - 0.5 * (proj_dist / target_distance))
                            
                            # If obstacle is within perception radius of direct path, consider it blocking
                            if perp_dist < perception_radius:
                                # Store both distance to obstacle and how centered it is on path
                                blocking_score = obs_dist * (1.0 - perp_dist/perception_radius)
                                direct_path_obstacles.append((obs_dist, perp_dist, blocking_score))
                
                # If we found obstacles in the direct path
                if direct_path_obstacles:
                    # IMPORTANT: Sort by distance to find closest obstacle
                    direct_path_obstacles.sort(key=lambda x: x[0])
                    nearest_blocking = direct_path_obstacles[0][0]  # Distance to closest blocker
                    
                    # IMPROVED: Only consider path blocked if obstacle is actually close
                    if nearest_blocking < 5.0:
                        direct_path_clear = False
                        # Calculate suitability - higher for farther obstacles
                        direct_path_suitability = min(0.7, nearest_blocking / 10.0)
                    else:
                        # If obstacles are far enough, still consider path clear
                        direct_path_clear = True
                        # Slightly reduced suitability to indicate minor obstruction
                        direct_path_suitability = min(1.0, 0.7 + (nearest_blocking - 5.0) / 15.0)
                else:
                    # No obstacles in path
                    direct_path_clear = True
                    direct_path_suitability = 1.0
                    
                print(f"Direct path assessment: clear={direct_path_clear}, suitability={direct_path_suitability:.2f}")
            
            # Ensure direct_path_suitability is a valid value
            direct_path_suitability = max(0.0, min(1.0, float(direct_path_suitability)))
            
            # IMPORTANT: Calculate path suitability for adaptive waypoint radius
            # Higher value = clearer path, lower value = more obstacles
            # This allows larger steps in clear areas
            path_suitability = direct_path_suitability
            
            # If not checking direct path (far from target), calculate based on density and distance
            if target_distance >= 20.0:
                # Compute suitability (0.0-1.0) - lower when dense obstacles or close obstacles
                density_factor = max(0.0, min(1.0, 1.0 - obstacle_density))
                distance_factor = max(0.0, min(1.0, nearest_obstacle_dist / 20.0))
                
                # Combine factors, weighing distance more heavily
                path_suitability = distance_factor * 0.7 + density_factor * 0.3
            
            # Adjust obstacle repulsion when close to target
            if target_distance < 5.0:
                # Reduce repulsion weight when very close to target, unless obstacle is extremely close
                if nearest_obstacle_dist > 2.0:
                    obstacle_repulsion_weight *= 0.5
            
            # IMPORTANT: Ensure path_suitability is never too low to prevent movement
            path_suitability = max(0.3, float(path_suitability))
            
            # Adaptive waypoint radius based on suitability
            # When suitability is high (clear path), use larger radius to allow longer steps
            # When suitability is low (obstacles), use smaller radius for more careful movement
            min_radius = 2.0
            max_radius = min(15.0, target_distance * 0.75)  # Allow longer steps in clear areas, but never more than 75% of distance to target
            
            # Linear interpolation between min and max based on suitability
            waypoint_radius = min_radius + path_suitability * (max_radius - min_radius)
            
            # Enhanced observation with direct path information and orientation data
            observation = {
                'drone_position': current_position,
                'drone_orientation': current_orientation,
                'forward_vector': forward_vector.tolist(),
                'target_position': target_location,
                'target_direction': target_direction,
                'target_distance': float(target_distance),
                'target_alignment': float(target_alignment),
                'obstacle_density': float(obstacle_density),
                'obstacle_repulsion_weight': float(obstacle_repulsion_weight),
                'nearest_obstacle_distance': float(nearest_obstacle_dist),
                'direct_path_clear': direct_path_clear,
                'direct_path_suitability': float(direct_path_suitability),
                'path_suitability': float(path_suitability),
                'waypoint_radius': float(waypoint_radius),
                'obstacle_positions': obstacle_positions,
                'obstacle_distances': obstacle_distances,
                'iteration': iteration,
                'close_obstacle_count_front': len(close_obstacles_front),
                'close_obstacle_count_other': len(close_obstacles_other),
                'medium_obstacle_count_front': len(medium_obstacles_front),
                'medium_obstacle_count_other': len(medium_obstacles_other)
            }
            
            return observation
        except Exception as e:
            # Return a fallback observation with default values
            print(f"Error collecting sensor data: {str(e)}")
            traceback.print_exc()
            
            # In case of error, provide safe default values that encourage moving toward target
            target_vector = np.array(target_location) - np.array(current_position)
            target_distance = np.linalg.norm(target_vector)
            if target_distance > 0:
                target_direction = (target_vector / target_distance).tolist()
            else:
                target_direction = [1.0, 0.0, 0.0]
                
            return {
                'drone_position': current_position,
                'drone_orientation': self.current_orientation,
                'forward_vector': [1.0, 0.0, 0.0],
                'target_position': target_location,
                'target_direction': target_direction,
                'target_distance': float(target_distance) if np.isfinite(target_distance) else 10.0,
                'target_alignment': 0.0,
                'obstacle_density': 0.0,
                'obstacle_repulsion_weight': 0.0,
                'nearest_obstacle_distance': 100.0,
                'direct_path_clear': True,
                'direct_path_suitability': 1.0,
                'path_suitability': 1.0,
                'waypoint_radius': min(target_distance * 0.5, 15.0) if np.isfinite(target_distance) else 5.0,
                'obstacle_positions': [],
                'obstacle_distances': [],
                'iteration': iteration,
                'close_obstacle_count_front': 0,
                'close_obstacle_count_other': 0,
                'medium_obstacle_count_front': 0,
                'medium_obstacle_count_other': 0
            }
    
    def check_for_obstacles(self, safety_threshold=2.0):
        """Check if there are obstacles too close to the drone with orientation awareness
        
        Args:
            safety_threshold: Minimum safe distance to obstacles (meters)
            
        Returns:
            bool: True if obstacle is detected within threshold, False otherwise
        """
        try:
            # Get updated obstacle information and drone orientation
            self.update_drone_state()
            result = self.scanner.fetch_density_distances()
            
            # Validate the result type and structure
            if not isinstance(result, tuple) or len(result) != 2:
                print(f"⚠️ Invalid result from fetch_density_distances: {type(result)}")
                return False
                
            obstacle_positions, obstacle_distances = result
            
            # Validate data types for positions and distances
            if not isinstance(obstacle_positions, list):
                print(f"⚠️ obstacle_positions is not a list: {type(obstacle_positions)}")
                obstacle_positions = []
                
            if not isinstance(obstacle_distances, list):
                print(f"⚠️ obstacle_distances is not a list: {type(obstacle_distances)}")
                obstacle_distances = []
            
            # Early return if no obstacles
            if not obstacle_positions or not obstacle_distances:
                print("🔍 OBSTACLE CHECK: No obstacles detected")
                return False
            
            # Calculate direction of drone's movement (forward vector)
            # Extract forward vector from quaternion orientation
            w, x, y, z = self.current_orientation
            # Forward vector calculation from quaternion
            forward_x = 2 * (x*z + w*y)
            forward_y = 2 * (y*z - w*x)
            forward_z = 1 - 2 * (x*x + y*y)
            forward_vector = np.array([forward_x, forward_y, forward_z])
            
            # Normalize the forward vector (handle zero vector case)
            forward_norm = np.linalg.norm(forward_vector)
            if forward_norm > 1e-6:  # Check if vector magnitude is not too small
                forward_vector = forward_vector / forward_norm
            else:
                # Default forward vector if calculation fails
                forward_vector = np.array([1.0, 0.0, 0.0])
                print("⚠️ Forward vector calculation failed, using default [1,0,0]")
            
            # Debug: Print obstacle information
            print(f"🔍 OBSTACLE CHECK: Found {len(obstacle_distances)} objects within sensing range")
            
            # Enhanced obstacle detection that prioritizes obstacles in front of the drone
            closest_obstacle_distance = 100.0
            closest_frontal_distance = 100.0
            
            # Make sure we have the same number of positions and distances
            if len(obstacle_positions) != len(obstacle_distances):
                print(f"⚠️ Mismatch between positions ({len(obstacle_positions)}) and distances ({len(obstacle_distances)})")
                # Use the smaller length to avoid index errors
                num_obstacles = min(len(obstacle_positions), len(obstacle_distances))
            else:
                num_obstacles = len(obstacle_distances)
            
            # Process each obstacle using direct indexing instead of zip
            for i in range(num_obstacles):
                # Validate the data at this index
                if i >= len(obstacle_positions) or i >= len(obstacle_distances):
                    print(f"⚠️ Index {i} out of range for obstacle data")
                    continue
                    
                position = obstacle_positions[i]
                distance = obstacle_distances[i]
                
                # Validate position and distance
                if not isinstance(position, (list, tuple, np.ndarray)) or len(position) != 3:
                    print(f"⚠️ Invalid position at index {i}: {position}")
                    continue
                    
                if not isinstance(distance, (int, float)) or np.isnan(distance) or np.isinf(distance):
                    print(f"⚠️ Invalid distance at index {i}: {distance}")
                    continue
                
                # Convert position to numpy array for vector operations
                try:
                    obstacle_vector = np.array(position) - np.array(self.current_position)
                except Exception as e:
                    print(f"⚠️ Error calculating obstacle vector: {str(e)}")
                    continue
                
                # Normalize obstacle vector
                obstacle_norm = np.linalg.norm(obstacle_vector)
                if obstacle_norm > 1e-6:  # Check if vector magnitude is not too small
                    obstacle_vector = obstacle_vector / obstacle_norm
                else:
                    # Skip obstacles with zero magnitude (same position as drone)
                    continue
                
                # Calculate dot product to determine if obstacle is in front
                # Dot product > 0 means the obstacle is in front (acute angle with forward vector)
                dot_product = np.dot(forward_vector, obstacle_vector)
                
                # Check if obstacle is in front of the drone (within 90 degrees of forward direction)
                if dot_product > 0:
                    # Scale threshold based on how directly in front the obstacle is
                    # Frontal obstacles (dot_product close to 1) have stricter thresholds
                    frontal_factor = 0.5 + 0.5 * dot_product  # 0.5-1.0 based on alignment
                    effective_threshold = safety_threshold * (1 + frontal_factor)
                    
                    if distance < closest_frontal_distance:
                        closest_frontal_distance = distance
                    
                    # If frontal obstacle is within adjusted threshold, return obstacle detected
                    if distance < effective_threshold:
                        print(f"⚠️ FRONTAL OBSTACLE DETECTED at {distance:.2f}m! (effective threshold: {effective_threshold:.2f}m)")
                        return True
                else:
                    # For obstacles behind or to the side, use normal threshold
                    if distance < safety_threshold:
                        print(f"⚠️ SIDE/REAR OBSTACLE DETECTED at {distance:.2f}m! (threshold: {safety_threshold:.2f}m)")
                        return True
                
                # Track closest overall obstacle regardless of direction
                if distance < closest_obstacle_distance:
                    closest_obstacle_distance = distance
            
            # Print information about closest obstacles
            print(f"🔍 Closest obstacle: {closest_obstacle_distance:.2f}m, Closest frontal: {closest_frontal_distance:.2f}m")
            
            # As a final safety check, if any obstacle is extremely close, detect it regardless of direction
            if closest_obstacle_distance < safety_threshold * 0.5:
                print(f"⚠️ VERY CLOSE OBSTACLE DETECTED at {closest_obstacle_distance:.2f}m!")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking for obstacles: {str(e)}")
            traceback.print_exc()
            # Assume there's an obstacle in case of error (safer)
            return True
    
    def move_to_waypoint(self, waypoint):
        """Command the drone to move to the specified waypoint with obstacle avoidance and orientation control"""
        try:
            # Validate input waypoint
            if not isinstance(waypoint, list) or len(waypoint) != 3:
                print(f"⚠️ Invalid waypoint format: {waypoint}")
                return "error"
                
            # Check for NaN or Infinity values
            if any(not isinstance(w, (int, float)) or np.isnan(w) or np.isinf(w) for w in waypoint):
                print(f"⚠️ Invalid waypoint values: {waypoint}")
                return "error"
                
            print(f"Moving to: {[round(w, 1) for w in waypoint]}")
        
            # Ensure waypoint is within simulation boundaries and round coordinates
            waypoint = [
                round(min(max(float(waypoint[0]), -100), 100), 2),  # X between -100 and 100
                round(min(max(float(waypoint[1]), -100), 100), 2),  # Y between -100 and 100
                round(min(max(float(waypoint[2]), -20), 0), 2)      # Z between -20 and 0 (remember NED)
            ]
            
            # Check if waypoint is too close to current position
            distance_to_waypoint = sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, waypoint)))
            if distance_to_waypoint < 0.1:
                print(f"Waypoint {waypoint} too close to current position ({self.current_position}), stabilizing in hover")
                self.client.hoverAsync().join()
                time.sleep(0.5)  # Short pause to ensure stability
                return "success"
            
            # Calculate yaw angle to face direction of movement
            yaw_angle = self._calculate_yaw_angle(self.current_position, waypoint)
            
            try:
                # Start movement asynchronously with yaw control
                movement_task = self.client.moveToPositionAsync(
                    x=waypoint[0],
                    y=waypoint[1],
                    z=waypoint[2],
                    velocity=2,
                    timeout_sec=15,  # Timeout after 15 seconds
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_angle),  # Set yaw angle
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
                no_movement_count = 0  # Counter for consecutive no-movement checks
                
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
                    
                    # Increment no-movement counter if drone is stationary
                    if distance_moved < 0.01:
                        no_movement_count += 1
                    else:
                        no_movement_count = 0  # Reset if there's movement
                    
                    # Determine if movement is complete by checking:
                    # 1. If we're very close to the target
                    # 2. Or if we've stopped moving for several consecutive checks
                    # 3. Or if we've timed out
                    if current_distance_to_waypoint < 0.5:  # Within 0.5m of target
                        print(f"Reached waypoint (distance: {current_distance_to_waypoint:.2f}m)")
                        movement_complete = True
                    elif no_movement_count >= 3 and time.time() - start_time > 2:  # No movement for ~0.3s after initial 2s
                        print(f"Movement stopped at distance {current_distance_to_waypoint:.2f}m from waypoint")
                        movement_complete = True
                    elif time.time() - start_time > 15:  # Timeout after 15 seconds
                        print("Movement timed out")
                        movement_complete = True
                    
                    # Check for obstacles
                    if self.check_for_obstacles(safety_threshold=2.5):
                        print("🛑 Obstacle detected! Stopping.")
                        self.client.cancelLastTask()  # Cancel the movement task
                        self.client.hoverAsync().join()  # Hover in place
                        obstacle_detected = True
                        break
                
                # Once movement is complete, ensure the drone is hovering stably
                print("Stabilizing in hover mode...")
                hover_task = self.client.hoverAsync()
                hover_task.join()
                
                # Add a short pause to ensure the drone has settled
                time.sleep(0.5)
                
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
                
                if final_distance > 0.3 and (is_final_approach and distance_to_target > ARRIVAL_THRESHOLD * 0.8):
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
                        
                        # Calculate yaw for precision movement
                        precision_yaw = self._calculate_yaw_angle(self.current_position, final_waypoint)
                        
                        # Start precision movement with yaw control
                        precision_task = self.client.moveToPositionAsync(
                            x=final_waypoint[0],
                            y=final_waypoint[1],
                            z=final_waypoint[2],
                            velocity=approach_velocity,  # Slower for precision
                            timeout_sec=15,
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=precision_yaw),  # Set precise yaw
                            lookahead=-1,
                            adaptive_lookahead=1
                        )
                        
                        # Monitor for obstacles during precision movement
                        start_time = time.time()
                        previous_position = self.current_position.copy()
                        movement_complete = False
                        no_movement_count = 0
                        
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
                            
                            # Track consecutive no-movement checks
                            if distance_moved < 0.005:  # Smaller threshold for precision movement
                                no_movement_count += 1
                            else:
                                no_movement_count = 0  # Reset if there's movement
                            
                            # Check if movement is complete
                            precision_threshold = 0.2 if is_final_approach else 0.3
                            
                            if current_distance < precision_threshold:
                                print(f"Precision target reached (distance: {current_distance:.2f}m)")
                                movement_complete = True
                            elif is_final_approach and target_distance < ARRIVAL_THRESHOLD:
                                print(f"Target reached: {target_distance:.2f}m")
                                movement_complete = True
                            elif no_movement_count >= 5 and time.time() - start_time > 3:  # No movement for ~0.5s after 3s
                                print(f"Precision movement stopped at {current_distance:.2f}m from target")
                                movement_complete = True
                            elif time.time() - start_time > 15:  # Timeout after 15 seconds
                                print("Precision movement timed out")
                                movement_complete = True
                            
                            # Check for obstacles
                            if self.check_for_obstacles(safety_threshold=2.0):
                                print("🛑 Obstacle detected during precision movement!")
                                self.client.cancelLastTask()
                                self.client.hoverAsync().join()
                                return "obstacle_detected"
                        
                        # Ensure final stabilization with explicit hover command
                        print("Final stabilization...")
                        hover_task = self.client.hoverAsync()
                        hover_task.join()
                        time.sleep(0.5)  # Additional pause to ensure stability
                
                # Update final position
                self.update_drone_state()
                
                # Final hover to prevent drift
                self.client.hoverAsync().join()
            
                # Return success
                return "success"
                
            except Exception as e:
                print(f"Error during movement: {str(e)}")
                traceback.print_exc()
                
                # Try to safely hover in place if there was an error
                try:
                    self.client.hoverAsync().join()
                except:
                    pass
                    
                return "error"
                
        except Exception as e:
            print(f"Critical error in move_to_waypoint: {str(e)}")
            traceback.print_exc()
            
            # Try to stabilize the drone in case of error
            try:
                self.client.hoverAsync().join()
            except:
                pass
                
            return "error"
    
    def distance_to_target(self):
        """Calculate distance to target"""
        return np.linalg.norm(np.array(self.current_position) - np.array(self.target_location))
    
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
        """
        Rotate a 3D vector using a quaternion rotation.
        
        Args:
            vector: 3D vector [x, y, z] to be rotated
            quaternion: Quaternion in format [w, x, y, z] where w is the scalar part
            
        Returns:
            numpy.ndarray: The rotated vector
        """
        # Ensure inputs are numpy arrays
        v = np.array(vector, dtype=float)
        q = np.array(quaternion, dtype=float)
        
        # Extract quaternion components
        w, x, y, z = q
        
        # Compute the quaternion-vector-quaternion conjugate product:
        # v' = q * v * q^-1
        
        # First, construct a quaternion from the vector (with w=0)
        v_quat = np.array([0, v[0], v[1], v[2]])
        
        # Quaternion multiplication: q * v_quat
        q_times_v = np.array([
            -x*v_quat[1] - y*v_quat[2] - z*v_quat[3],
            w*v_quat[1] + y*v_quat[3] - z*v_quat[2],
            w*v_quat[2] + z*v_quat[1] - x*v_quat[3],
            w*v_quat[3] + x*v_quat[2] - y*v_quat[1]
        ])
        
        # Compute conjugate of q (negate vector part)
        q_conj = np.array([w, -x, -y, -z])
        
        # Quaternion multiplication: (q * v_quat) * q_conj
        result_quat = np.array([
            -q_times_v[1]*q_conj[1] - q_times_v[2]*q_conj[2] - q_times_v[3]*q_conj[3],
            q_times_v[0]*q_conj[1] + q_times_v[2]*q_conj[3] - q_times_v[3]*q_conj[2],
            q_times_v[0]*q_conj[2] + q_times_v[3]*q_conj[1] - q_times_v[1]*q_conj[3],
            q_times_v[0]*q_conj[3] + q_times_v[1]*q_conj[2] - q_times_v[2]*q_conj[1]
        ])
        
        # The vector part of the resulting quaternion is our rotated vector
        return result_quat[1:4]

    def _is_orientation_valid(self):
        """Check if the drone's orientation is valid."""
        try:
            orientation = self.drone.get_orientation()
            return (
                orientation is not None 
                and len(orientation) == 4
                and all(isinstance(x, (int, float)) for x in orientation)
                and all(not (np.isnan(x) or np.isinf(x)) for x in orientation)
            )
        except Exception:
            return False
            
    def _is_position_valid(self):
        """Check if the drone's position is valid."""
        try:
            position = self.drone.get_position()
            return (
                position is not None 
                and len(position) == 3
                and all(isinstance(x, (int, float)) for x in position)
                and all(not (np.isnan(x) or np.isinf(x)) for x in position)
            )
        except Exception:
            return False

    def convert_to_global_waypoint(self, egocentric_waypoint):
        """
        Convert an egocentric waypoint (relative to drone's current position and orientation)
        to a global waypoint in the world coordinate system.
        
        Args:
            egocentric_waypoint: A list or tuple [x, y, z] where:
                x: forward distance (positive in front, negative behind)
                y: right distance (positive right, negative left)
                z: up distance (positive up, negative down)
                
        Returns:
            list: Global waypoint [x, y, z] in the world coordinate system
        """
        try:
            # Get current drone position and orientation
            if not self._is_position_valid() or not self._is_orientation_valid():
                print("Error: Invalid drone position or orientation")
                return None
                
            current_position = self.current_position
            current_orientation = self.current_orientation
            
            # Validate egocentric waypoint
            if not egocentric_waypoint or len(egocentric_waypoint) != 3:
                print(f"Error: Invalid egocentric waypoint format: {egocentric_waypoint}")
                return None
                
            if not all(isinstance(val, (int, float)) for val in egocentric_waypoint):
                print(f"Error: Waypoint contains non-numeric values: {egocentric_waypoint}")
                return None
                
            if any(np.isnan(val) or np.isinf(val) for val in egocentric_waypoint):
                print(f"Error: Waypoint contains NaN or Inf values: {egocentric_waypoint}")
                return None
            
            # Convert to numpy arrays for calculations
            ego_waypoint = np.array(egocentric_waypoint, dtype=float)
            position = np.array(current_position, dtype=float)
            
            # Debug information
            print(f"Converting egocentric waypoint to global:")
            print(f"  Current position: {position}")
            print(f"  Egocentric waypoint: {ego_waypoint}")
            
            # AirSim quaternion is in [w, x, y, z] format - normalize to ensure unit quaternion
            quaternion = np.array(current_orientation, dtype=float)
            quat_magnitude = np.sqrt(np.sum(quaternion**2))
            
            if quat_magnitude < 1e-6:
                print(f"Error: Invalid quaternion magnitude: {quat_magnitude}")
                return None
                
            # Normalize quaternion
            quaternion = quaternion / quat_magnitude
            
            # Apply the quaternion rotation to transform the egocentric vector to global frame
            rotated_vector = self._rotate_vector_by_quaternion(ego_waypoint, quaternion)
            
            # Calculate global waypoint by adding the rotated vector to current position
            global_waypoint = position + rotated_vector
            
            # Format result as a list with 3 decimal precision
            result = [round(float(val), 3) for val in global_waypoint]
            
            print(f"  Global waypoint: {result}")
            return result
            
        except Exception as e:
            print(f"Error in convert_to_global_waypoint: {str(e)}")
            return None

    def inspect_suitability_metric(self, expected_state):
        """Analyze path planning suitability to diagnose potential issues"""
        # Extract key metrics from expected state
        suitability = expected_state.get('suitability', 0)
        distance = expected_state.get('distance', 0)
        
        print("\n==== PATH PLANNING INSIGHTS ====")
        
        # Analyze suitability value
        if suitability < 0.3:
            print(f"⚠️ Low suitability ({suitability:.2f}) indicating path planning concerns")
        else:
            print(f"✓ Good suitability ({suitability:.2f}) indicating clear path")
        
        # Compare with actual distance
        actual_distance = self.distance_to_target()
        distance_diff = abs(actual_distance - distance)
        if distance_diff > 2:
            print(f"⚠️ Significant discrepancy between actual ({actual_distance:.2f}m) and estimated ({distance:.2f}m) distance")
        
        # Check for nearby obstacles
        lidar_data = self.get_lidar_data()
        min_distance = min(lidar_data) if lidar_data else 100.0
        
        if min_distance < 3.0:
            print(f"⚠️ Very close obstacle detected ({min_distance:.2f}m)")
        elif min_distance < 5.0:
            print(f"ℹ️ Nearby obstacle ({min_distance:.2f}m)")
        else:
            print(f"✓ No very close obstacles (nearest: {min_distance:.2f}m)")
        
        # Analyze suitability for current scenario
        if suitability < 0.3 and min_distance > 5.0:
            print("❌ Path planning issue: Low suitability despite no close obstacles")
        
        if actual_distance > 15 and suitability < 0.4:
            print("❌ Path planning issue: Far from target but has low suitability")
        
        print("===============================")
    
    def get_lidar_data(self):
        """Get lidar data from AirSim"""
        lidar_data = []
        try:
            lidar_data = self.client.getLidarData()
        except Exception as e:
            print(f"Error getting lidar data: {str(e)}")
        return lidar_data

    def navigate_to_target(self, zmq_interface=None):
        """Execute the navigation to the target location using AirSim and Active Inference
        
        Args:
            zmq_interface: ZMQInterface instance for communicating with Julia
            
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        print("\n🚀 Starting autonomous navigation to target...")
        print(f"Target location: {self.target_location}")
        
        # Safety check - ensure we have a ZMQ interface
        if zmq_interface is None:
            print("⚠️ No ZMQ interface provided - creating default instance")
            zmq_interface = ZMQInterface()
        
        # Initialize performance tracking
        start_time = time.time()
        iteration = 0
        closest_distance = float('inf')
        movement_attempts = 0
        success = False
        
        try:
            # Continue until we reach the target or exceed max iterations
            while True:
                iteration += 1
                print(f"\n===== ITERATION {iteration} =====")
                
                # Check if we've reached the target
                self.update_drone_state()
                current_distance = self.distance_to_target()
                print(f"Current position: {[round(p, 2) for p in self.current_position]}")
                print(f"Distance to target: {round(current_distance, 2)}m")
                
                if current_distance < closest_distance:
                    closest_distance = current_distance
                    print(f"✓ New closest approach: {round(closest_distance, 2)}m")
                
                # Check for obstacles in the environment
                obstacle_detected = self.check_for_obstacles(safety_threshold=2.0)
                
                # Check if we're close to target - switch to direct approach mode for final positioning
                if current_distance < 5.0 and not obstacle_detected:
                    print(f"\n🏁 Entering direct approach mode at {current_distance:.2f}m from target")
                    
                    # Direct approach to target with smaller steps
                    direct_approach_success = self._direct_approach_to_target()
                    
                    if direct_approach_success:
                        print(f"🎯 Direct approach successful! Distance: {self.distance_to_target():.2f}m")
                        success = True
                        break
                    else:
                        print("⚠️ Direct approach interrupted, continuing with regular navigation")
                
                # Check arrival condition
                if current_distance < ARRIVAL_THRESHOLD:
                    print(f"\n🎯 Target reached! Distance: {round(current_distance, 2)}m")
                    success = True
                    break
                
                if iteration > MAX_ITERATIONS:
                    print(f"\n⚠️ Maximum iterations ({MAX_ITERATIONS}) reached.")
                    break
                
                # Collect sensor data about drone state and environment
                observation = self.collect_sensor_data(
                    self, self.scanner, self.current_position, self.target_location, iteration
                )
                
                # Calculate path suitability for adaptive waypoint radius
                # Higher value = clearer path, lower value = more obstacles
                path_suitability = 1.0
                
                # Calculate based on obstacle density and nearest obstacle
                if 'obstacle_density' in observation and 'nearest_obstacle_distance' in observation:
                    # Path is less suitable with high density and close obstacles
                    obstacle_density = observation['obstacle_density']
                    nearest_obstacle = observation['nearest_obstacle_distance']
                    
                    # Compute suitability (0.0-1.0) - lower when dense obstacles or close obstacles
                    density_factor = max(0.0, min(1.0, 1.0 - obstacle_density * 0.8))
                    distance_factor = max(0.0, min(1.0, nearest_obstacle / 20.0))
                    
                    # Combine factors, weighing distance more heavily
                    path_suitability = distance_factor * 0.7 + density_factor * 0.3
                    
                    # Include direct path information if available
                    if 'direct_path_suitability' in observation:
                        direct_path_factor = observation['direct_path_suitability']
                        # Blend with direct path suitability (30% influence)
                        path_suitability = path_suitability * 0.7 + direct_path_factor * 0.3
                
                # Add suitability to observation for use by the planner
                observation['path_suitability'] = path_suitability
                
                # Adaptive waypoint radius based on suitability
                # When suitability is high (clear path), use larger radius to allow longer steps
                # When suitability is low (obstacles), use smaller radius for more careful movement
                base_waypoint_radius = 5.0  # Base value
                min_radius = 2.0
                max_radius = 12.0  # Increased from 10.0 to allow longer steps in clear areas
                
                # Ensure path_suitability is a valid float between 0 and 1
                try:
                    path_suitability = float(path_suitability)
                    path_suitability = max(0.0, min(1.0, path_suitability))
                except (ValueError, TypeError):
                    print(f"⚠️ Invalid path_suitability value: {path_suitability}, using default 0.5")
                    path_suitability = 0.5
                
                if path_suitability > 0.8:
                    # Very clear path - use larger radius for longer steps
                    waypoint_radius = max_radius
                elif path_suitability < 0.3:
                    # Very cluttered path - use smaller radius for careful movement
                    waypoint_radius = min_radius
                else:
                    # Linear interpolation between min and max
                    # Maps suitability 0.3-0.8 to radius min-max (opposite of before)
                    normalized = (path_suitability - 0.3) / 0.5  # 0.0 to 1.0
                    waypoint_radius = min_radius + normalized * (max_radius - min_radius)
                
                # Ensure waypoint_radius is a valid float value
                try:
                    waypoint_radius = float(waypoint_radius)
                    waypoint_radius = max(min_radius, min(max_radius, waypoint_radius))
                except (ValueError, TypeError):
                    print(f"⚠️ Invalid waypoint_radius calculation result: {waypoint_radius}, using default {base_waypoint_radius}")
                    waypoint_radius = base_waypoint_radius
                
                # Add waypoint radius to observation for use by planner
                observation['waypoint_radius'] = waypoint_radius
                print(f"Path suitability: {path_suitability:.2f}, Waypoint radius: {waypoint_radius:.2f}m")
                
                # Get waypoint using active inference via ZMQ
                next_waypoint = None
                policy = None
                
                try:
                    # Send the observation to the Julia server via ZMQ
                    print("Sending observation to ZMQ server...")
                    result = zmq_interface.send_observation_and_receive_action(observation)
                    
                    # Check if result is None (failed communication)
                    if result is None or not isinstance(result, tuple) or len(result) != 2:
                        raise ValueError(f"Invalid result from ZMQ interface: {result}")
                    
                    next_waypoint, policy = result
                    
                    # Check if next_waypoint is None or invalid
                    if next_waypoint is None or not isinstance(next_waypoint, list) or len(next_waypoint) != 3:
                        raise ValueError(f"Invalid waypoint received from ZMQ: {next_waypoint}")
                    
                    print(f"Received action from ZMQ server: {[round(w, 2) for w in next_waypoint]}")
                    
                    # IMPORTANT: Verify the waypoint magnitude is reasonable
                    waypoint_magnitude = sqrt(sum(x*x for x in next_waypoint))
                    print(f"Waypoint magnitude: {waypoint_magnitude:.2f}m")
                    
                    # Set minimum and maximum allowed magnitudes
                    min_magnitude = 0.1  # Minimum to prevent zero movement
                    max_magnitude = 15.0  # Maximum to prevent too large movements
                    
                    # If waypoint magnitude is outside of reasonable bounds, scale it
                    if waypoint_magnitude < min_magnitude:
                        # Too small - scale up to minimum
                        if waypoint_magnitude > 0:
                            scaling_factor = min_magnitude / waypoint_magnitude
                            next_waypoint = [x * scaling_factor for x in next_waypoint]
                            print(f"⚠️ Waypoint too small, scaled up: {[round(w, 2) for w in next_waypoint]}")
                        else:
                            # If magnitude is zero, create a small step in the target direction
                            target_vector = [
                                self.target_location[0] - self.current_position[0],
                                self.target_location[1] - self.current_position[1],
                                self.target_location[2] - self.current_position[2]
                            ]
                            target_distance = sqrt(sum(x*x for x in target_vector))
                            if target_distance > 0.1:
                                next_waypoint = [x * min_magnitude / target_distance for x in target_vector]
                                print(f"⚠️ Zero waypoint, using minimal step toward target: {[round(w, 2) for w in next_waypoint]}")
                            else:
                                next_waypoint = [min_magnitude, 0, 0]  # Default minimal forward step
                                print(f"⚠️ Zero waypoint and at target, using minimal forward step: {next_waypoint}")
                    elif waypoint_magnitude > max_magnitude:
                        # Too large - scale down to maximum
                        scaling_factor = max_magnitude / waypoint_magnitude
                        next_waypoint = [x * scaling_factor for x in next_waypoint]
                        print(f"⚠️ Waypoint too large, scaled down: {[round(w, 2) for w in next_waypoint]}")
                    
                    # IMPORTANT: Verify the waypoint makes progress toward the target
                    # Compute the vector from current position to target
                    target_vector = [
                        self.target_location[0] - self.current_position[0],
                        self.target_location[1] - self.current_position[1],
                        self.target_location[2] - self.current_position[2]
                    ]
                    target_distance = sqrt(sum(x*x for x in target_vector))
                    
                    # Normalize target vector
                    if target_distance > 0.1:
                        target_unit_vector = [x/target_distance for x in target_vector]
                    else:
                        target_unit_vector = [1.0, 0.0, 0.0]  # Default forward direction
                    
                    # Compute the dot product to measure alignment with target direction
                    waypoint_magnitude = sqrt(sum(x*x for x in next_waypoint))
                    if waypoint_magnitude > 0.1:
                        waypoint_unit_vector = [x/waypoint_magnitude for x in next_waypoint]
                        alignment = sum(a*b for a, b in zip(target_unit_vector, waypoint_unit_vector))
                    else:
                        alignment = 0.0
                    
                    print(f"Target alignment: {alignment:.2f} (-1 to 1, 1 is perfect alignment)")
                    
                    # If alignment is negative or very low and no obstacles, use direct approach
                    if (alignment < 0.3 or waypoint_magnitude < 0.5) and observation['obstacle_density'] < 0.1:
                        print("⚠️ Waypoint does not make progress toward target. Using direct approach.")
                        
                        # Calculate step size based on target distance
                        if target_distance > 20:
                            # Far from target - large steps (up to 10m)
                            step_size = min(10.0, target_distance * 0.3)
                        elif target_distance > 5:
                            # Medium distance - moderate steps (up to 3m)
                            step_size = min(3.0, target_distance * 0.25)
                        else:
                            # Close to target - small steps for precision (up to 1m)
                            step_size = min(1.0, target_distance * 0.2)
                        
                        # Create direct waypoint toward target
                        next_waypoint = [v * step_size for v in target_unit_vector]
                        print(f"Using direct waypoint: {[round(w, 2) for w in next_waypoint]}")
                    
                except Exception as e:
                    print(f"⚠️ Error in ZMQ communication: {str(e)}")
                    print("Using fallback direct approach to target...")
                    
                    # Calculate direct vector to target
                    direction = [
                        self.target_location[0] - self.current_position[0],
                        self.target_location[1] - self.current_position[1],
                        self.target_location[2] - self.current_position[2]
                    ]
                    
                    # IMPROVED: Calculate magnitude once for efficiency and clear step size calculation
                    magnitude = sqrt(sum(d*d for d in direction))
                    
                    # IMPORTANT: Never use zero magnitude even if at target
                    if magnitude < 0.1:
                        print("Very close to target, making minimal movement")
                        magnitude = 0.1
                        direction = [0.1, 0, 0]  # Minimal forward movement
                    
                    # Use adaptive step size based on distance to target
                    # - Closer to target: smaller steps for precision
                    # - Further from target: larger steps for efficiency
                    target_distance = magnitude
                    
                    # IMPROVED: Calculate optimal step size based on distance to target
                    if target_distance > 20:
                        # Far from target - large steps (up to 15m)
                        max_step = min(15.0, target_distance * 0.4)
                    elif target_distance > 5:
                        # Medium distance - moderate steps (up to 5m)
                        max_step = min(5.0, target_distance * 0.3)
                    else:
                        # Close to target - small steps for precision (up to 1m)
                        max_step = min(1.0, target_distance * 0.3)
                    
                    # Calculate unit vector toward target
                    unit_vector = [d / magnitude for d in direction]
                    
                    # Create waypoint using calculated step size
                    next_waypoint = [d * max_step for d in unit_vector]
                    
                    # Print details of the direct approach
                    print(f"Direct approach: distance={target_distance:.2f}m, step={max_step:.2f}m")
                    print(f"Direction: {[round(d, 2) for d in unit_vector]}")
                    
                    policy = [next_waypoint] * 3  # Simple policy repeating the same action
                
                # Convert the egocentric waypoint to global coordinates
                global_waypoint = self.convert_to_global_waypoint(next_waypoint)
                
                # Execute the movement
                movement_attempts += 1
                movement_result = self.move_to_waypoint(global_waypoint)
                
                # Check movement result
                if movement_result == "obstacle_detected":
                    print("⚠️ Movement interrupted by obstacle detection. Replanning...")
                    continue
                elif movement_result == "error":
                    print("⚠️ Error during movement. Replanning...")
                    continue
                
                # Basic safety check - detect if we're not making progress
                if movement_attempts >= 5:
                    self.update_drone_state()
                    current_distance = self.distance_to_target()
                    
                    # If we haven't made significant progress towards the target
                    if current_distance > closest_distance * 0.95:
                        print(f"⚠️ No significant progress after {movement_attempts} movements.")
                        print(f"Current: {round(current_distance, 2)}m, Best: {round(closest_distance, 2)}m")
                        
                        # IMPROVED: Always try a direct approach when stuck, not just randomly
                        print("Attempting direct approach to target...")
                        
                        # Calculate direct vector to target
                        direction = [
                            self.target_location[0] - self.current_position[0],
                            self.target_location[1] - self.current_position[1],
                            self.target_location[2] - self.current_position[2]
                        ]
                        
                        # Calculate magnitude
                        magnitude = sqrt(sum(d*d for d in direction))
                        
                        # Determine optimal step size (more aggressive when stuck)
                        if magnitude > 20:
                            # Far from target - large steps
                            step_size = min(15.0, magnitude * 0.5)
                        elif magnitude > 5:
                            # Medium distance - moderate steps
                            step_size = min(5.0, magnitude * 0.4) 
                        else:
                            # Close to target - small steps for precision
                            step_size = min(2.0, magnitude * 0.5)
                        
                        # Ensure we're not making tiny moves when stuck
                        step_size = max(0.5, step_size)
                        
                        if magnitude > 0:
                            # Calculate unit vector and apply step size
                            next_waypoint = [d * step_size / magnitude for d in direction]
                            
                            # Calculate waypoint
                            global_waypoint = self.convert_to_global_waypoint(next_waypoint)
                            
                            # Execute direct movement with orientation control
                            print(f"Executing direct approach with step size {step_size:.2f}m...")
                            self.move_to_waypoint(global_waypoint)
                    
                    # Reset counter
                    movement_attempts = 0
            
            # Navigation complete - report results
            elapsed_time = time.time() - start_time
            print(f"\n==== NAVIGATION SUMMARY ====")
            print(f"Result: {'SUCCESS ✅' if success else 'INCOMPLETE ❌'}")
            print(f"Iterations: {iteration}")
            print(f"Final distance: {round(self.distance_to_target(), 2)}m")
            print(f"Closest approach: {round(closest_distance, 2)}m")
            print(f"Time taken: {round(elapsed_time, 1)} seconds")
            print(f"============================\n")
            
            return success
            
        except Exception as e:
            print(f"❌ Error during navigation: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            # Always close the ZMQ connection
            if zmq_interface:
                print("Closing ZMQ connection...")
                zmq_interface.close()

    def _calculate_yaw_angle(self, current_position, target_position):
        """Calculate the yaw angle (in degrees) required to face toward the target
        
        Args:
            current_position: Current drone position [x, y, z]
            target_position: Target position [x, y, z]
            
        Returns:
            float: Yaw angle in degrees (-180 to 180)
        """
        # Calculate direction vector in the XY plane
        dx = target_position[0] - current_position[0]
        dy = target_position[1] - current_position[1]
        
        # Calculate yaw angle in radians
        # In AirSim's NED coordinate system:
        # - Positive X is North
        # - Positive Y is East
        # - So 0 degrees yaw points North (along positive X)
        yaw_rad = math.atan2(dy, dx)
        
        # Convert to degrees
        yaw_deg = math.degrees(yaw_rad)
        
        print(f"Direction vector: [{dx:.2f}, {dy:.2f}], Yaw angle: {yaw_deg:.1f}°")
        
        return yaw_deg

    def _direct_approach_to_target(self, max_attempts=5, safety_threshold=1.5):
        """Execute a direct approach to target with small, precise steps
        
        Args:
            max_attempts: Maximum number of movement attempts
            safety_threshold: Safety distance for obstacle detection
            
        Returns:
            bool: True if target was reached, False if approach was interrupted
        """
        print("\n==== DIRECT APPROACH TO TARGET ====")
        
        for attempt in range(1, max_attempts + 1):
            # Update current position
            self.update_drone_state()
            current_distance = self.distance_to_target()
            
            # Check if we've reached the target
            if current_distance < ARRIVAL_THRESHOLD:
                print(f"✅ Target reached in direct approach! Distance: {current_distance:.2f}m")
                return True
                
            print(f"Direct approach attempt {attempt}/{max_attempts}, distance: {current_distance:.2f}m")
            
            # Check for obstacles
            if self.check_for_obstacles(safety_threshold=safety_threshold):
                print("⚠️ Obstacle detected during direct approach, aborting")
                return False
                
            # Calculate direct vector to target
            direction = [
                self.target_location[0] - self.current_position[0],
                self.target_location[1] - self.current_position[1],
                self.target_location[2] - self.current_position[2]
            ]
            
            # Calculate magnitude
            magnitude = sqrt(sum(d*d for d in direction))
            
            # Use decreasing step sizes as we get closer
            if current_distance > 3.0:
                step_size = min(1.0, current_distance * 0.3)
            else:
                step_size = min(0.5, current_distance * 0.5)
                
            # Ensure minimum step size
            step_size = max(0.2, step_size)
            
            # Calculate waypoint
            if magnitude > 0:
                next_waypoint = [d * step_size / magnitude for d in direction]
                global_waypoint = self.convert_to_global_waypoint(next_waypoint)
                
                # Execute movement with lower velocity for precision
                print(f"Moving directly toward target with step size {step_size:.2f}m")
                
                try:
                    # Use moveToPositionAsync directly with low velocity for precision
                    move_task = self.client.moveToPositionAsync(
                        x=global_waypoint[0],
                        y=global_waypoint[1],
                        z=global_waypoint[2],
                        velocity=1.0,  # Lower velocity for precision
                        timeout_sec=10,
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        lookahead=-1,
                        adaptive_lookahead=1
                    )
                    
                    # Wait for completion
                    move_task.join()
                    
                    # Hover to stabilize
                    self.client.hoverAsync().join()
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error during direct approach: {str(e)}")
                    return False
        
        # Final check if we've reached the target
        self.update_drone_state()
        current_distance = self.distance_to_target()
        
        if current_distance < ARRIVAL_THRESHOLD:
            print(f"✅ Target reached in direct approach! Final distance: {current_distance:.2f}m")
            return True
        else:
            print(f"❌ Failed to reach target in direct approach. Distance: {current_distance:.2f}m")
            return False

def ensure_zmq_server_ready():
    """Ensures the ZMQ server is running and ready to receive connections
    
    Returns:
        bool: True if server is ready, False otherwise
    """
    print("\n==== Checking ZMQ Server Status ====")
    
    # First check if Julia process is running
    julia_running = False
    if platform.system() == "Windows":
        result = subprocess.run(["tasklist"], capture_output=True, text=True)
        if "julia" in result.stdout.lower():
            print("✅ Julia process found running")
            julia_running = True
        else:
            print("⚠️ No Julia process found running")
    
    # Use the single default port for better reliability
    server_address = f"tcp://localhost:{DEFAULT_ZMQ_PORT}"
    print(f"Checking ZMQ server at {server_address}...")
    
    try:
        # Create a connection with timeout
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 500)
        socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
        socket.setsockopt(zmq.SNDTIMEO, 2000)  # 2 second timeout
        
        # Connect to server
        try:
            socket.connect(server_address)
        except zmq.ZMQError as e:
            print(f"Error connecting to server: {str(e)}")
            socket.close()
            context.term()
            
            if not julia_running:
                print("No Julia process running, attempting to start ZMQ server...")
                return _start_zmq_server()
            return False
        
        # Try a simple ping with polling for better timeout handling
        try:
            # Use polling for more reliable timeout
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            
            # Send ping
            socket.send_string('{"ping": true}')
            
            # Wait for response with timeout
            if poller.poll(3000):  # 3 second timeout
                response = socket.recv_string()
                print("✅ ZMQ server responded")
                socket.close()
                context.term()
                return True
            else:
                print("⚠️ Timeout waiting for response from ZMQ server")
        except zmq.ZMQError as e:
            print(f"⚠️ ZMQ error during ping test: {str(e)}")
        
        # Clean up
        socket.close()
        context.term()
        
        # Server not responding, try to start it
        if not julia_running:
            print("No response from ZMQ server and no Julia process running")
            print("Attempting to start ZMQ server...")
            return _start_zmq_server()
        else:
            print("Julia process is running but ZMQ server is not responding")
            print("The Julia server may be running on a different port or have an issue")
            return False
        
    except Exception as e:
        print(f"❌ Error checking ZMQ server: {str(e)}")
        traceback.print_exc()
        return False

def _start_zmq_server():
    """Helper function to start the ZMQ server
    
    Returns:
        bool: True if server was started successfully, False otherwise
    """
    try:
        # Create a ZMQInterface instance to use its _start_server method
        interface = ZMQInterface()
        
        # Try to start the server
        if interface._start_server():
            print("✅ Successfully started ZMQ server")
            # Wait a moment for server to initialize
            time.sleep(2)
            return True
        else:
            print("❌ Failed to start ZMQ server automatically")
            print("\nPlease start the Julia ZMQ server manually in a separate terminal:")
            
            # Determine correct script path to suggest
            script_path = "zmq_server.jl"
            if not os.path.exists(script_path):
                script_path = os.path.join("actinf", "zmq_server.jl")
            
            print(f"\njulia --project=. {script_path}\n")
            return False
    except Exception as e:
        print(f"❌ Error starting ZMQ server: {str(e)}")
        traceback.print_exc()
        return False

def wait_for_server(zmq_interface, max_attempts=30, delay=1.0):
    """Poll and wait for the Julia ZMQ server to become ready
    
    Args:
        zmq_interface: Initialized ZMQInterface instance
        max_attempts: Maximum number of connection attempts
        delay: Delay between attempts in seconds
        
    Returns:
        bool: True if server connected successfully, False otherwise
    """
    print(f"\nWaiting for Julia ZMQ server to be ready (max {max_attempts} attempts)...")
    
    # First check if zmq_interface has a valid socket
    if not hasattr(zmq_interface, 'socket') or zmq_interface.socket is None:
        print("⚠️ ZMQ socket is not initialized - attempting to set up connection first")
        if not zmq_interface._setup_zmq_connection():
            print("❌ Failed to set up ZMQ connection")
            return False
    
    # Ping messages specifically for Julia ZMQ server - try different formats
    ping_messages = [
        '{"ping": true}',
        '{"type": "ping"}',
        '{"command": "ping"}',
        # Try a minimal observation that Julia might recognize
        '{"observation": {"position": [0.0, 0.0, 0.0]}}',
        # Try a more complete observation that matches what Julia expects
        '{"observation": {"position": [0.0, 0.0, 0.0], "target": [10.0, 10.0, 10.0], "obstacles": []}}',
    ]
    
    current_ping_idx = 0
    socket_resets = 0
    
    for attempt in range(1, max_attempts + 1):
        # Reset socket periodically
        if attempt > 1 and attempt % 3 == 0:
            socket_resets += 1
            print(f"Resetting socket (reset #{socket_resets})...")
            zmq_interface._reset_socket()
            time.sleep(delay)
        
        # Try different ping messages in sequence
        ping_message = ping_messages[current_ping_idx % len(ping_messages)]
        current_ping_idx += 1
        
        print(f"Attempt {attempt}/{max_attempts}: Sending {ping_message[:30]}...")
        
        # Clear any pending messages in the socket
        try:
            # Try to flush any pending messages
            poller = zmq.Poller()
            poller.register(zmq_interface.socket, zmq.POLLIN)
            
            # Check if there's anything to receive and clear it
            if poller.poll(100):  # Very short timeout
                try:
                    zmq_interface.socket.recv_string(flags=zmq.NOBLOCK)
                    print("Cleared pending message from socket")
                except zmq.ZMQError:
                    pass  # Ignore errors during cleanup
        except Exception:
            pass  # Ignore any errors during cleanup
        
        try:
            # Send the ping message
            zmq_interface.socket.send_string(ping_message, flags=zmq.NOBLOCK)
            
            # Wait for response with timeout
            poller = zmq.Poller()
            poller.register(zmq_interface.socket, zmq.POLLIN)
            
            if poller.poll(3000):  # 3 second timeout
                response = zmq_interface.socket.recv_string()
                print(f"✅ Server responded: {response[:50]}...")
                
                # If we got a response that looks like valid JSON, consider it successful
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict):
                        if "action" in parsed or "waypoint" in parsed or "pong" in parsed:
                            print("✅ Verified Julia ZMQ server is operational!")
                            return True
                        else:
                            print(f"Response format OK but missing expected keys. Found: {list(parsed.keys())}")
                except json.JSONDecodeError:
                    # If it's not JSON but we got something, still count as connected
                    print("Response is not valid JSON but connection established")
                
                # Any response is better than none
                return True
            else:
                print("⏳ No response (timeout)")
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                print("⏳ Socket would block (EAGAIN)")
            else:
                print(f"⚠️ ZMQ error: {str(e)}")
                # Reset socket if not EAGAIN
                if e.errno != zmq.EAGAIN:
                    zmq_interface._reset_socket()
        except Exception as e:
            print(f"⚠️ Connection error: {str(e)}")
            zmq_interface._reset_socket()
        
        # Try to detect if Julia process is still running
        if attempt % 5 == 0:
            if platform.system() == "Windows":
                try:
                    result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq julia.exe"], 
                                         capture_output=True, text=True)
                    if "julia.exe" in result.stdout:
                        print("✓ Julia process is still running")
                    else:
                        print("⚠️ Julia process not found! Attempting to restart...")
                        # Try to restart the Julia server
                        start_julia_server()
                        time.sleep(3)  # Give it time to start
                        # Reset ZMQ connection
                        zmq_interface._reset_socket()
                except Exception as e:
                    print(f"Error checking Julia process: {str(e)}")
        
        # Wait before next attempt
        time.sleep(delay)
    
    print("❌ Maximum connection attempts reached. Server not responding.")
    return False

def start_julia_server():
    """Start the Julia ZMQ server directly using subprocess with proper output capturing
    
    Returns:
        bool: True if server was started successfully, False otherwise
    """
    print("\n==== Starting Julia ZMQ Server Directly ====")
    
    # Determine the correct Julia script path
    script_path = "zmq_server.jl"
    if not os.path.exists(script_path):
        script_path = os.path.join("actinf", "zmq_server.jl")
        if not os.path.exists(script_path):
            print(f"⚠️ Server script not found at any location!")
            return False
    
    print(f"Found server script at: {script_path}")
    
    # Clean up any existing Julia processes first
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
    
    # Build the command
    cmd = ["julia", "--project=.", script_path]
    
    # Force execution in a visible window using different methods
    try:
        # Launch the Julia process
        if platform.system() == "Windows":
            # On Windows, use a direct CREATE_NEW_CONSOLE to ensure visibility
            print(f"Executing Julia with command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=os.path.abspath(".")  # Use absolute path to current directory
            )
            
            # Capture the PID for later reference
            print(f"Started Julia server with PID: {process.pid}")
            
            # Wait a bit for process to start
            time.sleep(5)
            
            # Check if process is still running
            if process.poll() is not None:
                print(f"⚠️ Julia process ended with return code: {process.returncode}")
                return False
            
            # Process is running
            print("✅ Julia server process is running")
            return True
            
        else:
            # On Unix systems, use nohup to keep process running
            print(f"Executing Julia with command: nohup {' '.join(cmd)} &")
            subprocess.Popen(
                ["nohup"] + cmd,
                stdout=open("julia_server.log", "w"),
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                cwd=os.path.abspath(".")
            )
            
            # Wait a bit for the process to start
            time.sleep(5)
            
            # Check if process is running
            ps_result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
            if "julia" in ps_result.stdout and script_path in ps_result.stdout:
                print("✅ Julia server process is running")
                return True
            else:
                print("⚠️ Julia server process not found after startup")
                return False
                
    except Exception as e:
        print(f"❌ Error starting Julia server: {str(e)}")
        traceback.print_exc()
        return False

# Main function
def main():
    try:
        # Create controller
        controller = DroneController()
        
        # Precompile Julia components if needed
        print("Precompiling Julia components...")
        if not controller.precompile_julia_components():
            print("Warning: Precompilation had issues but continuing anyway")
        
        # Step 1: Start the Julia server FIRST before any ZMQ connection attempts
        print("\n===== Step 1: Starting Julia ZMQ Server =====")
        server_started = start_julia_server()
        
        if not server_started:
            print("❌ Failed to start Julia ZMQ server! Please check Julia installation.")
            return
            
        print("✅ Julia ZMQ server started successfully")
        
        # Step 2: Reset AirSim and take off
        print("\n===== Step 2: Initializing AirSim =====")
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        
        # Hover in place to stabilize
        time.sleep(2)
        
        # Get current position after takeoff for tracking
        current_position = client.getMultirotorState().kinematics_estimated.position
        start_position = [current_position.x_val, current_position.y_val, current_position.z_val]
        print(f"Drone ready at starting position: {start_position}")
        
        # Wait for stability
        time.sleep(2)
        
        # Step 3: Initialize ZMQ interface and connect to the running server
        print("\n===== Step 3: Connecting to Julia ZMQ Server =====")
        zmq_interface = ZMQInterface()  # Create and connect immediately
        
        # Wait for the server to become fully ready for communication
        server_ready = wait_for_server(zmq_interface, max_attempts=15, delay=1.0)
        
        if not server_ready:
            print("\n⚠️ Failed to establish communication with Julia ZMQ server.")
            print("The server might be running but not responding correctly.")
            return
            
        print("✅ Successfully connected to Julia ZMQ server")
        
        # Step 4: Start navigation
        print("\n===== Step 4: Starting Navigation =====")
        success = controller.navigate_to_target(zmq_interface)
        
        # Plot path
        if success:
            path_history = [controller.current_position]
            x = [p[0] for p in path_history]
            y = [p[1] for p in path_history]
            z = [p[2] for p in path_history]
            
            plt.figure(figsize=(10, 8))
            plt.plot(x, y, 'b-', label='Drone Path')
            plt.scatter([start_position[0]], [start_position[1]], color='g', s=100, label='Start')
            plt.scatter([TARGET_LOCATION[0]], [TARGET_LOCATION[1]], color='r', s=100, label='Target')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Drone Navigation Path')
            plt.legend()
            plt.grid(True)
            plt.savefig('drone_path.png')
            plt.show()
        
    except KeyboardInterrupt:
        print("Navigation interrupted by user")
    finally:
        # Return control
        client.armDisarm(False)
        client.enableApiControl(False)
        print("Navigation complete, control released")

if __name__ == "__main__":
    main()

