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
JULIA_PATH = "julia"  # Base Julia executable path
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
            
        # Check if connection was established and start server if needed
        if not defer_connection and (self.socket is None or not self._is_socket_connected()):
            print("Initial connection failed, attempting to start server...")
            self._start_server()
            # Try to connect again after starting the server
            if self._setup_zmq_connection():
                print("✅ Connection established after starting server")
            else:
                print("❌ Failed to establish connection even after starting server")
    
    def _is_socket_connected(self):
        """Check if the socket is properly connected"""
        if self.socket is None:
            return False
            
        try:
            # Try a quick non-blocking send to test connection
            self.socket.send_string("ping", zmq.NOBLOCK)
            return True
        except zmq.ZMQError:
            return False
        except Exception:
            return False
    
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
    
    def _start_server(self):
        """Start the Julia ZMQ server as a separate process"""
        print("Starting Julia ZMQ server...")
        
        # Get current directory
        cwd = os.path.dirname(os.path.abspath(__file__))
        
        # Find Julia executable
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
        
        # Look for zmq_server.jl in the actinf directory
        server_script = os.path.join(cwd, "actinf", "zmq_server.jl")
        if not os.path.exists(server_script):
            print(f"❌ Server script not found at {server_script}")
            # Try alternative locations
            alt_script_paths = [
                os.path.join(cwd, "zmq_server.jl"),
                os.path.join(cwd, "actinf", "src", "zmq_server.jl")
            ]
            for alt_path in alt_script_paths:
                if os.path.exists(alt_path):
                    server_script = alt_path
                    print(f"✅ Found server script at alternative location: {server_script}")
                    break
            else:
                print("❌ Could not find zmq_server.jl in any expected location")
                return False
        else:
            print(f"✅ Found server script at: {server_script}")
        
        try:
            # Start Julia ZMQ server with project activation
            cmd = [julia_path, "--project=.", server_script]
            
            print(f"Starting server with command: {' '.join(cmd)}")
            
            # Start the server as a background process
            if platform.system() == "Windows":
                # Windows needs different flags to start detached
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 1  # SW_SHOWNORMAL
                
                server_process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    startupinfo=startupinfo
                )
            else:
                # Unix-like systems
                server_process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    start_new_session=True
                )
            
            # Wait a bit for server to start
            print("Waiting for server to start...")
            time.sleep(5)
            
            # Check if process is still running (not crashed immediately)
            if server_process.poll() is None:
                print("✅ Server process started successfully")
                return True
            else:
                print(f"❌ Server process exited with code {server_process.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            traceback.print_exc()
            return False
    
    def _is_server_running(self):
        """Check if the Julia ZMQ server is actually running and listening"""
        try:
            # Try to create a temporary socket and connect to see if server is running
            temp_context = zmq.Context()
            temp_socket = temp_context.socket(zmq.REQ)
            temp_socket.setsockopt(zmq.LINGER, 500)  # Don't wait long when closing
            temp_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout for recv
            temp_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1s timeout for send
            
            # Try to connect
            temp_socket.connect(self.server_address)
            
            # Send a ping message
            try:
                temp_socket.send_string("ping", zmq.NOBLOCK)
                
                # Use poller with timeout
                poller = zmq.Poller()
                poller.register(temp_socket, zmq.POLLIN)
                
                # Wait for response with timeout
                if poller.poll(1000):  # 1 second timeout
                    response = temp_socket.recv_string()
                    if response == "pong":
                        print("✅ ZMQ server is running and responding")
                        temp_socket.close()
                        temp_context.term()
                        return True
                else:
                    print("⚠️ ZMQ server did not respond to ping")
            except zmq.ZMQError as e:
                print(f"⚠️ ZMQ error during server check: {e}")
            finally:
                temp_socket.close()
                temp_context.term()
                
            return False
        except Exception as e:
            print(f"Error checking server status: {e}")
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

    def send_observation_and_receive_action(self, observation):
        """Send observation data to Julia and receive next waypoint.
        
        Args:
            observation: Dict containing drone observation data
                Required keys: drone_position, target_position
                
        Returns:
            tuple: (next_waypoint, policy) or (None, None) if communication failed
        """
        try:
            print("\nSending observation to Active Inference engine...")
            
            # Make sure interface directory exists
            interface_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interface")
            os.makedirs(interface_dir, exist_ok=True)
            
            # Path to the input/output files
            obs_input_path = os.path.join(interface_dir, "obs_input.json")
            
            # Write observation to JSON file
            try:
                with open(obs_input_path, 'w') as f:
                    json.dump(observation, f, cls=NumpyJSONEncoder)
                print("Successfully wrote observation to file")
            except Exception as e:
                print(f"Error writing observation to file: {e}")
                return None, None
            
            # Use ZMQ approach
            if hasattr(self, 'socket') and self.socket:
                print("Using ZMQ for inference communication...")
                
                # Convert observation to JSON string
                obs_json = json.dumps(observation, cls=NumpyJSONEncoder)
                
                # Send request with retry mechanism
                max_retries = ZMQ_MAX_RETRIES
                request_successful = False
                
                for retry in range(max_retries):
                    try:
                        # Send observation data
                        self.socket.send_string(obs_json)
                        print(f"Sent observation data via ZMQ (retry {retry+1}/{max_retries})")
                        
                        # Set up poller for timeout
                        poller = zmq.Poller()
                        poller.register(self.socket, zmq.POLLIN)
                        
                        # Wait for response with timeout
                        if poller.poll(ZMQ_TIMEOUT):
                            # Receive response
                            response = self.socket.recv_string()
                            request_successful = True
                            print("Received response from ZMQ server")
                            break
                        else:
                            print(f"ZMQ response timeout (attempt {retry+1}/{max_retries})")
                            
                            # Reset socket on timeout
                            if retry < max_retries - 1:  # Don't reset on last attempt
                                print("Resetting socket for next attempt...")
                                self._reset_socket()
                                time.sleep(1)  # Brief pause before retry
                    except zmq.ZMQError as e:
                        print(f"ZMQ error during communication (attempt {retry+1}/{max_retries}): {e}")
                        if retry < max_retries - 1:
                            print("Resetting socket for next attempt...")
                            self._reset_socket()
                            time.sleep(1)
                
                # Process the response from ZMQ if successful
                if request_successful:
                    try:
                        result = json.loads(response)
                        waypoint = result.get("next_waypoint")
                        policy = result.get("policy", [])
                        
                        if waypoint:
                            print(f"Next waypoint from ZMQ: {[round(p, 2) for p in waypoint]}")
                            return waypoint, policy
                        else:
                            print("Error: No waypoint in ZMQ response")
                            return None, None
                    except json.JSONDecodeError:
                        print(f"Error decoding ZMQ JSON response")
                        return None, None
                    except Exception as e:
                        print(f"Error processing ZMQ response: {str(e)}")
                        return None, None
                
                # ZMQ communication failed
                print("❌ ZMQ communication failed and no fallback available")
                print("Terminating navigation.")
                return None, None
            else:
                # No ZMQ socket available
                print("❌ ZMQ socket not available for communication")
                print("Terminating navigation.")
                return None, None
                
        except Exception as e:
            print(f"❌ Unhandled error in active inference processing: {e}")
            traceback.print_exc()
            print("Terminating navigation.")
            return None, None

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
    
    # Check if precompilation has already been done successfully
    precomp_success_flag = os.path.join(cwd, ".precompilation_success")
    precomp_status_file = os.path.join(cwd, ".precompilation_status.json")
    
    if os.path.exists(precomp_success_flag):
        print(f"✅ Precompilation success flag found: {precomp_success_flag}")
        precompile_success = True
    elif os.path.exists(precomp_status_file):
        try:
            with open(precomp_status_file, 'r') as f:
                status_data = json.load(f)
                status = status_data.get('status')
                message = status_data.get('message', 'No message')
                
                if status == 'success' or status == 'complete':
                    print(f"✅ Precompilation status file indicates success: {message}")
                    precompile_success = True
                else:
                    print(f"⚠️ Precompilation status file indicates issues: {message}")
                    print("Will attempt precompilation again")
        except Exception as e:
            print(f"⚠️ Could not read precompilation status file: {e}")
    
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
        
        # Install required packages BEFORE precompilation, but only ONCE
        # Create a flag file to track whether installation has already been done
        install_flag_file = os.path.join(cwd, ".packages_installed")
        if not os.path.exists(install_flag_file):
            print("\n==== Installing Required Julia Packages (First Run Only) ====")
            
            # Create a temporary Julia script to install packages
            install_script_path = os.path.join(cwd, "install_packages.jl")
            with open(install_script_path, "w") as f:
                f.write("""
                # Install required packages for active inference
                import Pkg
                
                # Activate the project
                Pkg.activate(".")
                
                # Add required packages
                required_packages = [
                    "JSON",
                    "LinearAlgebra",
                    "StaticArrays",
                    "ZMQ"
                ]
                
                for pkg in required_packages
                    if pkg != "LinearAlgebra"  # LinearAlgebra is a standard library
                        println("Adding package: $pkg")
                        try
                            Pkg.add(pkg)
                        catch e
                            println("Error adding $pkg: $e")
                        end
                    end
                end
                
                # Develop the actinf package
                println("Developing actinf package...")
                try
                    Pkg.develop(path="actinf")
                catch e
                    println("Error developing actinf: $e")
                end
                
                # Instantiate and precompile
                println("Instantiating project...")
                Pkg.instantiate()
                
                println("Resolving project...")
                Pkg.resolve()
                
                println("Building project...")
                Pkg.build()
                
                println("Installation complete!")
                """)
            
            # Run the installation script
            print(f"Running package installation script...")
            try:
                result = subprocess.run(
                    [julia_path, install_script_path], 
                    capture_output=True, 
                    text=True, 
                    timeout=600  # 10 minutes timeout
                )
                if result.returncode == 0:
                    print("✅ Package installation successful")
                    print(f"Output: {result.stdout[-200:]}")  # Show the last 200 chars
                    
                    # Create flag file to indicate successful installation
                    with open(install_flag_file, 'w') as f:
                        f.write(f"Packages installed on {time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"⚠️ Package installation returned code {result.returncode}")
                    print(f"Error: {result.stderr[:500]}")
            except Exception as e:
                print(f"⚠️ Error during package installation: {str(e)}")
            finally:
                # Clean up installation script
                try:
                    os.remove(install_script_path)
                except:
                    pass
        else:
            print(f"✅ Julia packages already installed (flag file exists: {install_flag_file})")
        
        # Run precompilation if not already done or we've had issues
        if not precompile_success:
            # Run precompilation script to prepare Julia environment with timeout
            print("Starting Julia precompilation (this may take several minutes)...")
            print("Precompiling Active Inference package dependencies...")
            
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
                                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                                    startupinfo=startupinfo
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
                                    
                            # Check for the success flag file
                            if os.path.exists(precomp_success_flag):
                                print("✅ Precompilation success flag created")
                                precompile_success = True
                            elif os.path.exists(precomp_status_file):
                                try:
                                    with open(precomp_status_file, 'r') as f:
                                        status_data = json.load(f)
                                        status = status_data.get('status')
                                        message = status_data.get('message', 'No message')
                                        
                                        print(f"Precompilation status: {status} - {message}")
                                        if status == 'success' or status == 'complete':
                                            precompile_success = True
                                except Exception as e:
                                    print(f"Could not read precompilation status file: {e}")
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
                            
                        # Check for success indicators
                        if os.path.exists(precomp_success_flag):
                            print("✅ Precompilation success flag created")
                            precompile_success = True
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
    
    # Create ZMQ interface with immediate connection
    print("\n==== Starting ZMQ Interface ====")
    zmq_interface = ZMQInterface(defer_connection=False)
    
    # Test if ZMQ interface is connected
    if hasattr(zmq_interface, 'socket') and zmq_interface.socket:
        print("✅ ZMQ interface connected successfully")
        return zmq_interface
        
    # If not connected, check if server is already running
    server_running_flag = os.path.join(cwd, ".zmq_server_running")
    server_status_file = os.path.join(cwd, ".zmq_server_status.json")
    
    if os.path.exists(server_running_flag):
        print(f"🔍 Found ZMQ server running flag file: {server_running_flag}")
        # Server is running but we couldn't connect - try to get port from status file
        if os.path.exists(server_status_file):
            try:
                with open(server_status_file, 'r') as f:
                    status_data = json.load(f)
                    server_port = status_data.get('port', DEFAULT_ZMQ_PORT)
                    server_status = status_data.get('status', 'unknown')
                    server_message = status_data.get('message', 'No message')
                    
                    print(f"ZMQ server status: {server_status} - {server_message}")
                    print(f"ZMQ server port: {server_port}")
                    
                    # If server is running but on a different port, try to connect to that port
                    if server_status == "running" and server_port != DEFAULT_ZMQ_PORT:
                        print(f"Attempting to connect to alternate port: {server_port}")
                        alt_address = f"tcp://localhost:{server_port}"
                        zmq_interface = ZMQInterface(server_address=alt_address, defer_connection=False)
                        
                        if hasattr(zmq_interface, 'socket') and zmq_interface.socket:
                            print(f"✅ Connected to ZMQ server on alternate port {server_port}")
                            return zmq_interface
            except Exception as e:
                print(f"Error reading server status file: {e}")
    
    # If not connected, try starting the server and connecting again
    print("⚠️ ZMQ interface not connected, attempting to start server...")
    
    # If we have a method to start the server, try it
    if hasattr(zmq_interface, '_start_server'):
        server_started = zmq_interface._start_server()
        
        if server_started:
            print("✅ ZMQ server started successfully")
            # Now connect to the server
            if zmq_interface._setup_zmq_connection():
                print("✅ ZMQ connection established after starting server")
                return zmq_interface
            else:
                print("❌ Failed to establish connection after starting server")
                
                # Check if server started on an alternate port
                if os.path.exists(server_status_file):
                    try:
                        time.sleep(2)  # Give server time to write status file
                        with open(server_status_file, 'r') as f:
                            status_data = json.load(f)
                            server_port = status_data.get('port', DEFAULT_ZMQ_PORT)
                            
                            if server_port != DEFAULT_ZMQ_PORT:
                                print(f"Server started on alternate port {server_port}, attempting to connect...")
                                alt_address = f"tcp://localhost:{server_port}"
                                zmq_interface = ZMQInterface(server_address=alt_address, defer_connection=False)
                                
                                if hasattr(zmq_interface, 'socket') and zmq_interface.socket:
                                    print(f"✅ Connected to ZMQ server on alternate port {server_port}")
                                    return zmq_interface
                    except Exception as e:
                        print(f"Error checking for alternate port: {e}")
        else:
            print("❌ Failed to start ZMQ server")
    
    print("⚠️ Could not establish proper ZMQ connection")
    return zmq_interface  # Return interface even if not connected

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
    
    # Check if ZMQ interface is properly connected
    if not hasattr(zmq_interface, 'socket') or zmq_interface.socket is None:
        print("❌ ZMQ connection is not available. Navigation requires ZMQ server connection.")
        print("Terminating navigation.")
        
        # Land the drone and return
        print("Landing the drone...")
        client.landAsync().join()
        client.armDisarm(False)
        return
    
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






