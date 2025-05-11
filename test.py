import airsim
import numpy as np
import time
import os
import json
import subprocess
import sys
import zmq
import math
import platform
import traceback
from matplotlib import pyplot as plt
from os import path

# Custom NumPy-aware JSON encoder
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyJSONEncoder, self).default(obj)

# Constants and hyperparameters
TARGET_LOCATION = [-20.0, -20.0, -30.0]  # [x, y, z] in NED coordinates
MARGIN = 1.5  # Safety margin for waypoint generation (meters)
WAYPOINT_SAMPLE_COUNT = 75  # Number of waypoints to consider
POLICY_LENGTH = 3  # Number of steps in the policy
DENSITY_RADIUS = 5.0  # Radius for density evaluation
ARRIVAL_THRESHOLD = 1.5 # meters
MAX_ITERATIONS = 100
DEFAULT_ZMQ_PORT = 5555  # Default ZMQ server port
ZMQ_TIMEOUT = 10000  # ZMQ socket timeout in milliseconds (10 seconds)
ZMQ_MAX_RETRIES = 3  # Maximum number of retries for ZMQ communication

# Add the airsim directory to the Python path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'airsim'))

# Import the scanner
try:
    from Sensory_Input_Processing import EnvironmentScanner
except ImportError:
    from airsim.Sensory_Input_Processing import EnvironmentScanner

class JuliaServer:
    """Manages the Julia environment and ZMQ server for active inference"""
    
    def __init__(self):
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.server_process = None
        self.julia_path = self._find_julia_executable()
    
    def _find_julia_executable(self):
        """Find the Julia executable on the system"""
        print("Locating Julia executable...")
        
        # Default command assuming Julia is in PATH
        julia_path = "julia"
        
        if platform.system() == "Windows":
            # Windows-specific Julia paths
            possible_paths = [
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
            
            # Expand %USERNAME% environment variable
            username = os.environ.get('USERNAME', '')
            possible_paths = [p.replace('%USERNAME%', username) for p in possible_paths]
            
            # Check each path
            for path in possible_paths:
                if os.path.exists(path):
                    julia_path = path
                    print(f"Found Julia at: {julia_path}")
                    break
        
        # Test if Julia is actually available
        try:
            result = subprocess.run([julia_path, "--version"], 
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"Julia version: {result.stdout.strip()}")
            else:
                print(f"Julia executable found but returned error: {result.stderr}")
        except Exception as e:
            print(f"Warning: Error testing Julia executable: {e}")
        
        return julia_path
    
    def precompile_packages(self):
        """Precompile Julia packages to speed up server startup"""
        print("\n==== Precompiling Julia Packages ====")
        
        # Check if precompilation has already been done
        precomp_success_flag = os.path.join(self.cwd, ".precompilation_success")
        if os.path.exists(precomp_success_flag):
            print(f"✅ Precompilation success flag found. Skipping precompilation.")
            return True
        
        # Run precompile script
        precompile_script = os.path.join(self.cwd, "precompile.jl")
        if not os.path.exists(precompile_script):
            print(f"❌ Precompilation script not found at {precompile_script}")
            return False
        
        print("Running Julia precompilation (this may take a few minutes)...")
        try:
            # Use a more reliable process management approach
            if platform.system() == "Windows":
                # Windows-specific precompilation
                precompile_log = os.path.join(self.cwd, "julia_precompile.log")
                
                # Create a visible window process
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 1  # SW_SHOWNORMAL
                
                with open(precompile_log, "wb") as log_file:  # Use binary mode to avoid encoding issues
                    process = subprocess.Popen(
                        [self.julia_path, precompile_script],
                        cwd=self.cwd,
                        stdout=log_file,
                        stderr=log_file,
                        startupinfo=startupinfo
                    )
                
                # Wait with a reasonable timeout
                try:
                    print("Waiting for precompilation to complete (timeout: 10 min)...")
                    return_code = process.wait(timeout=600)
                    
                    if return_code == 0:
                        print("✅ Precompilation completed successfully")
                        return True
                    else:
                        print(f"⚠️ Precompilation returned code {return_code}")
                        
                        # Show the log tail with proper encoding handling
                        try:
                            with open(precompile_log, 'rb') as f:  # Use binary mode
                                log_data = f.read()
                                # Try different encodings to handle special characters
                                encodings = ['utf-8', 'latin-1', 'cp1252']
                                log_text = None
                                
                                for encoding in encodings:
                                    try:
                                        log_text = log_data.decode(encoding, errors='replace')
                                        break
                                    except Exception:
                                        continue
                                
                                if log_text:
                                    log_lines = log_text.splitlines()[-10:]
                                    print("Last 10 lines of precompilation log:")
                                    for line in log_lines:
                                        print(f"  {line}")
                        except Exception as e:
                            print(f"Could not read precompilation log: {e}")
                        
                        # Check status file which is JSON and should be more reliable
                        status_file = os.path.join(self.cwd, ".precompilation_status.json")
                        if os.path.exists(status_file):
                            try:
                                with open(status_file, 'r') as f:
                                    status_data = json.load(f)
                                    print(f"Precompilation status: {status_data.get('status')} - {status_data.get('message')}")
                            except Exception as e:
                                print(f"Could not read status file: {e}")
                                
                        # If success flag exists, consider it successful despite return code
                        if os.path.exists(precomp_success_flag):
                            print("✅ Success flag found despite return code. Continuing.")
                            return True
                            
                        return False
                except subprocess.TimeoutExpired:
                    print("⚠️ Precompilation timed out after 10 minutes")
                    process.terminate()
                    return False
            else:
                # Unix approach
                result = subprocess.run(
                    [self.julia_path, precompile_script],
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    encoding='latin-1',  # Use latin-1 encoding which can handle all byte values
                    timeout=600
                )
                
                if result.returncode == 0:
                    print("✅ Precompilation completed successfully")
                    return True
                else:
                    print(f"⚠️ Precompilation failed with code {result.returncode}")
                    print(f"Output: {result.stderr[:500]}")
                    
                    # Check for success flag anyway
                    if os.path.exists(precomp_success_flag):
                        print("✅ Success flag found despite return code. Continuing.")
                        return True
                        
                    return False
                
        except Exception as e:
            print(f"❌ Error during precompilation: {e}")
            traceback.print_exc()
            
            # Check for success flag anyway
            if os.path.exists(precomp_success_flag):
                print("✅ Success flag found despite exception. Continuing.")
                return True
                
            return False
    
    def start_server(self):
        """Start the ZMQ server"""
        print("\n==== Starting ZMQ Server ====")
        
        # Check if server is already running
        server_running_flag = os.path.join(self.cwd, ".zmq_server_running")
        server_status_file = os.path.join(self.cwd, ".zmq_server_status.json")
        
        # First attempt to ping an already running server to verify it's truly responding
        if os.path.exists(server_running_flag) and os.path.exists(server_status_file):
            try:
                with open(server_status_file, 'r') as f:
                    status_data = json.load(f)
                    if status_data.get('status') == 'running':
                        port = status_data.get('port', DEFAULT_ZMQ_PORT)
                        print(f"Found server status file indicating a running server on port {port}")
                        
                        # Try to connect and ping the server to verify it's actually responding
                        print("Verifying server is responsive...")
                        try:
                            # Quick connection test
                            context = zmq.Context()
                            socket = context.socket(zmq.REQ)
                            socket.setsockopt(zmq.LINGER, 500)
                            socket.setsockopt(zmq.RCVTIMEO, 2000)  # Short timeout for ping
                            socket.setsockopt(zmq.SNDTIMEO, 2000)
                            socket.connect(f"tcp://localhost:{port}")
                            
                            # Send ping and wait for response
                            socket.send_string("ping")
                            response = socket.recv_string()
                            
                            if response == "pong":
                                print(f"✅ Existing ZMQ server is responsive on port {port}")
                                socket.close()
                                context.term()
                                return port
                            else:
                                print(f"⚠️ Unexpected response from server: {response}")
                        except Exception as e:
                            print(f"⚠️ Existing server not responsive: {e}")
                        finally:
                            try:
                                socket.close()
                                context.term()
                            except:
                                pass
                        
                        # If we get here, the server wasn't responsive
                        print("Detected a non-responsive server. Will restart it.")
                        # Clean up stale server flags
                        if os.path.exists(server_running_flag):
                            os.remove(server_running_flag)
                        # No return here - fall through to start a new server
            except Exception as e:
                print(f"Error reading server status: {e}")
                # Clean up potentially corrupted files
                for file in [server_running_flag, server_status_file]:
                    if os.path.exists(file):
                        try:
                            os.remove(file)
                        except:
                            pass
        
        # Find the server script
        server_script = os.path.join(self.cwd, "actinf", "zmq_server.jl")
        if not os.path.exists(server_script):
            print(f"❌ Server script not found at {server_script}")
            return None
        
        print(f"Starting ZMQ server from {server_script}...")
        
        try:
            # Make sure any existing server process is terminated before starting a new one
            if hasattr(self, 'server_process') and self.server_process:
                try:
                    if self.server_process.poll() is None:  # Still running
                        print("Terminating previous server process...")
                        self.server_process.terminate()
                        try:
                            self.server_process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            self.server_process.kill()
                            print("Force-killed previous server process")
                except:
                    pass
            
            # Start the server as a background process with a visible console for easier debugging
            if platform.system() == "Windows":
                # Windows needs different flags to start detached with visible window
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 1  # SW_SHOWNORMAL
                
                self.server_process = subprocess.Popen(
                    [self.julia_path, "--project=.", server_script],
                    cwd=self.cwd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    startupinfo=startupinfo
                )
            else:
                # Unix-like systems
                self.server_process = subprocess.Popen(
                    [self.julia_path, "--project=.", server_script],
                    cwd=self.cwd,
                    start_new_session=True
                )
            
            # Wait for server to start and check status file for port
            print("Waiting for server to initialize...")
            
            # Try several times to get the server port from status file
            max_attempts = 15  # Increased timeout for server startup
            server_port = None
            
            for attempt in range(max_attempts):
                time.sleep(1)  # Wait a bit between checks
                
                if os.path.exists(server_status_file):
                    try:
                        with open(server_status_file, 'r') as f:
                            status_data = json.load(f)
                            if status_data.get('status') == 'running':
                                server_port = status_data.get('port', DEFAULT_ZMQ_PORT)
                                print(f"✅ ZMQ server started successfully on port {server_port}")
                                
                                # Verify server is actually responsive with a ping
                                try:
                                    context = zmq.Context()
                                    socket = context.socket(zmq.REQ)
                                    socket.setsockopt(zmq.LINGER, 500)
                                    socket.setsockopt(zmq.RCVTIMEO, 2000)
                                    socket.setsockopt(zmq.SNDTIMEO, 2000)
                                    socket.connect(f"tcp://localhost:{server_port}")
                                    
                                    socket.send_string("ping")
                                    response = socket.recv_string()
                                    
                                    if response == "pong":
                                        print(f"✅ ZMQ server verified responsive on port {server_port}")
                                        socket.close()
                                        context.term()
                                        break
                                    else:
                                        print(f"⚠️ Unexpected response from server: {response}")
                                except Exception as e:
                                    print(f"⚠️ Server not immediately responsive, will retry: {e}")
                                finally:
                                    try:
                                        socket.close()
                                        context.term()
                                    except:
                                        pass
                    except Exception as e:
                        print(f"Error reading status file (attempt {attempt+1}): {e}")
                
                # Check if process is still running
                if self.server_process.poll() is not None:
                    print(f"❌ Server process exited with code {self.server_process.returncode}")
                    return None
                
                print(f"Waiting for server to start... (attempt {attempt+1}/{max_attempts})")
            
            # Return the port (or default if not found)
            return server_port or DEFAULT_ZMQ_PORT
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            traceback.print_exc()
            return None
    
    def shutdown(self):
        """Shutdown the server if running"""
        if self.server_process and self.server_process.poll() is None:
            print("Shutting down ZMQ server...")
            try:
                # Try to terminate gracefully first
                self.server_process.terminate()
                
                # Wait a bit to let it close
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.server_process.kill()
                    print("Server process had to be forcefully terminated")
            except Exception as e:
                print(f"Error shutting down server: {e}")

class ZMQInterface:
    """Interface for communicating with the Julia Active Inference server via ZMQ"""
    
    def __init__(self, server_port=DEFAULT_ZMQ_PORT):
        """Initialize ZMQ connection to Julia server"""
        self.server_address = f"tcp://localhost:{server_port}"
        print(f"Initializing ZMQ interface: {self.server_address}")
        
        # Initialize socket
        self.context = None
        self.socket = None
        
        # Connect to the server
        self._setup_connection()
    
    def _setup_connection(self):
        """Set up the ZMQ connection with proper error handling"""
        try:
            # Close existing connections first
            self._close_connection()
            
            # Create new context and socket
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            
            # Configure socket options
            self.socket.setsockopt(zmq.LINGER, 1000)  # Wait up to 1 second when closing
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT)  # Receive timeout
            self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT)  # Send timeout
            self.socket.setsockopt(zmq.REQ_RELAXED, 1)  # More relaxed REQ socket behavior
            self.socket.setsockopt(zmq.REQ_CORRELATE, 1)  # Correlate replies with requests
            
            # Connect to server
            print(f"Connecting to server at {self.server_address}...")
            self.socket.connect(self.server_address)
            
            # Test connection with ping
            print("Testing connection with ping...")
            try:
                self.socket.send_string("ping")
                poller = zmq.Poller()
                poller.register(self.socket, zmq.POLLIN)
                
                if poller.poll(5000):  # 5 second timeout for first ping
                    response = self.socket.recv_string()
                    if response == "pong":
                        print("✅ ZMQ connection established successfully")
                        return True
                    else:
                        print(f"⚠️ Unexpected ping response: {response}")
                else:
                    print("⚠️ No response to ping")
            except Exception as e:
                print(f"⚠️ Error during ping test: {e}")
            
            return False
        
        except Exception as e:
            print(f"❌ Failed to setup ZMQ connection: {e}")
            self._close_connection()
            return False
    
    def _close_connection(self):
        """Close the ZMQ connection cleanly"""
        if hasattr(self, 'socket') and self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        if hasattr(self, 'context') and self.context:
            try:
                self.context.term()
            except:
                pass
            self.context = None
    
    def send_observation_and_receive_action(self, observation):
        """Send observation data to Julia and receive next waypoint
        
        Args:
            observation: Dict containing drone observation data
                
        Returns:
            tuple: (next_waypoint, policy) or (None, None) if communication failed
        """
        try:
            print("\nSending observation to Active Inference engine...")
            
            # Convert observation to JSON string
            obs_json = json.dumps(observation, cls=NumpyJSONEncoder)
            
            # Send with retry mechanism
            for retry in range(ZMQ_MAX_RETRIES):
                try:
                    # Send observation data
                    self.socket.send_string(obs_json)
                    print(f"Sent observation data (attempt {retry+1}/{ZMQ_MAX_RETRIES})")
                    
                    # Wait for response
                    poller = zmq.Poller()
                    poller.register(self.socket, zmq.POLLIN)
                    
                    if poller.poll(ZMQ_TIMEOUT):
                        # Receive and parse response
                        response = self.socket.recv_string()
                        print(f"Received response ({len(response)} bytes)")
                        
                        result = json.loads(response)
                        waypoint = result.get("next_waypoint")
                        policy = result.get("policy", [])
                        
                        if waypoint:
                            print(f"Next waypoint: {[round(p, 2) for p in waypoint]}")
                            return waypoint, policy
                        else:
                            print("Error: No waypoint in response")
                            if "error" in result:
                                print(f"Server error: {result['message']}")
                            return None, None
                    else:
                        print(f"Response timeout (attempt {retry+1}/{ZMQ_MAX_RETRIES})")
                        
                        # Reset connection on timeout (except on last attempt)
                        if retry < ZMQ_MAX_RETRIES - 1:
                            print("Resetting connection...")
                            self._setup_connection()
                            time.sleep(1)
                except zmq.ZMQError as e:
                    print(f"ZMQ error (attempt {retry+1}/{ZMQ_MAX_RETRIES}): {e}")
                    if retry < ZMQ_MAX_RETRIES - 1:
                        print("Resetting connection...")
                        self._setup_connection()
                        time.sleep(1)
            
            print("❌ Failed to get valid response after all retries")
            return None, None
                
        except Exception as e:
            print(f"❌ Unhandled error in active inference processing: {e}")
            traceback.print_exc()
            return None, None
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self._close_connection()

class Scanner:
    """Scanner for obstacle detection using AirSim's LiDAR data"""
    
    def __init__(self, client, scan_range=20.0):
        self.client = client
        self.scan_range = scan_range
    
    def fetch_density_distances(self):
        """Get obstacle positions and distances with orientation-aware transformation"""
        try:
            # Initialize empty lists for return values
            obstacle_positions = []
            obstacle_distances = []

            # Get drone state
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
                return [], []
            
            # Convert point cloud to positions
            try:
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error reshaping point cloud: {e}")
                return [], []

            # Quaternion utility functions
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
                # Convert point to quaternion form
                point_quat = [0.0, point[0], point[1], point[2]]
                q_conj = quaternion_conjugate(quaternion)
                rotated = quaternion_multiply(
                    quaternion_multiply(quaternion, point_quat),
                    q_conj
                )
                return rotated[1:4]

            # Process each point
            for point in points:
                # Skip invalid points
                if np.isnan(point).any() or np.isinf(point).any():
                    continue
                
                try:
                    # Transform point to global reference frame
                    global_point = rotate_point_by_quaternion(point, drone_orientation)
                    
                    # Add drone position
                    global_point = [
                        global_point[0] + drone_pos[0],
                        global_point[1] + drone_pos[1],
                        global_point[2] + drone_pos[2]
                    ]
                    
                    # Calculate distance
                    dist = np.sqrt(np.sum((np.array(drone_pos) - np.array(global_point)) ** 2))
                    
                    # Skip invalid distances
                    if np.isnan(dist) or np.isinf(dist):
                        continue
                    
                    # Only include obstacles within scan range
                    if dist <= self.scan_range and dist > 0.0:
                        obstacle_positions.append(global_point)
                        obstacle_distances.append(float(dist))
                except Exception:
                    continue
            
            return obstacle_positions, obstacle_distances
        
        except Exception as e:
            print(f"Error in fetch_density_distances: {e}")
            traceback.print_exc()
            return [], []

def move_to_waypoint(client, current_pos, waypoint, velocity=2):
    """Move to waypoint with calculated yaw so drone faces direction of travel
    
    Args:
        client: AirSim client instance
        current_pos: Current drone position [x, y, z]
        waypoint: Target waypoint [x, y, z]
        velocity: Movement velocity in m/s
    """
    try:
        # Calculate movement vector and distance
        movement_vector = np.array(waypoint) - np.array(current_pos)
        distance = np.linalg.norm(movement_vector)
        
        # Safety check for movement distance
        if distance < 0.1:
            print("Movement distance too small, using current orientation")
            client.moveToPositionAsync(
                waypoint[0], waypoint[1], waypoint[2],
                velocity
            ).join()
            return
        
        # Calculate yaw angle in radians
        # In NED coordinates, yaw is measured from North (x-axis) and increases clockwise
        # atan2(y, x) will give us the angle in the standard math frame, but we need to ensure
        # it's properly mapped to the NED coordinate system
        yaw = math.atan2(movement_vector[1], movement_vector[0])
        
        # Convert to degrees for display and for AirSim's YawMode
        yaw_degrees = math.degrees(yaw)
        
        # AirSim expects yaw in degrees, ranging from -180 to 180 or 0 to 360
        # We already have this range from atan2, but ensure it's clamped correctly
        if yaw_degrees < -180:
            yaw_degrees += 360
        elif yaw_degrees > 180:
            yaw_degrees -= 360
            
        print(f"Moving with yaw: {yaw_degrees:.1f}° at velocity: {velocity} m/s")
        print(f"Movement vector: [{movement_vector[0]:.2f}, {movement_vector[1]:.2f}, {movement_vector[2]:.2f}]")
        
        # Move drone with yaw control
        client.moveToPositionAsync(
            waypoint[0], waypoint[1], waypoint[2],
            velocity,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_degrees)
        ).join()
    except Exception as e:
        print(f"Error in move_to_waypoint: {e}")
        traceback.print_exc()
        # Fallback to simple movement without yaw control
        try:
            client.moveToPositionAsync(
                waypoint[0], waypoint[1], waypoint[2],
                velocity
            ).join()
        except Exception as e2:
            print(f"Fallback movement also failed: {e2}")
            # Last resort - hover in place
            client.hoverAsync().join()

def main():
    """Main navigation process"""
    print("\n==== Autonomous Drone Navigation with Active Inference ====")
    print(f"Target location: {TARGET_LOCATION}")
    
    # Initialize and precompile Julia environment
    julia_server = JuliaServer()
    precompile_success = julia_server.precompile_packages()
    
    if not precompile_success:
        print("⚠️ Precompilation had issues but will continue")
    
    # Start ZMQ server
    server_port = julia_server.start_server()
    if not server_port:
        print("❌ Failed to start ZMQ server. Cannot continue.")
        return
    
    # Initialize ZMQ interface
    zmq_interface = ZMQInterface(server_port=server_port)
    
    # Connect to AirSim
    print("\n==== Connecting to AirSim ====")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Initialize scanner
    scanner = Scanner(client)
    
    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    
    # Get initial position
    drone_state = client.getMultirotorState().kinematics_estimated
    drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
    print(f"Position after takeoff: {[round(p, 2) for p in drone_pos]}")
    
    # Initialize navigation variables
    iteration = 0
    distance_to_target = np.linalg.norm(np.array(TARGET_LOCATION) - np.array(drone_pos))
    trajectory = [drone_pos]
    
    print(f"Initial distance to target: {distance_to_target:.2f} meters")
    print(f"Starting navigation with Active Inference...")
    
    # Main navigation loop
    while distance_to_target > ARRIVAL_THRESHOLD and iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n=== Iteration {iteration}/{MAX_ITERATIONS} ===")
        
        # Get current drone position
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
        
        # Manual check for target proximity - immediately exit loop if within 1 meter
        if distance_to_target <= 1.5:
            print("\n🎯 Target reached! Distance is within 1 meter threshold.")
            print("🎉 Successful navigation - terminating early.")
            break
        
        # Get obstacle data
        obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
        print(f"Detected {len(obstacle_positions)} obstacles")
        
        # Create observation for active inference
        observation = {
            "drone_position": drone_pos,
            "target_position": TARGET_LOCATION,
            "obstacle_positions": obstacle_positions,
            "obstacle_distances": obstacle_distances,
            "waypoint_count": WAYPOINT_SAMPLE_COUNT,
            "safety_margin": MARGIN,
            "policy_length": POLICY_LENGTH,
            "density_radius": DENSITY_RADIUS
        }
        
        # Get next waypoint
        print("Computing next waypoint with Active Inference...")
        next_waypoint, policy = zmq_interface.send_observation_and_receive_action(observation)
        
        # Check if waypoint is valid
        if next_waypoint is None:
            print("❌ Failed to get valid waypoint. Terminating navigation.")
            break
        
        print(f"Next waypoint: {[round(p, 2) for p in next_waypoint]}")
        
        # Move drone to waypoint
        move_to_waypoint(client, drone_pos, next_waypoint)
        
        # Record trajectory
        trajectory.append(next_waypoint)
        
        # Check if target reached
        if distance_to_target <= ARRIVAL_THRESHOLD:
            print("\n🎉 Target reached successfully!")
            break
        
        # Brief pause
        time.sleep(0.5)
    
    # Final status
    if distance_to_target <= ARRIVAL_THRESHOLD:
        print(f"\n✅ Navigation completed successfully in {iteration} iterations!")
        print(f"Final position: {[round(p, 2) for p in drone_pos]}")
        print(f"Distance to target: {distance_to_target:.2f} meters")
    elif iteration >= MAX_ITERATIONS:
        print(f"\n⚠️ Maximum iterations reached without reaching target")
        print(f"Final position: {[round(p, 2) for p in drone_pos]}")
        print(f"Remaining distance: {distance_to_target:.2f} meters")
    else:
        print("\n❌ Navigation terminated due to errors")
        print(f"Final position: {[round(p, 2) for p in drone_pos]}")
    
    # Visualize trajectory
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
        
        # Plot obstacles if any
        if len(obstacle_positions) > 0:
            obstacle_array = np.array(obstacle_positions)
            ax.scatter(obstacle_array[:, 0], obstacle_array[:, 1], obstacle_array[:, 2], 
                       c='orange', marker='^', s=50, alpha=0.5, label='Obstacles')
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('Drone Navigation Trajectory with Active Inference')
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        
        # Save figure
        plt.savefig('drone_path.png')
        print("Trajectory visualization saved to 'drone_path.png'")
        plt.show()
    except Exception as e:
        print(f"Error generating visualization: {e}")
    
    # Land the drone
    print("\nLanding the drone...")
    client.landAsync().join()
    client.armDisarm(False)
    
    # Shutdown server
    julia_server.shutdown()
    print("\nNavigation sequence complete")

if __name__ == "__main__":
    main()








