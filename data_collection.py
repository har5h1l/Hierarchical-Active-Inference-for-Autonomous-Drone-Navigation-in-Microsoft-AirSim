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
import logging
import pandas as pd
import random
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from matplotlib import pyplot as plt
from os import path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

# Default experiment configuration
DEFAULT_CONFIG = {
    "num_episodes": 10,
    "target_distance_range": (15.0, 50.0),  # min, max in meters
    "random_seed": 42,
    "max_steps_per_episode": 100,
    "output_dir": "experiment_results",
    "stuck_timeout": 15.0,  # seconds to consider the drone stuck
    "stuck_check_interval": 3.0,  # seconds between stuck checks
    "episode_timeout": 120,  # maximum episode duration in seconds
    "precompile_julia": True,  # whether to precompile Julia packages
    "zmq_port": DEFAULT_ZMQ_PORT,  # ZMQ server port
    
    # Raycasting debug configuration
    "debug_raycasting": False,  # Enable debug mode for raycasts
    "min_ray_checks": 3,  # Minimum number of rays for target validation
    "max_ray_checks": 7,  # Maximum number of rays for target validation 
    "visualize_raycasts": False,  # Visualize raycasts in AirSim
    "visualization_duration": 60.0  # Duration of visualization markers in seconds
}

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
            print(f"✅ Precompilation success flag found. Removing flag to force precompilation.")
            try:
                os.remove(precomp_success_flag)
            except Exception as e:
                print(f"Failed to remove precompilation flag: {e}")
        
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
    
    def start_server(self, port=DEFAULT_ZMQ_PORT):
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
        
        # Enforce minimum movement distance
        min_movement_distance = 0.2  # Reduced minimum movement to 0.2m for finer control in dense areas
        if distance < min_movement_distance:
            logging.warning(f"Waypoint too close ({distance:.2f}m), extending to minimum distance {min_movement_distance}m")
            # Extend the waypoint in the same direction to meet minimum distance
            if distance > 0.001:  # Avoid division by zero
                # Normalize the movement vector and scale to minimum distance
                normalized_vector = movement_vector / distance
                extended_vector = normalized_vector * min_movement_distance
                waypoint = (np.array(current_pos) + extended_vector).tolist()
                # Recalculate movement vector and distance
                movement_vector = np.array(waypoint) - np.array(current_pos)
                distance = np.linalg.norm(movement_vector)
            else:
                # If distance is effectively zero, create movement in a default direction
                logging.warning("Movement distance effectively zero, creating default movement direction")
                waypoint = [current_pos[0] + min_movement_distance, current_pos[1], current_pos[2]]
                movement_vector = np.array(waypoint) - np.array(current_pos)
                distance = np.linalg.norm(movement_vector)
        
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
            
        logging.debug(f"Moving with yaw: {yaw_degrees:.1f}° at velocity: {velocity} m/s")
        logging.debug(f"Movement vector: [{movement_vector[0]:.2f}, {movement_vector[1]:.2f}, {movement_vector[2]:.2f}]")
        
        # Move drone with yaw control
        client.moveToPositionAsync(
            waypoint[0], waypoint[1], waypoint[2],
            velocity,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_degrees)
        ).join()
    except Exception as e:
        logging.error(f"Error in move_to_waypoint: {e}")
        traceback.print_exc()
        # In case of error, just hover in place
        client.hoverAsync().join()

def sample_visible_target(current_pos: List[float], distance_range: Tuple[float, float], 
                          client: airsim.MultirotorClient, max_attempts: int = 100,
                          episode_id: int = 0, seed: int = None, 
                          ray_checks: int = 5) -> List[float]:
    """Sample a random target position within distance bounds that is visible to the drone
    
    Args:
        current_pos: Current drone position [x, y, z]
        distance_range: (min_distance, max_distance) in meters
        client: AirSim client instance
        max_attempts: Maximum sampling attempts before giving up
        episode_id: Current episode ID to influence randomization
        seed: Random seed for deterministic behavior
        ray_checks: Number of rays to use for validating each target
        
    Returns:
        List[float]: Target position [x, y, z] in NED coordinates
        
    Raises:
        ValueError: If no valid target is found within max_attempts
    """
    min_dist, max_dist = distance_range
    
    # Create a separate random generator for deterministic target generation
    if seed is not None:
        # Use episode_id to get different but deterministic targets for each episode
        target_rng = random.Random(seed + episode_id)
    else:
        target_rng = random.Random()
    
    # Adapt the distance range based on the episode
    # This creates a progression of targets with varying distances
    if episode_id > 0:
        # The early episodes get shorter distances, later episodes get longer distances
        episode_factor = min(1.0, episode_id / 10)  # Scales from 0.1 to 1.0 over first 10 episodes
        
        # Adjust the distance range
        min_dist = min_dist + (episode_factor * 5)  # Gradually increase minimum distance
        max_dist = min(max_dist, min_dist + 20 + (episode_factor * 30))  # Gradually increase range span
    
    logging.info(f"Target distance range for episode {episode_id}: {min_dist:.1f}m - {max_dist:.1f}m")
    
    # Verify the drone's position is valid for sampling
    if current_pos[2] > -1.0:  # NED coordinates: Z is negative for altitude
        logging.warning(f"Sampling target with drone too close to ground (alt: {current_pos[2]:.2f}m)")
    
    valid_targets = []  # Track all valid targets for potential selection
    best_target = None  # Best target based on obstacle clearance
    best_obstacle_clearance = 0.0  # Track the best clearance found
    los_failures = []   # Track target positions with failed line of sight checks for logging
    ray_test_results = []  # Track detailed ray test results for debugging
    
    # Get current drone state once for reference
    try:
        drone_state = client.getMultirotorState().kinematics_estimated
        drone_orientation = [
            drone_state.orientation.w_val,
            drone_state.orientation.x_val,
            drone_state.orientation.y_val,
            drone_state.orientation.z_val
        ]
    except Exception as e:
        logging.error(f"Failed to get drone state: {e}")
        drone_orientation = [1.0, 0.0, 0.0, 0.0]  # Default orientation (no rotation)
    
    for attempt in range(max_attempts):
        # Sample random direction (uniform on sphere)
        theta = target_rng.uniform(0, 2 * math.pi)  # Azimuth angle
        
        # Constrain polar angle to avoid targets that are too vertical
        phi = target_rng.uniform(math.pi/6, math.pi/2.5)  # Between ~30° and ~70° from vertical
        
        # Convert spherical to cartesian coordinates
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        
        # Sample random distance within range
        distance = target_rng.uniform(min_dist, max_dist)
        
        # Scale direction vector by distance
        direction = np.array([x, y, z]) * distance
        
        # Calculate target position in NED coordinates
        target_pos = np.array(current_pos) + direction
        
        # Ensure z-coordinate is negative (below ground level in NED)
        target_pos[2] = min(target_pos[2], -2.0)  # Keep at least 2m below ground level
        
        # Convert to Vector3r for line of sight test
        start_vector = airsim.Vector3r(current_pos[0], current_pos[1], current_pos[2])
        target_vector = airsim.Vector3r(target_pos[0], target_pos[1], target_pos[2])
        
        # Flag to track if this target is valid across all ray checks
        target_valid = True
        current_ray_results = []
        
        # Use multiple rays with slight perturbations to verify line of sight
        for ray_idx in range(ray_checks):
            try:
                # For the first ray, test direct line of sight
                if ray_idx == 0:
                    # Use simTestLineOfSightBetweenPoints for better reliability
                    los_result = client.simTestLineOfSightBetweenPoints(start_vector, target_vector)
                    ray_origin = start_vector
                    ray_end = target_vector
                else:
                    # For additional rays, add small perturbation (±10cm) to ensure robust detection
                    perturb = 0.1  # 10cm perturbation
                    offset = np.array([
                        target_rng.uniform(-perturb, perturb),
                        target_rng.uniform(-perturb, perturb),
                        target_rng.uniform(-perturb, perturb)
                    ])
                    
                    # Create slightly perturbed start and end points
                    perturbed_start = airsim.Vector3r(
                        current_pos[0] + offset[0], 
                        current_pos[1] + offset[1], 
                        current_pos[2] + offset[2]
                    )
                    perturbed_target = airsim.Vector3r(
                        target_pos[0] + offset[0], 
                        target_pos[1] + offset[1], 
                        target_pos[2] + offset[2]
                    )
                    
                    # Check line of sight with perturbed rays
                    los_result = client.simTestLineOfSightBetweenPoints(perturbed_start, perturbed_target)
                    ray_origin = perturbed_start
                    ray_end = perturbed_target
                
                # Store ray test result for debugging
                current_ray_results.append({
                    "ray_idx": ray_idx,
                    "origin": [ray_origin.x_val, ray_origin.y_val, ray_origin.z_val],
                    "end": [ray_end.x_val, ray_end.y_val, ray_end.z_val],
                    "result": los_result
                })
                
                # If any ray test fails, mark the target as invalid
                if not los_result:
                    target_valid = False
                    # For debugging, attempt to get more detailed collision information
                    try:
                        # Try to use the hit test API for more details
                        hit_result = client.simGetCollisionInfo()
                        if hit_result.has_collided:
                            collision_point = [
                                hit_result.impact_point.x_val,
                                hit_result.impact_point.y_val,
                                hit_result.impact_point.z_val
                            ]
                            logging.debug(f"Ray {ray_idx} hit obstacle at {collision_point}")
                    except Exception as hit_error:
                        logging.debug(f"Could not get collision details: {hit_error}")
                    
                    # Don't check additional rays if one already failed
                    break
                
            except Exception as e:
                logging.warning(f"Error in line of sight test for ray {ray_idx}: {e}")
                target_valid = False
                break
                
        # Store all ray test results for debugging
        ray_test_results.append({
            "attempt": attempt,
            "target": target_pos.tolist(),
            "rays": current_ray_results,
            "valid": target_valid
        })
        
        # If target passed all ray checks, proceed with validation
        if target_valid:
            # Target is visible - get additional information about clearance if possible
            valid_targets.append(target_pos.tolist())
            
            # Check obstacle clearance around target
            try:
                # Use simplified clearance check with line of sight tests in multiple directions
                clearance_dirs = [
                    np.array([1, 0, 0]), np.array([-1, 0, 0]),
                    np.array([0, 1, 0]), np.array([0, -1, 0]),
                    np.array([0, 0, 1]), np.array([0, 0, -1])
                ]
                
                min_obstacle_dist = float('inf')
                
                for direction in clearance_dirs:
                    # Check line of sight in this direction from target
                    # Try distances from 1m to 10m
                    for dist in range(1, 11):
                        check_point = target_pos + (direction * dist)
                        check_vector = airsim.Vector3r(check_point[0], check_point[1], check_point[2])
                        
                        # If line of sight fails, we found the obstacle distance
                        if not client.simTestLineOfSightBetweenPoints(target_vector, check_vector):
                            if dist < min_obstacle_dist:
                                min_obstacle_dist = dist
                            break
                
                # If we found a valid obstacle distance
                if min_obstacle_dist < float('inf'):
                    # If this target has better clearance than our previous best, update it
                    if min_obstacle_dist > best_obstacle_clearance:
                        best_obstacle_clearance = min_obstacle_dist
                        best_target = target_pos.tolist()
                        logging.debug(f"Better target found with {min_obstacle_dist:.2f}m obstacle clearance")
                
                # If we found a really good target (>5m clearance), stop early
                if best_obstacle_clearance > 5.0:
                    logging.info(f"Found excellent target with {best_obstacle_clearance:.2f}m obstacle clearance")
                    return best_target
            except Exception as e:
                logging.debug(f"Error checking obstacle clearance near target: {e}")
            
            # Log that we found a valid target
            logging.debug(f"Found valid target at {target_pos.tolist()} (attempt {attempt+1})")
            
            # If this is the first valid target, we'll use it as our initial best
            if best_target is None:
                best_target = target_pos.tolist()
                logging.info(f"Found first valid target at {best_target}")
            
            # Every 10 valid targets, report progress
            if len(valid_targets) % 10 == 0:
                logging.info(f"Found {len(valid_targets)} valid targets so far")
            
            # If we've found at least 3 valid targets, we can start being selective
            if len(valid_targets) >= 3:
                # Return the best target we've found so far
                if best_target:
                    return best_target
                # Or just return the last valid target
                return valid_targets[-1]
        else:
            # For failed targets, store position for visualization
            los_failures.append(target_pos.tolist())
            
    # After all attempts, if we have debugging enabled, save visualizations of ray tests
    if logging.getLogger().level <= logging.DEBUG and len(ray_test_results) > 0:
        debug_dir = "debug_raycasts"
        try:
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f"raycast_debug_ep{episode_id}.json")
            with open(debug_file, "w") as f:
                json.dump({
                    "ray_tests": ray_test_results,
                    "failures": los_failures,
                    "valid_targets": valid_targets,
                    "drone_pos": current_pos,
                    "drone_orientation": drone_orientation
                }, f, cls=NumpyJSONEncoder, indent=2)
            logging.debug(f"Saved raycast debug information to {debug_file}")
        except Exception as e:
            logging.warning(f"Failed to save raycast debug information: {e}")
    
    # After exhausting all attempts, return the best target if found
    if best_target:
        logging.info(f"Using best target found with {best_obstacle_clearance:.2f}m clearance")
        return best_target
    
    # Or return any valid target we found
    if valid_targets:
        logging.info(f"No ideal target found, using one of {len(valid_targets)} valid targets")
        return target_rng.choice(valid_targets)  # Random choice from valid targets
    
    # If we couldn't find ANY valid target, try one last direct check with a fixed target
    # This helps in environments where random sampling might be challenging
    try:
        logging.warning("No valid targets found with random sampling, trying a fixed fallback target")
        # Try a target directly ahead of the drone at min distance
        forward_vector = np.array([1.0, 0.0, 0.0])  # Forward in NED
        forward_target_pos = np.array(current_pos) + (forward_vector * min_dist)
        forward_target_pos[2] = min(forward_target_pos[2], -2.0)  # Keep below ground level
        
        # Check if this forward target is valid
        forward_vector3r = airsim.Vector3r(forward_target_pos[0], forward_target_pos[1], forward_target_pos[2])
        forward_los_result = client.simTestLineOfSightBetweenPoints(start_vector, forward_vector3r)
        
        if forward_los_result:
            logging.info(f"Using fallback forward target at {forward_target_pos.tolist()}")
            return forward_target_pos.tolist()
    except Exception as e:
        logging.warning(f"Error checking fallback target: {e}")
    
    # If we couldn't find ANY valid target, raise an error
    raise ValueError(f"No unobstructed target found after {max_attempts} attempts")

def visualize_raycasts(client, raycast_data_file, duration=60.0):
    """Visualize the results of raycasting tests by drawing lines in the AirSim environment
    
    Args:
        client: AirSim client instance
        raycast_data_file: Path to a JSON file containing raycast debug information
        duration: Duration in seconds for visualization markers to remain visible
    """
    try:
        # Load raycast debug data
        with open(raycast_data_file, 'r') as f:
            raycast_data = json.load(f)
        
        # Clear previous visualizations
        client.simFlushPersistentMarkers()
        
        # Draw drone position
        drone_pos = raycast_data.get("drone_pos", [0, 0, 0])
        client.simAddMarker(
            airsim.Vector3r(drone_pos[0], drone_pos[1], drone_pos[2]),
            1.0,  # size
            [0, 0, 255, 255],  # blue
            "drone_position",
            duration  # duration in seconds
        )
        
        # Draw successful rays in green
        for test_result in raycast_data.get("ray_tests", []):
            if test_result.get("valid", False):
                target = test_result.get("target", [0, 0, 0])
                # Draw successful target
                client.simAddMarker(
                    airsim.Vector3r(target[0], target[1], target[2]),
                    0.5,  # size
                    [0, 255, 0, 255],  # green
                    f"valid_target_{target[0]}_{target[1]}_{target[2]}",
                    duration  # duration
                )
                
                # Draw rays for this target
                for ray in test_result.get("rays", []):
                    if ray.get("result", False):
                        origin = ray.get("origin", [0, 0, 0])
                        end = ray.get("end", [0, 0, 0])
                        # Draw successful ray
                        client.simDrawLine(
                            airsim.Vector3r(origin[0], origin[1], origin[2]),
                            airsim.Vector3r(end[0], end[1], end[2]),
                            0.05,  # thickness
                            [0, 255, 0, 255],  # green
                            duration  # duration
                        )
        
        # Draw failed rays in red
        for test_result in raycast_data.get("ray_tests", []):
            if not test_result.get("valid", True):
                target = test_result.get("target", [0, 0, 0])
                # Draw failed target
                client.simAddMarker(
                    airsim.Vector3r(target[0], target[1], target[2]),
                    0.5,  # size
                    [255, 0, 0, 255],  # red
                    f"invalid_target_{target[0]}_{target[1]}_{target[2]}",
                    duration  # duration
                )
                
                # Draw rays for this target
                for ray in test_result.get("rays", []):
                    origin = ray.get("origin", [0, 0, 0])
                    end = ray.get("end", [0, 0, 0])
                    color = [255, 0, 0, 255] if not ray.get("result", True) else [0, 255, 0, 255]
                    # Draw ray
                    client.simDrawLine(
                        airsim.Vector3r(origin[0], origin[1], origin[2]),
                        airsim.Vector3r(end[0], end[1], end[2]),
                        0.05,  # thickness
                        color,
                        duration  # duration
                    )
        
        # Draw all failed targets
        for failed_target in raycast_data.get("failures", []):
            client.simAddMarker(
                airsim.Vector3r(failed_target[0], failed_target[1], failed_target[2]),
                0.3,  # size
                [255, 0, 0, 128],  # red with some transparency
                f"failed_{failed_target[0]}_{failed_target[1]}_{failed_target[2]}",
                duration  # duration
            )
        
        logging.info(f"Visualized raycast debug data from {raycast_data_file} (visible for {duration}s)")
        return True
    except Exception as e:
        logging.error(f"Failed to visualize raycasts: {e}")
        traceback.print_exc()
        return False

def run_episode(episode_id: int, client: airsim.MultirotorClient, 
                zmq_interface: ZMQInterface, scanner: Scanner, 
                config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run a single navigation episode
    
    Args:
        episode_id: Episode identifier
        client: AirSim client instance
        zmq_interface: ZMQ communication interface
        scanner: Scanner for obstacle detection
        config: Experiment configuration
        
    Returns:
        Tuple containing per-step metrics and episode summary
    """
    logging.info(f"=== Starting Episode {episode_id} ===")
    
    # Reset the drone
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Takeoff - CRITICAL: We must take off before checking target visibility
    logging.info(f"Episode {episode_id}: Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)  # Extended stabilization time to ensure proper altitude before sampling target
    
    # Get initial position - AFTER takeoff
    drone_state = client.getMultirotorState().kinematics_estimated
    drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
    
    # Ensure drone is at a reasonable altitude before sampling targets
    if drone_pos[2] > -1.0:  # In NED coordinates, negative Z is up
        logging.warning(f"Episode {episode_id}: Drone altitude too low ({drone_pos[2]:.2f}), adjusting height")
        # Move up to ensure better visibility for target sampling
        adjusted_height_pos = [drone_pos[0], drone_pos[1], -5.0]  # Minimum 5m above ground
        client.moveToPositionAsync(
            adjusted_height_pos[0], adjusted_height_pos[1], adjusted_height_pos[2], 2
        ).join()
        time.sleep(1)  # Wait for height adjustment
        
        # Update position after height adjustment
        drone_state = client.getMultirotorState().kinematics_estimated
        drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
        logging.info(f"Episode {episode_id}: Adjusted drone height to {drone_pos[2]:.2f}m")
    
    # Configure number of rays for target validation based on environment complexity
    # Use more rays for later episodes which might be more challenging
    min_rays = config.get("min_ray_checks", 3)
    max_rays = config.get("max_ray_checks", 7)
    ray_checks = min(max_rays, min_rays + (episode_id // 3))  # Start with min_rays, gradually increase
    
    # Enable debug log level temporarily for detailed raycast information if needed
    original_log_level = logging.getLogger().level
    if config.get("debug_raycasting", False) and episode_id % 5 == 0:  # Debug every 5th episode
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(f"Episode {episode_id}: Enhanced logging enabled for raycast debugging")
        logging.debug(f"Using {ray_checks} rays for target validation")
    
    # Sample a random target - NOW we can safely check visibility
    target_sampling_start = time.time()
    try:
        target_pos = sample_visible_target(
            drone_pos, 
            config["target_distance_range"], 
            client,
            episode_id=episode_id,
            seed=config["random_seed"],
            ray_checks=ray_checks
        )
        target_sampling_time = time.time() - target_sampling_start
        logging.info(f"Episode {episode_id}: Target sampling took {target_sampling_time:.2f}s with {ray_checks} rays")
    except ValueError as e:
        logging.warning(f"Episode {episode_id}: {str(e)}")
        
        # Restore original log level if it was changed
        if config.get("debug_raycasting", False) and episode_id % 5 == 0:
            logging.getLogger().setLevel(original_log_level)
            
        return [], {
            "episode_id": episode_id,
            "status": "skipped",
            "reason": "No valid target found"
        }
    
    # Restore original log level if it was changed
    if config.get("debug_raycasting", False) and episode_id % 5 == 0:
        logging.getLogger().setLevel(original_log_level)
    
    # Initialize episode tracking
    step_metrics = []
    collisions = 0
    replanning_count = 0
    start_time = time.time()
    trajectory = [drone_pos]
    
    # Initialize stuck detection variables
    last_significant_movement_time = start_time
    last_position = np.array(drone_pos)
    stuck_timeout = config.get("stuck_timeout", 15.0)  # Time in seconds to consider the drone stuck
    min_movement_threshold = 0.5  # Minimum movement in meters to consider significant
    stuck_check_interval = config.get("stuck_check_interval", 3.0)  # Only check for stuck condition every X seconds
    last_stuck_check_time = start_time
    time_since_movement = 0.0  # Initialize time_since_movement to avoid UnboundLocalError
    
    # Initial distance to target
    distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(drone_pos))
    initial_distance = distance_to_target
    
    logging.info(f"Episode {episode_id}: Initial position: {[round(p, 2) for p in drone_pos]}")
    logging.info(f"Episode {episode_id}: Target position: {[round(p, 2) for p in target_pos]}")
    logging.info(f"Episode {episode_id}: Initial distance: {distance_to_target:.2f}m")
    
    # Additional validation: Double-check target visibility with multiple rays
    verify_rays = 3  # Use a few rays for verification
    verification_failures = 0
    
    for v_idx in range(verify_rays):
        try:
            # Slight perturbation for robustness
            perturb = 0.05  # 5cm perturbation
            offset = np.array([
                random.uniform(-perturb, perturb),
                random.uniform(-perturb, perturb),
                random.uniform(-perturb, perturb)
            ])
            
            # Create perturbed points
            start_point = airsim.Vector3r(
                drone_pos[0] + offset[0], 
                drone_pos[1] + offset[1], 
                drone_pos[2] + offset[2]
            )
            end_point = airsim.Vector3r(
                target_pos[0] + offset[0], 
                target_pos[1] + offset[1], 
                target_pos[2] + offset[2]
            )
            
            # Verify line of sight
            los_result = client.simTestLineOfSightBetweenPoints(start_point, end_point)
            if not los_result:
                verification_failures += 1
                logging.warning(f"Target verification ray {v_idx} failed")
        except Exception as e:
            logging.warning(f"Error during target verification: {e}")
            verification_failures += 1
    
    if verification_failures > 0:
        logging.warning(f"Episode {episode_id}: Target verification detected possible issues ({verification_failures}/{verify_rays} rays failed)")
    else:
        logging.info(f"Episode {episode_id}: Target verified with {verify_rays} rays")
    
    # Main navigation loop
    status = "timeout"  # Default status
    for step in range(config["max_steps_per_episode"]):
        step_start_time = time.time()
        
        # Check for episode timeout
        if time.time() - start_time > config.get("episode_timeout", 120):
            logging.warning(f"Episode {episode_id}: Timeout after {config.get('episode_timeout', 120)} seconds")
            status = "timeout"
            break
        
        # Get current drone position
        drone_state = client.getMultirotorState().kinematics_estimated
        drone_pos = [
            drone_state.position.x_val,
            drone_state.position.y_val,
            drone_state.position.z_val
        ]
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(drone_pos))
        
        # Check if target reached - IMPORTANT: Do this before stuck detection
        if distance_to_target <= ARRIVAL_THRESHOLD:
            logging.info(f"Episode {episode_id}: Target reached at step {step}")
            status = "success"
            break
        
        # Check if drone has moved significantly
        current_position = np.array(drone_pos)
        movement = np.linalg.norm(current_position - last_position)
        
        if movement > min_movement_threshold:
            # Drone has moved significantly, update time and position
            last_significant_movement_time = time.time()
            last_position = current_position
            logging.debug(f"Episode {episode_id}: Significant movement detected: {movement:.2f}m")
        
        # Only check for stuck condition at specified intervals (not every step)
        # And don't check when drone is close to the target (within 2x the arrival threshold)
        current_time = time.time()
        if (current_time - last_stuck_check_time >= stuck_check_interval and 
                distance_to_target > ARRIVAL_THRESHOLD * 2):
            
            # Update the last check time
            last_stuck_check_time = current_time
            
            # Check if drone is stuck (no significant movement for too long)
            time_since_movement = current_time - last_significant_movement_time
            if time_since_movement > stuck_timeout:
                logging.error(f"Episode {episode_id}: Drone stuck for {time_since_movement:.1f}s (>= {stuck_timeout}s), aborting")
                status = "stuck"
                break
                
            # Log stuck detection progress if approaching timeout
            if time_since_movement > stuck_timeout * 0.6:  # At 60% of timeout
                logging.warning(f"Episode {episode_id}: No significant movement for {time_since_movement:.1f}s " +
                              f"(timeout: {stuck_timeout}s)")

        # Check for collisions
        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided:
            collisions += 1
            logging.warning(f"Episode {episode_id}: Collision detected at step {step}")
        
        # Get obstacle data
        obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
        
        # Calculate obstacle density metrics for adaptive planning
        closest_obstacle_dist = min(obstacle_distances) if obstacle_distances else float('inf')
        obstacle_count = len(obstacle_positions)
        
        # Determine if we're in a high-density obstacle area
        high_density_area = obstacle_count > 10 or closest_obstacle_dist < 3.0
        
        # Calculate proximity to goal for adaptive behavior
        near_goal = distance_to_target < 5.0  # Consider 5m to target as "near goal"
        
        # Modify the DENSITY_RADIUS parameter based on obstacle density and goal proximity
        adaptive_density_radius = DENSITY_RADIUS
        if high_density_area:
            # Reduce density radius in high-density areas for more precise planning
            adaptive_density_radius = max(2.0, DENSITY_RADIUS * 0.6)
            logging.debug(f"Episode {episode_id}: High obstacle density area - reduced density radius to {adaptive_density_radius:.1f}m")
        
        # Modify safety margin based on goal proximity - allow slightly closer approach near goal
        adaptive_safety_margin = MARGIN
        if near_goal:
            # Slightly reduce margin near goal but maintain safety
            adaptive_safety_margin = max(0.8, MARGIN * 0.85) 
            logging.debug(f"Episode {episode_id}: Near goal - adjusted safety margin to {adaptive_safety_margin:.1f}m")

        # Log obstacle detection info for debugging
        if step % 10 == 0 or high_density_area:  # Log more frequently in high-density areas
            logging.debug(f"Episode {episode_id}: Detected {len(obstacle_positions)} obstacles")
            if len(obstacle_positions) > 0:
                min_dist = min(obstacle_distances) if obstacle_distances else "N/A"
                logging.debug(f"Episode {episode_id}: Closest obstacle at {min_dist}m")
        
        # Create observation for active inference with adaptive parameters
        observation = {
            "drone_position": drone_pos,
            "target_position": target_pos,
            "obstacle_positions": obstacle_positions,
            "obstacle_distances": obstacle_distances,
            "waypoint_count": WAYPOINT_SAMPLE_COUNT,
            "safety_margin": adaptive_safety_margin,
            "policy_length": POLICY_LENGTH,
            "density_radius": adaptive_density_radius,
            "near_goal": near_goal,  # Add flag to inform planner we're near the goal
            "distance_to_target": distance_to_target  # Explicitly include distance to target
        }
        
        # Verify the observation is valid before sending
        has_nans = any(np.isnan(np.array(drone_pos))) or any(np.isnan(np.array(target_pos)))
        has_infs = any(np.isinf(np.array(drone_pos))) or any(np.isinf(np.array(target_pos)))
        
        if has_nans or has_infs:
            logging.error(f"Episode {episode_id}: Invalid position data detected (NaN or Inf)")
            status = "invalid_data"
            break

        # Get next waypoint and planning metrics
        inference_start = time.time()
        next_waypoint, policy = zmq_interface.send_observation_and_receive_action(observation)
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms
        
        # Check if planning succeeded
        if next_waypoint is None:
            logging.error(f"Episode {episode_id}: Failed to get valid waypoint at step {step}")
            retry_count = 0
            # Try replanning up to 3 times
            while next_waypoint is None and retry_count < 3:
                retry_count += 1
                replanning_count += 1
                logging.info(f"Episode {episode_id}: Replanning attempt {retry_count}")
                next_waypoint, policy = zmq_interface.send_observation_and_receive_action(observation)
            
            if next_waypoint is None:
                logging.error(f"Episode {episode_id}: Aborting after failed replanning")
                status = "planning_failure"
                break
        
        # Verify waypoint is valid
        waypoint_array = np.array(next_waypoint)
        if np.any(np.isnan(waypoint_array)) or np.any(np.isinf(waypoint_array)):
            logging.error(f"Episode {episode_id}: Invalid waypoint received (contains NaN or Inf)")
            status = "invalid_waypoint"
            break
        
        # Calculate action magnitude (waypoint distance)
        action_magnitude = np.linalg.norm(np.array(next_waypoint) - np.array(drone_pos))
        
        # Adaptively adjust waypoint based on obstacle density and goal proximity
        # In high-density areas, we want to maintain the direction but potentially reduce step size
        if high_density_area and not near_goal and action_magnitude > 3.0:
            # In high density areas (but not near goal), we limit large movements for safety
            # Scale down the waypoint distance to avoid large jumps in dense areas
            direction = (waypoint_array - np.array(drone_pos)) / action_magnitude
            adjusted_distance = min(3.0, action_magnitude)  # Cap at 3m in dense areas
            next_waypoint = (np.array(drone_pos) + direction * adjusted_distance).tolist()
            logging.debug(f"Episode {episode_id}: Scaled down waypoint in dense area from {action_magnitude:.2f}m to {adjusted_distance:.2f}m")
            # Update action magnitude after scaling
            action_magnitude = adjusted_distance
        
        # Near goal with high obstacle density - prioritize target approach with careful steps
        if near_goal and high_density_area:
            # Calculate direction to target
            to_target = np.array(target_pos) - np.array(drone_pos)
            to_target_dist = np.linalg.norm(to_target)
            to_target_dir = to_target / to_target_dist if to_target_dist > 0 else np.array([1, 0, 0])
            
            # Calculate direction of planned waypoint
            to_waypoint = np.array(next_waypoint) - np.array(drone_pos)
            to_waypoint_dist = np.linalg.norm(to_waypoint)
            to_waypoint_dir = to_waypoint / to_waypoint_dist if to_waypoint_dist > 0 else np.array([1, 0, 0])
            
            # Calculate dot product to see if waypoint is roughly in direction of target
            alignment = np.dot(to_target_dir, to_waypoint_dir)
            
            # If waypoint is not aligned with target direction and we're near the goal
            # in a dense area, blend the directions to favor target approach
            if alignment < 0.7 and distance_to_target < 3.0:  # Less than 70% aligned and very close
                # Create a blended direction (60% to target, 40% original waypoint direction)
                blended_dir = 0.6 * to_target_dir + 0.4 * to_waypoint_dir
                blended_dir = blended_dir / np.linalg.norm(blended_dir)  # Normalize
                
                # Use a conservative step size in this sensitive region
                step_size = min(1.5, distance_to_target * 0.6)  # 60% of distance to target or 1.5m max
                
                # Create new waypoint that's more aligned with target
                next_waypoint = (np.array(drone_pos) + blended_dir * step_size).tolist()
                logging.debug(f"Episode {episode_id}: Near goal in dense area - adjusted waypoint for better target alignment")
                
                # Update action magnitude after adjustment
                action_magnitude = step_size
        
        # Extract additional metrics from policy if available
        # These would need to be included in the Julia server response
        vfe = policy[0].get("vfe", 0.0) if policy and len(policy) > 0 else 0.0
        efe = policy[0].get("efe", 0.0) if policy and len(policy) > 0 else 0.0
        suitability = policy[0].get("suitability", 0.0) if policy and len(policy) > 0 else 0.0
        
        # Record step metrics
        step_data = {
            "episode_id": episode_id,
            "step": step,
            "distance_to_target": distance_to_target,
            "normalized_distance": distance_to_target / initial_distance,
            "position_x": drone_pos[0],
            "position_y": drone_pos[1],
            "position_z": drone_pos[2],
            "waypoint_x": next_waypoint[0],
            "waypoint_y": next_waypoint[1],
            "waypoint_z": next_waypoint[2],
            "action_magnitude": action_magnitude,
            "policy_length": len(policy) if policy else 0,
            "vfe": vfe,
            "efe": efe,
            "suitability": suitability,
            "collision": collision_info.has_collided,
            "obstacles_count": len(obstacle_positions),
            "closest_obstacle": closest_obstacle_dist if obstacle_distances else float('inf'),
            "high_density_area": high_density_area,
            "near_goal": near_goal,
            "inference_time_ms": inference_time,
            "replanning_occurred": retry_count > 0 if 'retry_count' in locals() else False,
            "time_since_movement": time_since_movement
        }
        step_metrics.append(step_data)
        
        # Move drone to waypoint
        move_to_waypoint(client, drone_pos, next_waypoint)
        
        # Record trajectory
        trajectory.append(next_waypoint)
        
        # Brief pause to stabilize
        time.sleep(0.2)
        
        # Check step duration for performance monitoring
        step_duration = time.time() - step_start_time
        if step_duration > 5.0:  # If a step takes more than 5 seconds, it might indicate issues
            logging.warning(f"Episode {episode_id}: Step {step} took {step_duration:.2f}s (unusually long)")
    
    # Episode ended - calculate final metrics
    episode_duration = time.time() - start_time

    # Final position and distance
    drone_state = client.getMultirotorState().kinematics_estimated
    final_pos = [
        drone_state.position.x_val,
        drone_state.position.y_val,
        drone_state.position.z_val
    ]
    final_distance = np.linalg.norm(np.array(target_pos) - np.array(final_pos))
    
    # Land the drone
    client.landAsync().join()
    client.armDisarm(False)
    
    # Compile episode summary
    episode_summary = {
        "episode_id": episode_id,
        "status": status,
        "steps_taken": len(step_metrics),
        "target_reached": status == "success",
        "final_distance": final_distance,
        "initial_distance": initial_distance,
        "distance_improvement": initial_distance - final_distance,
        "normalized_improvement": (initial_distance - final_distance) / initial_distance,
        "collisions": collisions,
        "replanning_count": replanning_count,
        "replanning_percentage": (replanning_count / len(step_metrics) * 100) if step_metrics else 0,
        "duration_seconds": episode_duration,
        "avg_step_time": episode_duration / len(step_metrics) if step_metrics else 0,
        "avg_inference_time_ms": np.mean([m["inference_time_ms"] for m in step_metrics]) if step_metrics else 0,
        "stuck_detected": status == "stuck"
    }
    
    logging.info(f"Episode {episode_id} completed: {status}")
    logging.info(f"  Steps: {len(step_metrics)}, Duration: {episode_duration:.2f}s")
    logging.info(f"  Final distance: {final_distance:.2f}m, Collisions: {collisions}")
    
    return step_metrics, episode_summary

def run_experiment(config: Dict[str, Any]) -> None:
    """Run a full navigation experiment with multiple episodes
    
    Args:
        config: Experiment configuration
    """
    # Merge with default config
    full_config = {**DEFAULT_CONFIG, **config}
    
    # Set up logging for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = full_config.get("output_dir", "experiment_results")
    experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Configure log file
    log_file = os.path.join(experiment_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Create debug directories if needed
    if full_config.get("debug_raycasting", False):
        debug_dir = os.path.join(experiment_dir, "debug_raycasts")
        os.makedirs(debug_dir, exist_ok=True)
    
    # Set random seed
    random_seed = full_config.get("random_seed")
    if random_seed is not None:
        logging.info(f"Using random seed: {random_seed}")
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Log configuration
    logging.info(f"Starting experiment with configuration:")
    for key, value in full_config.items():
        logging.info(f"  {key}: {value}")
    
    # Save configuration
    config_file = os.path.join(experiment_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(full_config, f, cls=NumpyJSONEncoder, indent=2)
    
    # Initialize AirSim client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Initialize scanner
    scanner = Scanner(client)
    
    # Start the Julia server
    julia_server = JuliaServer()
    
    # Precompile Julia packages (only if needed)
    if full_config.get("precompile_julia", True):
        julia_server.precompile_packages()
    
    # Start the server
    server_port = full_config.get("zmq_port", DEFAULT_ZMQ_PORT)
    julia_server.start_server(port=server_port)
    
    # Initialize ZMQ interface
    zmq_interface = ZMQInterface(server_port=server_port)
    
    # Create all episode metrics lists
    all_episodes_metrics = []
    episode_summaries = []
    
    # Run episodes
    for episode_id in range(full_config["num_episodes"]):
        # Run episode
        step_metrics, episode_summary = run_episode(
            episode_id, client, zmq_interface, scanner, full_config
        )
        
        # Track metrics
        if step_metrics:
            all_episodes_metrics.extend(step_metrics)
            episode_summaries.append(episode_summary)
            
            # Visualize debug data if enabled and available
            if full_config.get("debug_raycasting", False):
                debug_file = os.path.join("debug_raycasts", f"raycast_debug_ep{episode_id}.json")
                if os.path.exists(debug_file):
                    # Move the debug file to the experiment directory
                    experiment_debug_dir = os.path.join(experiment_dir, "debug_raycasts")
                    os.makedirs(experiment_debug_dir, exist_ok=True)
                    experiment_debug_file = os.path.join(experiment_debug_dir, f"raycast_debug_ep{episode_id}.json")
                    try:
                        os.rename(debug_file, experiment_debug_file)
                    except Exception:
                        pass
                    
                    # Visualize raycasts if requested
                    if full_config.get("visualize_raycasts", False) and episode_id % 3 == 0:
                        visualization_duration = full_config.get("visualization_duration", 60.0)
                        visualize_raycasts(client, experiment_debug_file, duration=visualization_duration)
        
        # Export metrics after each episode (to avoid data loss in case of crash)
        if all_episodes_metrics:
            metrics_file = os.path.join(experiment_dir, "metrics.csv")
            pd.DataFrame(all_episodes_metrics).to_csv(metrics_file, index=False)
            
            summary_file = os.path.join(experiment_dir, "episode_summaries.csv")
            pd.DataFrame(episode_summaries).to_csv(summary_file, index=False)
    
    # Shut down Julia server
    julia_server.shutdown()
    
    # Final export of all metrics
    if all_episodes_metrics:
        metrics_file = os.path.join(experiment_dir, "metrics.csv")
        pd.DataFrame(all_episodes_metrics).to_csv(metrics_file, index=False)
        
        summary_file = os.path.join(experiment_dir, "episode_summaries.csv")
        pd.DataFrame(episode_summaries).to_csv(summary_file, index=False)
        
        # Create plots
        try:
            plot_metrics(metrics_file, os.path.join(experiment_dir, "metrics_plots"))
            logging.info(f"Metrics plots saved to {os.path.join(experiment_dir, 'metrics_plots')}")
        except Exception as e:
            logging.error(f"Failed to create plots: {e}")
    
    logging.info(f"Experiment completed. Results saved to {experiment_dir}")

def plot_metrics(metrics_file, output_dir):
    """Create plots for the experiment metrics
    
    Args:
        metrics_file: Path to the CSV file with metrics
        output_dir: Directory to save the plots
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metrics data
        df = pd.read_csv(metrics_file)
        
        # Load episode summaries if available
        summary_file = metrics_file.replace("metrics.csv", "episode_summaries.csv")
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
        else:
            # Generate summary from metrics if needed
            episode_stats = df.groupby("episode_id").agg({
                "distance_to_target": ["first", "last", "min"],
                "step": "max",
                "collision": "sum",
                "inference_time_ms": "mean"
            })
            episode_stats.columns = ["initial_distance", "final_distance", "min_distance", 
                                    "steps_taken", "collisions", "avg_inference_time_ms"]
            
            # Calculate if target was reached
            target_threshold = 2.0  # Assume this is the arrival threshold
            episode_stats["target_reached"] = episode_stats["min_distance"] < target_threshold
            
            # Calculate normalized improvement
            episode_stats["distance_improvement"] = episode_stats["initial_distance"] - episode_stats["final_distance"]
            episode_stats["normalized_improvement"] = episode_stats["distance_improvement"] / episode_stats["initial_distance"]
            
            summary_df = episode_stats.reset_index()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot trajectory for each episode (if position data available)
        if "drone_x" in df.columns and "drone_y" in df.columns:
            for episode_id in df["episode_id"].unique():
                episode_data = df[df["episode_id"] == episode_id]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(episode_data["drone_x"], episode_data["drone_y"], 'b-', label="Trajectory")
                ax.scatter(episode_data["drone_x"].iloc[0], episode_data["drone_y"].iloc[0], 
                        c='green', s=100, marker='^', label="Start")
                
                # Plot target if available
                if "target_x" in episode_data.columns and "target_y" in episode_data.columns:
                    target_x = episode_data["target_x"].iloc[0]
                    target_y = episode_data["target_y"].iloc[0]
                    ax.scatter(target_x, target_y, c='red', s=100, marker='*', label="Target")
                
                # Plot end position
                ax.scatter(episode_data["drone_x"].iloc[-1], episode_data["drone_y"].iloc[-1], 
                        c='orange', s=100, marker='o', label="End")
                
                # Add grid, legend, and labels
                ax.grid(True)
                ax.legend()
                ax.set_title(f"Episode {episode_id} Trajectory")
                ax.set_xlabel("X Position (m)")
                ax.set_ylabel("Y Position (m)")
                ax.axis('equal')
                
                # Save figure
                plt.savefig(os.path.join(output_dir, f"trajectory_episode_{episode_id}.png"))
                plt.close(fig)
        
        # 2. Plot episode summaries and overall statistics
        if not summary_df.empty:
            # Create summary figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Success rate calculation
            successful = summary_df[summary_df["target_reached"] == True]
            success_rate = len(successful) / len(summary_df) if len(summary_df) > 0 else 0
            
            # Plot 1: Success rate over episodes
            rolling_success = summary_df["target_reached"].astype(int).rolling(
                window=min(5, len(summary_df)), min_periods=1
            ).mean()
            axes[0, 0].plot(rolling_success.index, rolling_success)
            axes[0, 0].set_title(f"Success Rate: {success_rate:.1%}")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Rolling Success Rate (5-ep window)")
            axes[0, 0].set_ylim(0, 1.1)
            
            # Plot 2: Steps per episode
            axes[0, 1].bar(summary_df["episode_id"], summary_df["steps_taken"])
            axes[0, 1].set_title("Steps per Episode")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Step Count")
            
            # Plot 3: Average inference time
            axes[1, 0].plot(summary_df["episode_id"], summary_df["avg_inference_time_ms"])
            axes[1, 0].set_title("Average Inference Time")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Time (ms)")
            
            # Plot 4: Performance by distance
            if "normalized_improvement" in summary_df.columns:
                scatter = axes[1, 1].scatter(
                    summary_df["initial_distance"], 
                    summary_df["normalized_improvement"],
                    c=summary_df["target_reached"].astype(bool).map({True: 'green', False: 'red'}),
                    alpha=0.7
                )
                axes[1, 1].set_title("Performance by Initial Distance")
                axes[1, 1].set_xlabel("Initial Distance (m)")
                axes[1, 1].set_ylabel("Normalized Improvement")
                
                # Add legend for target reached status
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                        markersize=10, label='Target Reached'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                        markersize=10, label='Target Not Reached')
                ]
                axes[1, 1].legend(handles=legend_elements)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "episode_summary.png"))
            plt.close(fig)
            
            # Additional plot: Collision count per episode
            if "collisions" in summary_df.columns:
                plt.figure(figsize=(10, 5))
                plt.bar(summary_df["episode_id"], summary_df["collisions"])
                plt.title("Collisions per Episode")
                plt.xlabel("Episode")
                plt.ylabel("Collision Count")
                plt.savefig(os.path.join(output_dir, "collision_count.png"))
                plt.close()
        
        # 3. Plot metrics across all episodes
        # Distance to target over time for all episodes
        plt.figure(figsize=(12, 6))
        for episode_id in df["episode_id"].unique():
            episode_data = df[df["episode_id"] == episode_id]
            plt.plot(episode_data["step"], episode_data["distance_to_target"], 
                    label=f"Episode {episode_id}")
        
        plt.title("Distance to Target Over Time")
        plt.xlabel("Step")
        plt.ylabel("Distance (m)")
        plt.grid(True)
        
        # Add custom legend or remove if too many episodes
        if len(df["episode_id"].unique()) <= 10:
            plt.legend()
        
        plt.savefig(os.path.join(output_dir, "distance_to_target.png"))
        plt.close()
        
        # 4. Create raycast visualization summary if debug data exists
        debug_dir = os.path.join(os.path.dirname(metrics_file), "debug_raycasts")
        if os.path.exists(debug_dir):
            raycast_files = [f for f in os.listdir(debug_dir) if f.endswith('.json')]
            
            if raycast_files:
                # Create a summary of raycast success rates
                raycast_stats = []
                
                for raycast_file in raycast_files:
                    try:
                        with open(os.path.join(debug_dir, raycast_file), 'r') as f:
                            data = json.load(f)
                            
                            # Extract episode number
                            episode_id = int(raycast_file.split('_')[-1].split('.')[0].replace('ep', ''))
                            
                            # Calculate success rate
                            valid_tests = sum(1 for test in data.get("ray_tests", []) if test.get("valid", False))
                            total_tests = len(data.get("ray_tests", []))
                            
                            if total_tests > 0:
                                raycast_stats.append({
                                    "episode_id": episode_id,
                                    "valid_tests": valid_tests,
                                    "total_tests": total_tests,
                                    "success_rate": valid_tests / total_tests,
                                    "failures": len(data.get("failures", [])),
                                    "valid_targets": len(data.get("valid_targets", []))
                                })
                    except Exception as e:
                        logging.warning(f"Failed to process raycast file {raycast_file}: {e}")
                
                if raycast_stats:
                    # Create a DataFrame from the stats
                    raycast_df = pd.DataFrame(raycast_stats)
                    
                    # Plot raycast success rate by episode
                    plt.figure(figsize=(10, 6))
                    plt.bar(raycast_df["episode_id"], raycast_df["success_rate"] * 100)
                    plt.title("Raycast Success Rate by Episode")
                    plt.xlabel("Episode ID")
                    plt.ylabel("Success Rate (%)")
                    plt.ylim(0, 100)
                    plt.grid(True, axis='y')
                    plt.savefig(os.path.join(output_dir, "raycast_success_rate.png"))
                    plt.close()
                    
                    # Plot valid targets found by episode
                    plt.figure(figsize=(10, 6))
                    plt.bar(raycast_df["episode_id"], raycast_df["valid_targets"], color='green')
                    plt.title("Valid Targets Found by Episode")
                    plt.xlabel("Episode ID")
                    plt.ylabel("Valid Target Count")
                    plt.grid(True, axis='y')
                    plt.savefig(os.path.join(output_dir, "valid_targets_count.png"))
                    plt.close()
        
        logging.info(f"Created metrics plots in {output_dir}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating plots: {e}")
        traceback.print_exc()
        return False

def main():
    """Main entry point with experiment configuration"""
    print("\n==== Autonomous Drone Navigation Experiment with Active Inference ====")
    
    # Define experiment configuration (could be loaded from a file)
    config = DEFAULT_CONFIG.copy()
    
    # Display configuration
    print("\nExperiment configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run the experiment
    run_experiment(config)

if __name__ == "__main__":
    main()








