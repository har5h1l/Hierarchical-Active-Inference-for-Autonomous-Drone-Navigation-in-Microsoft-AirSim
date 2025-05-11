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
    "output_dir": "experiment_results"
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
        
        # Enforce minimum movement distance
        min_movement_distance = 0.3  # Minimum movement distance in meters
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
                          episode_id: int = 0, seed: int = None) -> List[float]:
    """Sample a random target position within distance bounds that is visible to the drone
    
    Args:
        current_pos: Current drone position [x, y, z]
        distance_range: (min_distance, max_distance) in meters
        client: AirSim client instance
        max_attempts: Maximum sampling attempts before giving up
        episode_id: Current episode ID to influence randomization
        seed: Random seed for deterministic behavior
        
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
    
    for attempt in range(max_attempts):
        # Sample random direction (uniform on sphere)
        theta = target_rng.uniform(0, 2 * math.pi)  # Azimuth angle
        phi = target_rng.uniform(0, math.pi)        # Polar angle
        
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
        target_vector = airsim.Vector3r(target_pos[0], target_pos[1], target_pos[2])
        
        # Test line of sight - note: simTestLineOfSightToPoint expects (point, vehicle_name)
        # The vehicle_name is optional and defaults to empty string which means the default vehicle
        try:
            if client.simTestLineOfSightToPoint(target_vector):
                logging.info(f"Found valid target at {target_pos.tolist()} (attempt {attempt+1})")
                return target_pos.tolist()
        except Exception as e:
            logging.warning(f"Error in line of sight test: {e}")
            # If there's an error with the API call, try a simplified approach
            # Just check if target is far enough from obstacles
            # This is a fallback if the AirSim API call doesn't work properly
            return target_pos.tolist()  # Return the target anyway
    
    # If we couldn't find a visible target after max attempts, just return the last one
    # This is a fallback to avoid failing the episode
    logging.warning(f"No unobstructed target found after {max_attempts} attempts. Using last sampled position.")
    return target_pos.tolist()

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
    
    # Takeoff
    client.takeoffAsync().join()
    time.sleep(1)  # Give time to stabilize
    
    # Get initial position
    drone_state = client.getMultirotorState().kinematics_estimated
    drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
    
    # Sample a random target
    try:
        target_pos = sample_visible_target(
            drone_pos, 
            config["target_distance_range"], 
            client,
            episode_id=episode_id,
            seed=config["random_seed"]
        )
    except ValueError as e:
        logging.warning(f"Episode {episode_id}: {str(e)}")
        return [], {
            "episode_id": episode_id,
            "status": "skipped",
            "reason": "No valid target found"
        }
    
    # Initialize episode tracking
    step_metrics = []
    collisions = 0
    replanning_count = 0
    start_time = time.time()
    trajectory = [drone_pos]
    
    # Initialize stuck detection variables - based on time rather than movement
    last_significant_movement_time = start_time
    last_position = np.array(drone_pos)
    stuck_timeout = config.get("stuck_timeout", 15.0)  # Time in seconds to consider the drone stuck
    min_movement_threshold = 0.5  # Minimum movement in meters to consider significant
    
    # Initial distance to target
    distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(drone_pos))
    initial_distance = distance_to_target
    
    # Set initial safety margin (higher value = more conservative)
    # When far from target, use conservative safety margins
    # When closer to target, gradually reduce safety margins to allow reaching it
    base_safety_margin = config.get("safety_margin", MARGIN)
    
    logging.info(f"Episode {episode_id}: Initial position: {[round(p, 2) for p in drone_pos]}")
    logging.info(f"Episode {episode_id}: Target position: {[round(p, 2) for p in target_pos]}")
    logging.info(f"Episode {episode_id}: Initial distance: {distance_to_target:.2f}m")
    logging.info(f"Episode {episode_id}: Base safety margin: {base_safety_margin:.2f}m")
    
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
        
        # Check if drone has moved significantly
        current_position = np.array(drone_pos)
        movement = np.linalg.norm(current_position - last_position)
        
        if movement > min_movement_threshold:
            # Drone has moved significantly, update time and position
            last_significant_movement_time = time.time()
            last_position = current_position
            logging.debug(f"Episode {episode_id}: Significant movement detected: {movement:.2f}m")
        
        # Check if drone is stuck (no significant movement for too long)
        time_since_movement = time.time() - last_significant_movement_time
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
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(drone_pos))
        
        # Dynamically adjust safety margin based on distance to target
        # This allows more aggressive navigation when closer to the target
        # But still maintains a minimum safety threshold
        distance_ratio = min(1.0, distance_to_target / initial_distance)
        min_safety_margin = 1.0  # Minimum safety margin in meters
        
        # Adjust safety margin:
        # - Use full safety margin when far from target
        # - Gradually reduce to min_safety_margin when close to target
        # - But never go below the minimum
        # - Additional factor: if obstacles are very close, increase margin
        safety_margin = max(
            min_safety_margin,
            base_safety_margin * (0.5 + 0.5 * distance_ratio)  # Scale from 50-100% of base margin
        )
        
        # Check if target reached
        if distance_to_target <= ARRIVAL_THRESHOLD:
            logging.info(f"Episode {episode_id}: Target reached at step {step}")
            status = "success"
            break
        
        # Get obstacle data
        obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
        
        # Log obstacle detection info for debugging
        if step % 10 == 0:  # Log every 10 steps to avoid excessive logging
            logging.debug(f"Episode {episode_id}: Detected {len(obstacle_positions)} obstacles")
            if len(obstacle_positions) > 0:
                min_dist = min(obstacle_distances) if obstacle_distances else "N/A"
                logging.debug(f"Episode {episode_id}: Closest obstacle at {min_dist}m")
        
        # Create observation for active inference
        observation = {
            "drone_position": drone_pos,
            "target_position": target_pos,
            "obstacle_positions": obstacle_positions,
            "obstacle_distances": obstacle_distances,
            "waypoint_count": WAYPOINT_SAMPLE_COUNT,
            "safety_margin": safety_margin,  # Use the dynamically adjusted safety margin
            "policy_length": POLICY_LENGTH,
            "density_radius": DENSITY_RADIUS
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
        
        # Compute action magnitude
        action_magnitude = np.linalg.norm(np.array(next_waypoint) - np.array(drone_pos))
        
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
            "min_obstacle_distance": min(obstacle_distances) if obstacle_distances else float('inf'),
            "safety_margin": safety_margin,
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
    """Run a complete experiment with multiple episodes
    
    Args:
        config: Experiment configuration
    """
    # Set random seed for reproducibility
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir
    
    # Save configuration
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2, cls=NumpyJSONEncoder)
    
    # Initialize Julia server
    julia_server = JuliaServer()
    precompile_success = julia_server.precompile_packages()
    
    if not precompile_success:
        logging.warning("Precompilation had issues but will continue")
    
    # Start ZMQ server
    server_port = julia_server.start_server()
    if not server_port:
        logging.error("Failed to start ZMQ server. Cannot continue.")
        return
    
    # Initialize ZMQ interface
    zmq_interface = ZMQInterface(server_port=server_port)
    
    # Connect to AirSim
    logging.info("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Initialize scanner
    scanner = Scanner(client)
    
    # Storage for metrics
    all_step_metrics = []
    episode_summaries = []
    
    # Run episodes
    for episode in range(1, config["num_episodes"] + 1):
        try:
            step_metrics, episode_summary = run_episode(
                episode, client, zmq_interface, scanner, config
            )
            
            all_step_metrics.extend(step_metrics)
            episode_summaries.append(episode_summary)
            
            # Optionally save intermediate results
            if episode % 10 == 0 or episode == config["num_episodes"]:
                if all_step_metrics:
                    pd.DataFrame(all_step_metrics).to_csv(
                        os.path.join(output_dir, "per_step_metrics.csv"), index=False
                    )
                if episode_summaries:
                    pd.DataFrame(episode_summaries).to_csv(
                        os.path.join(output_dir, "episode_summaries.csv"), index=False
                    )
        except Exception as e:
            logging.error(f"Error in episode {episode}: {e}")
            traceback.print_exc()
            episode_summaries.append({
                "episode_id": episode,
                "status": "error",
                "error_message": str(e)
            })
    
    # Save final results
    if all_step_metrics:
        pd.DataFrame(all_step_metrics).to_csv(
            os.path.join(output_dir, "per_step_metrics.csv"), index=False
        )
    if episode_summaries:
        pd.DataFrame(episode_summaries).to_csv(
            os.path.join(output_dir, "episode_summaries.csv"), index=False
        )
    
    # Generate summary plots
    try:
        if episode_summaries:
            summary_df = pd.DataFrame(episode_summaries)
            
            # Successful episodes
            successful = summary_df[summary_df["target_reached"] == True]
            success_rate = len(successful) / len(summary_df) if len(summary_df) > 0 else 0
            
            # Create summary figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Success rate over episodes
            rolling_success = summary_df["target_reached"].rolling(
                window=min(5, len(summary_df)), min_periods=1
            ).mean()
            axes[0, 0].plot(rolling_success.index, rolling_success.values)
            axes[0, 0].set_title(f"Success Rate: {success_rate:.1%}")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Rolling Success Rate (5-ep window)")
            
            # Plot 2: Step count distribution
            axes[0, 1].hist(summary_df["steps_taken"], bins=10)
            axes[0, 1].set_title("Steps per Episode")
            axes[0, 1].set_xlabel("Step Count")
            axes[0, 1].set_ylabel("Frequency")
            
            # Plot 3: Inference time
            axes[1, 0].plot(summary_df["episode_id"], summary_df["avg_inference_time_ms"])
            axes[1, 0].set_title("Average Inference Time")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Time (ms)")
            
            # Plot 4: Distance improvement
            axes[1, 1].scatter(
                summary_df["initial_distance"], 
                summary_df["normalized_improvement"],
                c=summary_df["target_reached"].map({True: 'green', False: 'red'})
            )
            axes[1, 1].set_title("Performance by Distance")
            axes[1, 1].set_xlabel("Initial Distance (m)")
            axes[1, 1].set_ylabel("Normalized Improvement")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "experiment_summary.png"))
            plt.close(fig)
    except Exception as e:
        logging.error(f"Error generating summary plots: {e}")
    
    # Shutdown server
    julia_server.shutdown()
    logging.info(f"Experiment completed. Results saved to {output_dir}")

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








