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
                return [], []
            
            # Convert point cloud to positions
            try:
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error reshaping point cloud: {str(e)}")
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
            return [], []

# Initialize scanner
scanner = Scanner(client)

class ZMQInterface:
    """Interface for communicating with the Julia Active Inference server via ZMQ"""
    
    def __init__(self, server_address=ZMQ_SERVER_ADDRESS):
        """Initialize ZMQ connection to Julia server
        
        Args:
            server_address: ZMQ server address (default: ZMQ_SERVER_ADDRESS)
        """
        self.server_address = server_address
        self._setup_zmq_connection()
        
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
            self.socket.setsockopt(zmq.LINGER, 100)  # Don't wait long for unsent messages on close
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second receive timeout
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second send timeout
            
            # Connect with proper error handling
            try:
                self.socket.connect(self.server_address)
                print(f"Connected to ZMQ server at {self.server_address}")
                
                # Test connection with ping
                self._test_connection()
                
                return True
            except zmq.ZMQError as e:
                print(f"Failed to connect to ZMQ server: {str(e)}")
                if self.socket:
                    self.socket.close()
                    self.socket = None
                return False
        
        except Exception as e:
            print(f"Failed to setup ZMQ connection: {str(e)}")
            self.socket = None
            self.context = None
            return False
    
    def _test_connection(self):
        """Test the ZMQ connection with a ping-pong exchange"""
        try:
            # Store current timeout and set shorter timeout for test
            current_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout for ping test
            
            # Send ping request
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
                return False
                
        except zmq.ZMQError as e:
            print(f"⚠️ ZMQ connection test failed: {str(e)}")
            
            # Reset timeout to original value in case of error
            try:
                if self.socket:
                    self.socket.setsockopt(zmq.RCVTIMEO, current_timeout)
            except:
                pass
                
            return False
    
    def _reset_socket(self):
        """Reset the ZMQ socket if it becomes unresponsive"""
        print("Resetting ZMQ connection...")
        try:
            # Close the current socket and context
            if self.socket:
                self.socket.close()
                self.socket = None
            if self.context:
                self.context.term()
                self.context = None
            
            # Wait a moment before reconnecting
            time.sleep(0.5)
            
            # Recreate socket
            success = self._setup_zmq_connection()
            
            if success:
                print("✅ ZMQ connection reset successfully")
            else:
                print("❌ Failed to reset ZMQ connection")
                
            return success
        except Exception as e:
            print(f"Error resetting ZMQ socket: {str(e)}")
            traceback.print_exc()
            return False
    
    def _is_server_running(self):
        """Check if the ZMQ server is running and responsive"""
        if not self.socket:
            return False
            
        try:
            # Set shorter timeout for ping test
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)
            self.socket.send_string("ping")
            response = self.socket.recv_string()
            # Reset timeout to normal
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)
            return response == "pong"
        except Exception:
            return False
    
    def _start_server(self):
        """Start the Julia ZMQ server (implementation depends on environment)"""
        # This is a placeholder - actual implementation would depend on how
        # the Julia server is set up in your environment
        print("Automatic server startup not implemented")
        return False
    
    def _diagnose_zmq_server_issues(self):
        """Run diagnostics on ZMQ server connection issues"""
        print("\n==== ZMQ SERVER DIAGNOSTICS ====")
        
        # Check socket status
        if not self.socket:
            print("⚠️ ZMQ socket is not initialized")
        
        # Check if Julia process is running
        julia_running = False
        try:
            # This command would need to be adapted based on your OS
            if platform.system() == "Windows":
                result = subprocess.run(["tasklist"], capture_output=True, text=True)
                if "julia" in result.stdout.lower():
                    print("✅ Julia process found running")
                    julia_running = True
                else:
                    print("❌ No Julia process found running")
            else:  # Linux/Mac
                result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
                if "julia" in result.stdout:
                    print("✅ Julia process found running")
                    julia_running = True
                else:
                    print("❌ No Julia process found running")
        except Exception as e:
            print(f"⚠️ Could not check for Julia process: {str(e)}")
        
        # Try to ping the server
        ping_success = self._is_server_running()
        if ping_success:
            print("✅ Successfully pinged ZMQ server")
        else:
            print("❌ Failed to ping ZMQ server")
        
        # Try socket reset if issues detected
        if not ping_success:
            print("Attempting to reset ZMQ socket...")
            if self._reset_socket():
                print("✅ Socket reset completed")
                # Try ping again
                if self._is_server_running():
                    print("✅ Server responding after socket reset")
                else:
                    print("❌ Server still not responding after socket reset")
            else:
                print("❌ Socket reset failed")
        
        print("================================\n")
        
        return julia_running and ping_success
    
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
        """
        # Check if ZMQ interface is initialized
        if not self.socket:
            print("⚠️ ZMQ socket not initialized, attempting to connect")
            if not self._setup_zmq_connection():
                print("❌ Failed to initialize ZMQ socket")
                return None, None
        
        # Maximum number of retry attempts
        max_retries = 2
        current_retry = 0
        
        while current_retry <= max_retries:
            try:
                # Add safety radius based on waypoint_radius if present
                if 'waypoint_radius' in observation:
                    radius = observation['waypoint_radius']
                    # Ensure radius is a valid float
                    if not isinstance(radius, (int, float)) or radius <= 0:
                        print(f"⚠️ Invalid waypoint_radius value: {radius}, using default")
                        radius = 5.0
                        observation['waypoint_radius'] = radius
                else:
                    # Default radius if not specified
                    radius = 5.0
                    observation['waypoint_radius'] = radius
                
                # Fully sanitize the observation using recursive method
                try:
                    sanitized_observation = self._sanitize_for_json(observation)
                except Exception as e:
                    print(f"⚠️ Error during observation sanitization: {str(e)}")
                    print("Falling back to simplified sanitization method")
                    
                    # Fallback sanitization (simpler but less thorough)
                    sanitized_observation = {}
                    for key, value in observation.items():
                        if isinstance(value, np.ndarray):
                            sanitized_observation[key] = value.tolist()
                        elif hasattr(value, 'dtype'): # Any NumPy type
                            sanitized_observation[key] = value.item() if hasattr(value, 'item') else float(value)
                        elif isinstance(value, list):
                            # Handle lists with basic conversion
                            sanitized_list = []
                            for item in value:
                                if isinstance(item, np.ndarray):
                                    sanitized_list.append(item.tolist())
                                elif hasattr(item, 'dtype'):
                                    sanitized_list.append(item.item() if hasattr(item, 'item') else float(item))
                                else:
                                    sanitized_list.append(item)
                            sanitized_observation[key] = sanitized_list
                        else:
                            sanitized_observation[key] = value
                
                # Verify the sanitized observation
                for key, value in sanitized_observation.items():
                    if hasattr(value, 'dtype'):
                        print(f"⚠️ WARNING: Failed to sanitize '{key}' with type {type(value)}")
                        sanitized_observation[key] = str(value)  # Last resort
                
                # Convert observation to JSON string
                try:
                    observation_json = json.dumps(sanitized_observation, cls=NumpyJSONEncoder)
                except TypeError as e:
                    print(f"⚠️ JSON serialization error: {str(e)}")
                    # Additional logging to help diagnose serialization issues
                    problematic_keys = []
                    for key, value in sanitized_observation.items():
                        try:
                            json.dumps({key: value}, cls=NumpyJSONEncoder)
                        except TypeError:
                            problematic_keys.append(key)
                            sanitized_observation[key] = str(value)  # Replace with string representation
                    
                    print(f"Problematic keys: {problematic_keys}")
                    observation_json = json.dumps(sanitized_observation, cls=NumpyJSONEncoder)  # Try again after fixing
                
                # Send observation to server
                print(f"Sending request to ZMQ server (attempt {current_retry+1}/{max_retries+1})...")
                self.socket.send_string(observation_json)
                
                # Receive action from server
                print("Waiting for response...")
                response_json = self.socket.recv_string()
                print("Received response from ZMQ server")
                    
                # Parse response
                try:
                    response = json.loads(response_json)
                    
                    # Extract next waypoint and policy from response
                    if "action" in response and response["action"] is not None:
                        next_waypoint = response["action"]
                        
                        # Validate waypoint format
                        if not isinstance(next_waypoint, list) or len(next_waypoint) != 3:
                            print(f"⚠️ Invalid waypoint format: {next_waypoint}")
                            return None, None
                        
                        # Ensure all waypoint values are numeric
                        try:
                            next_waypoint = [float(val) for val in next_waypoint]
                        except (ValueError, TypeError):
                            print(f"⚠️ Non-numeric values in waypoint: {next_waypoint}")
                            return None, None
                        
                        # Get policy if available, or use default
                        if "policy" in response and isinstance(response["policy"], list):
                            policy = response["policy"]
                            
                            # Validate policy format
                            if len(policy) == 0 or not all(isinstance(p, list) for p in policy):
                                print(f"⚠️ Invalid policy format, using default")
                                policy = [next_waypoint] * 3
                        else:
                            policy = [next_waypoint] * 3  # Default to repeat action 3 times
                        
                        # Apply safety limit to waypoint distance
                        # Dynamically adjust maximum step size based on waypoint radius
                        # Allow larger steps in areas with high suitability (higher radius)
                        max_magnitude = min(radius * 1.2, 15.0)  # Allow up to 15m steps for very clear areas
                        magnitude = sqrt(sum(x*x for x in next_waypoint))
                        
                        if magnitude > max_magnitude and magnitude > 0:
                            # Scale down to maximum allowed magnitude
                            scaling_factor = max_magnitude / magnitude
                            next_waypoint = [x * scaling_factor for x in next_waypoint]
                            print(f"⚠️ Waypoint distance {magnitude:.2f}m exceeds maximum {max_magnitude:.2f}m - scaled down")
                        
                        return next_waypoint, policy
                    else:
                        print("⚠️ No valid action field in response from ZMQ server")
                        
                        # Try to diagnose server issues
                        self._diagnose_zmq_server_issues()
                        
                        # Increment retry counter
                        current_retry += 1
                        if current_retry <= max_retries:
                            print(f"Will retry communication ({current_retry}/{max_retries})...")
                            # Reset socket before retrying
                            self._reset_socket()
                            time.sleep(1)  # Wait before retry
                            continue
                        else:
                            return None, None
                        
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON response from ZMQ server: {response_json[:100]}...")
                    # Increment retry counter
                    current_retry += 1
                    if current_retry <= max_retries:
                        print(f"Will retry communication ({current_retry}/{max_retries})...")
                        # Reset socket before retrying
                        self._reset_socket()
                        time.sleep(1)  # Wait before retry
                        continue
                    else:
                        return None, None
                    
            except zmq.ZMQError as e:
                print(f"⚠️ ZMQ communication error: {str(e)}")
                
                # Increment retry counter
                current_retry += 1
                
                # Attempt to reset connection if we have retries left
                if current_retry <= max_retries:
                    print(f"Resetting ZMQ connection for retry {current_retry}/{max_retries}...")
                    self._reset_socket()
                    time.sleep(1)  # Wait a moment before retrying
                    continue
                else:
                    print("Exceeded maximum retry attempts")
                    return None, None
                    
            except socket.timeout as e:
                print(f"⚠️ Socket timeout: {str(e)}")
                # Increment retry counter
                current_retry += 1
                if current_retry <= max_retries:
                    print(f"Will retry after timeout ({current_retry}/{max_retries})...")
                    # Reset socket before retrying
                    self._reset_socket()
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    return None, None
                    
            except Exception as e:
                print(f"⚠️ Unexpected error in ZMQ communication: {str(e)}")
                traceback.print_exc()
                # Increment retry counter
                current_retry += 1
                if current_retry <= max_retries:
                    print(f"Will retry after error ({current_retry}/{max_retries})...")
                    # Reset socket before retrying
                    self._reset_socket()
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    return None, None
    
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
        
        # Test scanner to validate it's working
        try:
            obstacle_positions, obstacle_distances = self.scanner.fetch_density_distances()
            print(f"Scanner test: Found {len(obstacle_positions)} obstacles")
        except Exception as e:
            print(f"Warning: Scanner test failed: {str(e)}")
            print("Reinitializing with basic scanner")
            self.scanner = Scanner(self.client)
        
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
            obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
            
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
            obstacle_positions, obstacle_distances = self.scanner.fetch_density_distances()
            
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
            forward_vector = forward_vector / np.linalg.norm(forward_vector)
            
            # Debug: Print obstacle information
            print(f"🔍 OBSTACLE CHECK: Found {len(obstacle_distances)} objects within sensing range")
            
            # Enhanced obstacle detection that prioritizes obstacles in front of the drone
            closest_obstacle_distance = 100.0
            closest_frontal_distance = 100.0
            
            for i, (position, distance) in enumerate(zip(obstacle_positions, obstacle_distances)):
                # Convert position to numpy array for vector operations
                obstacle_vector = np.array(position) - np.array(self.current_position)
                
                # Normalize obstacle vector
                if np.linalg.norm(obstacle_vector) > 0:
                    obstacle_vector = obstacle_vector / np.linalg.norm(obstacle_vector)
                
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
        """Convert an egocentric waypoint to global coordinates with enhanced orientation handling
        
        The egocentric waypoint is relative to the drone's current position and orientation.
        This method transforms it to global coordinates using quaternion rotation.
        """
        # Ensure drone state is updated with latest orientation
        self.update_drone_state()
        
        # Ensure egocentric_waypoint contains only Python native types, not NumPy types
        egocentric_waypoint = [float(v) for v in egocentric_waypoint]
        
        # Validate orientation
        if self.current_orientation is None or any(np.isnan(q) or np.isinf(q) for q in self.current_orientation):
            print("⚠️ Invalid orientation detected, using default [1,0,0,0]")
            self.current_orientation = [1.0, 0.0, 0.0, 0.0]  # Default identity quaternion
        
        # Validate egocentric waypoint
        if not all(isinstance(w, (int, float)) and np.isfinite(w) for w in egocentric_waypoint):
            print(f"⚠️ Invalid egocentric waypoint {egocentric_waypoint}, using [1,0,0]")
            egocentric_waypoint = [1.0, 0.0, 0.0]  # Default forward movement
        
        # Print initial information for debugging
        print("\n==== COORDINATE TRANSFORMATION ====")
        print(f"Current position: {[round(p, 2) for p in self.current_position]}")
        print(f"Egocentric waypoint: {[round(w, 2) for w in egocentric_waypoint]}")
        
        # Calculate distance to target to adjust step size
        distance_to_target = self.distance_to_target()
        print(f"Distance to target: {distance_to_target:.2f}m")
        
        # Store original waypoint for comparison
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
        
        # Print orientation for debugging
        w, x, y, z = self.current_orientation
        print(f"Drone orientation (quaternion): [{w:.3f}, {x:.3f}, {y:.3f}, {z:.3f}]")
        
        # Normalize quaternion to ensure it's valid for rotation
        quat_magnitude = math.sqrt(w*w + x*x + y*y + z*z)
        if abs(quat_magnitude - 1.0) > 0.01:  # Check if magnitude is not close to 1
            print(f"⚠️ Non-unit quaternion detected (magnitude: {quat_magnitude:.3f}), normalizing")
            self.current_orientation = [
                w / quat_magnitude,
                x / quat_magnitude,
                y / quat_magnitude,
                z / quat_magnitude
            ]
            w, x, y, z = self.current_orientation
        
        # IMPORTANT FIX: Create transformation matrix directly from quaternion
        # This creates a proper rotation matrix from the quaternion
        rotation_matrix = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Convert egocentric vector to numpy array
        ego_vector = np.array(egocentric_waypoint)
        
        # Apply rotation using matrix multiplication
        # This correctly transforms from drone's local frame to global frame
        rotated_vector = rotation_matrix @ ego_vector
        
        # Verify the result is valid
        if np.isnan(rotated_vector).any() or np.isinf(rotated_vector).any():
            print("⚠️ Invalid rotation result, using simple vector addition")
            rotated_vector = np.array(egocentric_waypoint)
        
        # Add the rotated vector to current position to get global waypoint
        global_waypoint = [
            self.current_position[0] + float(rotated_vector[0]),
            self.current_position[1] + float(rotated_vector[1]),
            self.current_position[2] + float(rotated_vector[2])
        ]
        
        # Print final waypoint details
        print(f"Global waypoint: {[round(w, 2) for w in global_waypoint]}")
        
        return global_waypoint

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

# Main function
def main():
    try:
        # Create controller
        controller = DroneController()
        
        # Precompile Julia components
        print("Precompiling Julia components...")
        if not controller.precompile_julia_components():
            print("Warning: Precompilation had issues but continuing anyway")
        
        # Reset AirSim, take off, and get starting position
        print("Resetting AirSim and taking off...")
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
        
        # Navigate to target
        success = controller.navigate_to_target()
        
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

