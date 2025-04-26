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
TARGET_LOCATION = [10.0, 0.0, -3.0]  # [x, y, z] in NED coordinates
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
            
            # Start the server process
            import subprocess
            cmd = ["julia", "--project=.", server_script]
            print(f"Starting server with command: {' '.join(cmd)}")
            
            # Use Popen for a non-blocking call, and redirect output to prevent blocking
            self._server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start up (check for status file)
            start_time = time.time()
            while not self._is_server_running() and time.time() - start_time < 10:
                time.sleep(0.5)
                
            if self._is_server_running():
                print("✅ ZMQ server started successfully")
                # Wait a bit more to ensure the socket is bound
                time.sleep(1.5)
                return True
            else:
                print("❌ Failed to start ZMQ server")
                return False
                
        except Exception as e:
            print(f"❌ Error starting ZMQ server: {str(e)}")
            return False
    
    def _cleanup_socket(self):
        """Clean up ZMQ socket and context properly."""
        try:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
                print("Socket closed")
                
            # Only terminate context if we're creating a new one
            if self._context is not None:
                self._context.term()
                self._context = None
                print("ZMQ context terminated")
                
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
        
        # Create new context and socket with simpler settings
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        
        # Set minimal socket options for better reliability
        self._socket.setsockopt(zmq.LINGER, 0)  # Don't wait for unsent messages
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        
        # Reconnect
        try:
            self._socket.connect(self.server_address)
            self._connected = True
            print("✅ Socket reset and reconnected")
            
            # Verify connection with a simple ping
            if self._simple_ping():
                print("✅ Connection verified with ping")
                return True
            else:
                print("❌ Ping failed after reset")
                self._connected = False
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
            return False
            
        # Use a very short timeout for ping
        try:
            # Save original timeout
            old_timeout = self._socket.getsockopt(zmq.RCVTIMEO)
            self._socket.setsockopt(zmq.RCVTIMEO, 2000)
            
            # Send ping
            try:
                self._socket.send_string("ping", zmq.NOBLOCK)
                response = self._socket.recv_string()
                success = response == "pong"
            except Exception:
                success = False
            
            # Restore original timeout
            self._socket.setsockopt(zmq.RCVTIMEO, old_timeout)
            return success
        except:
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
            if not self._setup_connection():
                print("❌ Not connected to ZMQ server and reconnection failed")
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
        obs_json = json.dumps(observation)
        msg_size = len(obs_json) / 1024  # Size in KB
        
        # Try to send observation and receive response
        for attempt in range(self.max_retries):
            try:
                # Send observation
                print(f"📤 Sending observation data ({round(msg_size, 1)} KB)...")
                start_time = time.time()  # Start timing the request
                self._socket.send_string(obs_json)
                
                # Receive response
                print(f"⏳ Waiting for response (timeout: {self.timeout/1000}s)...")
                response_json = self._socket.recv_string()
                
                # Calculate response time
                elapsed = time.time() - start_time
                
                resp_size = len(response_json) / 1024  # Size in KB
                print(f"📥 Received response ({round(resp_size, 1)} KB) in {round(elapsed, 2)}s")
                
                # Parse response
                response = json.loads(response_json)
                
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
                
                if attempt < self.max_retries - 1:
                    print("🔄 Retrying with fresh connection...")
                    self._reset_socket()
                    time.sleep(1)
        
        # If all attempts failed
        print("❌ All attempts to communicate with the server failed")
        return default_state, default_action
    
    def close(self):
        """Close the ZMQ connection."""
        try:
            self._cleanup_socket()
            print("✅ ZMQ connection closed")
            
            # Don't terminate the server process when closing the client
            # This allows multiple clients to use the same server instance
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
            precompile_script = os.path.normpath("./precompile.jl")
            print(f"Running Julia precompilation: {precompile_script}")
            result = subprocess.run(
                ["julia", "--project=.", precompile_script],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("✅ Julia precompilation completed successfully")
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
            num_obstacles, distances = self.scanner.fetch_density_distances()
            
            # Get voxel coordinates (obstacle map)
            points = self.scanner.collect_sensor_data()
            if points is None:
                voxel_grid = []
            else:
                voxel_grid = self.scanner.create_obstacle_voxel_grid(points)
                if voxel_grid is None:
                    voxel_grid = []
                else:
                    # Convert numpy array to list for JSON serialization
                    voxel_grid = voxel_grid.tolist()
            
            # Calculate obstacle density (number of obstacles within radius)
            density = num_obstacles / (4/3 * np.pi * DENSITY_RADIUS**3) if num_obstacles > 0 else 0
            
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
        print(f"Moving drone from {self.current_position} to {waypoint}")
        
        # Ensure waypoint is within simulation boundaries and round coordinates
        waypoint = [
            round(min(max(waypoint[0], -100), 100), 2),  # X between -100 and 100
            round(min(max(waypoint[1], -100), 100), 2),  # Y between -100 and 100
            round(min(max(waypoint[2], -20), 0), 2)      # Z between -20 and 0 (remember NED)
        ]
        
        try:
            print("Starting movement...")
            
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
                    print("🛑 Stopping movement due to obstacle detection!")
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
                print("Re-planning path due to obstacle...")
                return "obstacle_detected"
            
            # Verify movement accuracy
            final_distance = sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, waypoint)))
            
            if final_distance > 0.5:  # More than 0.5m away
                # Attempt final precision adjustment if close enough
                if final_distance < 2.0:
                    print("Making final precision adjustment...")
                    
                    # Start precision movement
                    precision_task = self.client.moveToPositionAsync(
                        x=waypoint[0],
                        y=waypoint[1],
                        z=waypoint[2],
                        velocity=1,  # Slower for precision
                        timeout_sec=10,
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
                        current_distance_to_waypoint = sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, waypoint)))
                        distance_moved = sqrt(sum((a - b) ** 2 for a, b in zip(self.current_position, previous_position)))
                        previous_position = self.current_position.copy()
                        
                        # Check if movement is complete
                        if current_distance_to_waypoint < 0.3:  # Closer threshold for precision
                            movement_complete = True
                        elif time.time() - start_time > 10:  # Timeout after 10 seconds
                            print("Precision movement timed out")
                            movement_complete = True
                        elif distance_moved < 0.005 and time.time() - start_time > 2:  # Stopped moving (smaller threshold)
                            movement_complete = True
                        
                        # Check for obstacles
                        if self.check_for_obstacles(safety_threshold=2.0):
                            print("🛑 Stopping precision movement due to obstacle detection!")
                            self.client.cancelLastTask()
                            self.client.hoverAsync().join()
                            return "obstacle_detected"
                    
                    # Ensure final stabilization
                    self.client.hoverAsync().join()
                    self.update_drone_state()
            
            print(f"Successfully reached waypoint. Final position: {self.current_position}")
            return "success"
                
        except Exception as e:
            print(f"Error during movement: {str(e)}")
            # Try to stabilize the drone
            print("Attempting to stabilize drone...")
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
    
    def convert_to_global_waypoint(self, egocentric_waypoint):
        """Convert an egocentric waypoint to global coordinates
        
        The egocentric waypoint is relative to the drone's current position and orientation.
        We need to transform it to global coordinates.
        """
        # For simplicity, we just add the egocentric vector to current position
        # This works because egocentric waypoints are in drone's NED frame
        global_waypoint = [
            self.current_position[0] + egocentric_waypoint[0],
            self.current_position[1] + egocentric_waypoint[1],
            self.current_position[2] + egocentric_waypoint[2]
        ]
        
        # Ensure waypoint is within simulation boundaries
        global_waypoint = [
            min(max(global_waypoint[0], -100), 100),  # X between -100 and 100
            min(max(global_waypoint[1], -100), 100),  # Y between -100 and 100
            min(max(global_waypoint[2], -20), 0)      # Z between -20 and 0 (remember NED)
        ]
        
        return global_waypoint


def main():
    print("Starting autonomous drone navigation test...")
    print(f"Target location: {TARGET_LOCATION}")
    
    # Initialize drone controller
    controller = DroneController()
    
    # Initialize ZMQ Interface
    zmq_interface = None
    
    try:
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
            print(f"Distance to target: {distance_to_target:.2f} meters")
            if distance_to_target < ARRIVAL_THRESHOLD:
                print("🎯 Target reached! Test complete.")
                break
            
            # 1. Collect sensor data
            print("📡 Collecting sensor data...")
            observation = controller.collect_sensor_data()
            
            # 2. Send observation to Julia server via ZMQ and receive expected state and action
            print("🧠 Sending observation to ZMQ server for processing...")
            expected_state, egocentric_waypoint = zmq_interface.send_observation_and_receive_action(observation)
            
            # If ZMQ communication failed, use local fallback (create a minimal action toward target)
            if expected_state is None:
                print("⚠️ ZMQ communication failed, using local fallback action")
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
            print("🌐 Converting to global coordinates...")
            global_waypoint = controller.convert_to_global_waypoint(egocentric_waypoint)
            
            print(f"🚁 Moving to waypoint: {global_waypoint}")
            movement_result = controller.move_to_waypoint(global_waypoint)
            
            # Check if an obstacle was detected during movement
            if movement_result == "obstacle_detected":
                print("⚠️ Obstacle encountered! Replanning route...")
                # No need to increment iteration - we'll retry with a fresh plan
                continue
            elif movement_result == "error":
                print("⚠️ Error during movement. Replanning...")
                # Brief pause to stabilize before replanning
                time.sleep(1)
                continue
            
            iteration += 1
        
        if iteration >= MAX_ITERATIONS:
            print("⚠️ Maximum iterations reached without finding target.")
        
    finally:
        # Cleanup ZMQ connection if it was initialized
        if zmq_interface is not None:
            print("Closing ZMQ connection...")
            zmq_interface.close()
        
        # Land the drone
        print("Landing drone...")
        controller.client.landAsync().join()
        controller.client.armDisarm(False)
        controller.client.enableApiControl(False)
        print("Test completed.")


if __name__ == "__main__":
    main()
