import airsim
import numpy as np
import time
import os
import json
import subprocess
import sys
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

# Add the airsim directory to the Python path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'airsim'))

# Import the scanner with the correct filename
try:
    from Sensory_Input_Processing import EnvironmentScanner
except ImportError:
    # Try with the original filename if renamed file not found
    from airsim.EnvironmentScanner import EnvironmentScanner


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
            
            # Start movement asynchronously (don't join yet)
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
            while not movement_task.is_done():
                # Check current position
                self.update_drone_state()
                
                # Check for obstacles
                if self.check_for_obstacles(safety_threshold=2.5):
                    print("🛑 Stopping movement due to obstacle detection!")
                    self.client.cancelLastTask()  # Cancel the movement task
                    self.client.hoverAsync().join()  # Hover in place
                    obstacle_detected = True
                    break
                
                # Short sleep to avoid CPU overload
                time.sleep(0.1)
            
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
                    while not precision_task.is_done():
                        # Check for obstacles
                        if self.check_for_obstacles(safety_threshold=2.0):
                            print("🛑 Stopping precision movement due to obstacle detection!")
                            self.client.cancelLastTask()
                            self.client.hoverAsync().join()
                            return "obstacle_detected"
                        
                        time.sleep(0.1)
                    
                    self.update_drone_state()
            else:
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
    
    # Ensure interface directory exists
    os.makedirs(INTERFACE_DIR, exist_ok=True)
    
    # Check if Julia scripts exist, otherwise use dummy actions
    use_julia = os.path.exists("./run_inference.jl") and os.path.exists("./run_planning.jl")
    
    controller = DroneController()
    controller.reset_and_takeoff()
    
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
        
        # 2. Save observation to JSON for Julia
        print("📤 Saving observation data...")
        controller.save_observation_to_json(observation)
        
        # 3. Run Julia inference and planning (or simulate it)
        print("🧠 Running inference and planning...")
        if use_julia:
            controller.run_julia_inference_and_planning()
        else:
            print("Julia scripts not found, using dummy action generator")
            controller.create_dummy_action_output()
        
        # 4. Load the next waypoint from Julia's output
        print("📬 Loading next waypoint...")
        egocentric_waypoint, policy = controller.load_action_from_json()
        
        # 5. Convert to global coordinates and move the drone
        print("🌐 Converting to global coordinates...")
        global_waypoint = controller.convert_to_global_waypoint(egocentric_waypoint)
        
        print("🚁 Moving to waypoint...")
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
    
    # Land the drone
    print("Landing drone...")
    controller.client.landAsync().join()
    controller.client.armDisarm(False)
    controller.client.enableApiControl(False)
    print("Test completed.")


if __name__ == "__main__":
    main()
