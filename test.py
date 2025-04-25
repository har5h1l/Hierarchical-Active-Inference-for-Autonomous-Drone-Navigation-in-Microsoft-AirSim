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
INTERFACE_DIR = "./interface"  # Directory for JSON exchange with Julia
JULIA_PATH = "julia"  # Path to Julia executable

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
        
        filepath = os.path.join(INTERFACE_DIR, "obs_input.json")
        with open(filepath, 'w') as f:
            json.dump(observation, f, indent=2)
        print(f"Observation saved to {filepath}")
        return filepath
    
    def run_julia_inference_and_planning(self):
        """Execute Julia scripts for inference and planning"""
        try:
            # Run inference script
            inference_script = "./run_inference.jl"
            print(f"Running Julia inference: {inference_script}")
            subprocess.run([JULIA_PATH, inference_script], check=True)
            
            # Run planning script
            planning_script = "./run_planning.jl"
            print(f"Running Julia planning: {planning_script}")
            subprocess.run([JULIA_PATH, planning_script], check=True)
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running Julia scripts: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error running Julia scripts: {str(e)}")
            return False
    
    def load_action_from_json(self):
        """Load the action from the JSON file created by Julia"""
        try:
            filepath = os.path.join(INTERFACE_DIR, "action_output.json")
            with open(filepath, 'r') as f:
                action_data = json.load(f)
            
            # Extract the egocentric waypoint
            egocentric_waypoint = action_data.get("next_waypoint", [1.0, 0.0, 0.0])  # Default forward 1m if missing
            policy = action_data.get("policy", [egocentric_waypoint])
            
            print(f"Loaded egocentric waypoint: {egocentric_waypoint}")
            print(f"Full policy: {policy}")
            
            return egocentric_waypoint, policy
        except Exception as e:
            print(f"Error loading action from JSON: {str(e)}. Using default action.")
            return [1.0, 0.0, 0.0], [[1.0, 0.0, 0.0]]
    
    def convert_to_global_waypoint(self, egocentric_waypoint):
        """Convert egocentric waypoint to global AirSim coordinates"""
        # Simple conversion - just add to current position
        # In a more advanced implementation, this would account for drone orientation
        global_waypoint = [
            self.current_position[0] + egocentric_waypoint[0],
            self.current_position[1] + egocentric_waypoint[1],
            self.current_position[2] + egocentric_waypoint[2]
        ]
        print(f"Converted to global waypoint: {global_waypoint}")
        return global_waypoint
    
    def move_to_waypoint(self, waypoint):
        """Command the drone to move to the specified waypoint"""
        print(f"Moving drone from {self.current_position} to {waypoint}")
        
        # Ensure waypoint is within simulation boundaries
        waypoint = [
            min(max(waypoint[0], -100), 100),  # X between -100 and 100
            min(max(waypoint[1], -100), 100),  # Y between -100 and 100
            min(max(waypoint[2], -20), 0)      # Z between -20 and 0 (remember NED)
        ]
        
        self.client.moveToPositionAsync(
            waypoint[0], waypoint[1], waypoint[2], 
            velocity=5,  # 5 m/s
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False)
        ).join()
        
        # Wait briefly for stability
        time.sleep(1)
        
        # Update state after movement
        self.update_drone_state()
        print(f"Drone now at position: {self.current_position}")
    
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
        
        with open(os.path.join(INTERFACE_DIR, "action_output.json"), 'w') as f:
            json.dump(action, f, indent=2)


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
        controller.move_to_waypoint(global_waypoint)
        
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