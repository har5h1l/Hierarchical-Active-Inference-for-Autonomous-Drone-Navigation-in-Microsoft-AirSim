#!/usr/bin/env python
# Test script to verify improved obstacle avoidance with the updated planning algorithm

import os
import json
import time
import numpy as np
import sys
import logging
from datetime import datetime
from drone_utils import AirSimDrone, ObstacleDetector, monitor_drone_position
from data_collection import Scanner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_obstacle_avoidance():
    """
    Test script to verify the improved obstacle avoidance in the path planning algorithm.
    This runs a single episode with the fixed pathfinding logic.
    """
    print("\n===== TESTING IMPROVED OBSTACLE AVOIDANCE LOGIC =====\n")
    
    # Initialize drone
    drone = AirSimDrone()
    drone.connect()
    drone.reset()
      # Set up obstacle detection
    obstacle_detector = ObstacleDetector(drone)
    scanner = Scanner(
        client=drone.client,
        scan_range=10.0  # Scan range for obstacle detection
    )
    
    # Load target - use the problem target from the experiment logs
    target_position = np.array([-31.58, -13.87, -8.21])  # Same target as in previous failed tests
    print(f"Target position: {target_position}")
    
    # Set up initial drone state
    takeoff_height = 5.0  # meters
    print(f"Taking off to {takeoff_height}m...")
    drone.takeoff(takeoff_height)
    time.sleep(2)  # Wait for stabilization
    
    # Get initial position
    initial_position = drone.get_position()
    distance_to_target = np.linalg.norm(target_position - initial_position)
    print(f"Initial position: {initial_position}")
    print(f"Initial distance to target: {distance_to_target:.2f}m")
    
    # Navigation parameters 
    max_steps = 50
    waypoint_reach_threshold = 2.0
    collision_recovery_distance = 4.0
    timeout_seconds = 120
    step_count = 0
    collision_count = 0
    revisited_positions = {}
    
    start_time = time.time()
    
    try:
        print("\nStarting navigation with improved obstacle avoidance...")
        
        # Main navigation loop
        while step_count < max_steps:
            current_position = drone.get_position()
            distance_to_target = np.linalg.norm(target_position - current_position)
            
            # Check if target reached
            if distance_to_target < waypoint_reach_threshold:
                print(f"\n🎯 TARGET REACHED in {step_count} steps!")
                break
            
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                print(f"\n⏰ Test timed out after {timeout_seconds} seconds")
                break
                
            # Round position to nearest meter to track revisits (detect loops)
            position_key = tuple(np.round(current_position).astype(int))
            revisited_positions[position_key] = revisited_positions.get(position_key, 0) + 1
            revisit_count = revisited_positions[position_key]
              if revisit_count > 1:
                print(f"Position {position_key} revisited {revisit_count} times")
            
            print(f"\nStep {step_count}: Current position {current_position}")
            print(f"Distance to target: {distance_to_target:.2f}m")
            
            # Scan for obstacles
            obstacle_positions = []
            obstacle_distances = []
            
            try:
                obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
                
                if obstacle_positions and obstacle_distances:
                    closest_dist = min(obstacle_distances)
                    print(f"Detected {len(obstacle_positions)} obstacle points. Closest at {closest_dist:.2f}m")
                else:
                    print("No obstacles detected")
            except Exception as e:
                print(f"Error scanning for obstacles: {e}")
            
            # Check if path to target is clear
            path_clear = obstacle_detector.is_path_clear(current_position, target_position)
            
            # Prepare the observation for path planning
            # Create observation data to send to the active inference planner
            obs_data = {
                "state": {
                    "distance": float(distance_to_target),
                    "azimuth": 0.0,  # Simplified for test
                    "elevation": 0.0,  # Simplified for test
                    "suitability": 1.0 if not obstacle_positions else max(0.3, 1.0 - len(obstacle_positions)/100)
                },
                "drone_position": current_position.tolist(),
                "target_position": target_position.tolist(),
                "obstacle_positions": [pos.tolist() for pos in obstacle_positions],
                "obstacle_density": len(obstacle_positions) / 100.0 if obstacle_positions else 0.0,
                "nearest_obstacle_distances": obstacle_distances if obstacle_distances else [100.0],
                "voxel_grid": [pos.tolist() for pos in obstacle_positions],
                "high_density_area": len(obstacle_positions) > 20,
                "direct_path_clear": path_clear,
            }
            
            # Save observation to file for Julia planner
            interface_dir = "interface"
            os.makedirs(interface_dir, exist_ok=True)
            with open(os.path.join(interface_dir, "obs_input.json"), "w") as f:
                json.dump(obs_data, f)
            
            # Run planning process
            print("Running path planning...")
            os.system("julia run_planning.jl")
            
            # Read the next waypoint
            try:
                with open(os.path.join(interface_dir, "next_waypoint.json"), "r") as f:
                    waypoint_data = json.load(f)
                    next_waypoint = np.array(waypoint_data["next_waypoint"])
                    print(f"Next waypoint: {next_waypoint}")
            except Exception as e:
                print(f"Error reading waypoint: {e}")
                break
                
            # Move to the waypoint
            try:
                drone.move_to_position(next_waypoint[0], next_waypoint[1], next_waypoint[2], velocity=5.0)
                
                # Check if we've collided
                collision_info = drone.client.simGetCollisionInfo()
                if collision_info.has_collided:
                    collision_count += 1
                    print(f"⚠️ COLLISION DETECTED ({collision_count} total)")
                    
                    # Get collision normal for recovery direction
                    collision_normal = np.array([
                        collision_info.normal.x_val, 
                        collision_info.normal.y_val,
                        collision_info.normal.z_val
                    ])
                    
                    # Move away from collision using the collision normal
                    recovery_vector = collision_normal * collision_recovery_distance
                    recovery_position = current_position + recovery_vector
                    print(f"Attempting collision recovery: moving {collision_recovery_distance}m along normal")
                    drone.move_to_position(
                        recovery_position[0], 
                        recovery_position[1], 
                        recovery_position[2],
                        velocity=3.0
                    )
            except Exception as e:
                print(f"Error during movement: {e}")
            
            step_count += 1
            time.sleep(1)  # Small pause between steps
            
        # Test summary
        elapsed_time = time.time() - start_time
        final_position = drone.get_position()
        final_distance = np.linalg.norm(target_position - final_position)
        
        print("\n===== TEST SUMMARY =====")
        print(f"Steps taken: {step_count}/{max_steps}")
        print(f"Collisions: {collision_count}")
        print(f"Time elapsed: {elapsed_time:.1f} seconds")
        print(f"Final position: {final_position}")
        print(f"Final distance to target: {final_distance:.2f}m")
        
        # Check success criteria
        if final_distance < waypoint_reach_threshold:
            print("\n✅ TEST PASSED: Target reached successfully!")
        else:
            print(f"\n❌ TEST FAILED: Final distance to target ({final_distance:.2f}m) exceeds threshold ({waypoint_reach_threshold}m)")
            
        # Check position revisits to detect loops
        loop_count = sum(1 for count in revisited_positions.values() if count > 1)
        repeat_locations = sum(count-1 for count in revisited_positions.values() if count > 1)
        print(f"Position loop analysis: {loop_count} locations were revisited, with {repeat_locations} total repeats")
        
        print("Most revisited positions:")
        top_revisits = sorted(revisited_positions.items(), key=lambda x: x[1], reverse=True)[:5]
        for pos, count in top_revisits:
            if count > 1:
                print(f"  {pos}: {count} visits")
                
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Land the drone
        print("\nLanding drone...")
        drone.land()
        drone.disconnect()

if __name__ == "__main__":
    try:
        test_obstacle_avoidance()
        print("Test completed successfully")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
