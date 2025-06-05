"""
Test script for voxel grid visualization integration with the drone navigation system
"""

import time
import numpy as np
import logging
from data_collection import Scanner
import airsim

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_voxel_visualization():
    """Test the voxel visualization with simulated data"""
    print("Testing Voxel Grid Visualization...")
    
    try:
        # Connect to AirSim
        print("Connecting to AirSim...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        # Create scanner with visualization enabled
        print("Creating scanner with visualization...")
        scanner = Scanner(client, scan_range=25.0, enable_visualization=True, voxel_size=0.3)
        
        if not scanner.enable_visualization:
            print("Visualization not enabled - check Open3D installation")
            return
        
        print("Visualization started successfully!")
        
        # Test with real AirSim data
        print("Testing with real obstacle scan...")
        for i in range(20):
            print(f"Scan iteration {i+1}/20")
            
            # Get drone state
            drone_state = client.getMultirotorState().kinematics_estimated
            drone_pos = [
                drone_state.position.x_val,
                drone_state.position.y_val,
                drone_state.position.z_val
            ]
            
            # Update drone position in visualization
            scanner.update_visualization_drone(drone_pos)
            
            # Set a target position
            target_pos = [drone_pos[0] + 10, drone_pos[1] + 5, drone_pos[2]]
            scanner.update_visualization_target(target_pos)
            
            # Perform comprehensive scan (this will update obstacles in visualization)
            scan_result = scanner.get_comprehensive_obstacle_scan(include_raw_points=False)
            
            # Print scan summary
            if 'summary' in scan_result:
                summary = scan_result['summary']
                print(f"  Obstacles: {summary['total_obstacles']}, "
                      f"Closest: {summary['closest_distance']:.2f}m, "
                      f"Density: {summary['obstacle_density']:.3f}")
            
            # Save screenshot every 5 iterations
            if (i + 1) % 5 == 0:
                filename = f"voxel_screenshot_{i+1:02d}.png"
                if scanner.save_visualization_screenshot(filename):
                    print(f"  Saved screenshot: {filename}")
            
            time.sleep(1.0)  # Wait between scans
        
        # Get visualization stats
        stats = scanner.get_visualization_stats()
        print(f"Visualization stats: {stats}")
        
        print("\nVisualization test completed. Press Enter to stop...")
        input()
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'scanner' in locals():
            scanner.stop_visualization()
        print("Test completed.")

def test_visualization_without_airsim():
    """Test visualization with simulated data (no AirSim required)"""
    print("Testing Voxel Visualization with simulated data...")
    
    try:
        from voxel_visualization import create_voxel_visualizer
        
        # Create visualizer
        visualizer = create_voxel_visualizer(voxel_size=0.5, visualization_range=20.0)
        
        print("Visualization started. Running simulation...")
        
        # Simulate drone movement and obstacles
        for i in range(50):
            t = i * 0.2
            
            # Simulate drone following a path
            drone_pos = [
                10 * np.cos(t * 0.3),
                10 * np.sin(t * 0.3),
                -5 + 3 * np.sin(t * 0.5)
            ]
            visualizer.update_drone_position(drone_pos)
            
            # Update target ahead of drone
            target_pos = [
                drone_pos[0] + 5 * np.cos(t * 0.3 + 0.5),
                drone_pos[1] + 5 * np.sin(t * 0.3 + 0.5),
                drone_pos[2]
            ]
            visualizer.update_target_position(target_pos)
            
            # Generate some random obstacles around the drone
            if i % 5 == 0:  # Update obstacles every 5 iterations
                obstacles = []
                num_obstacles = np.random.randint(15, 30)
                
                for _ in range(num_obstacles):
                    # Generate obstacles in a sphere around drone
                    angle = np.random.uniform(0, 2 * np.pi)
                    elevation = np.random.uniform(-np.pi/4, np.pi/4)
                    distance = np.random.uniform(2, 15)
                    
                    obs_x = drone_pos[0] + distance * np.cos(elevation) * np.cos(angle)
                    obs_y = drone_pos[1] + distance * np.cos(elevation) * np.sin(angle)
                    obs_z = drone_pos[2] + distance * np.sin(elevation)
                    
                    obstacles.append([obs_x, obs_y, obs_z])
                
                visualizer.update_obstacles(obstacles)
            
            # Save screenshot periodically
            if i % 10 == 0:
                filename = f"sim_voxel_screenshot_{i:02d}.png"
                visualizer.save_screenshot(filename)
                print(f"Saved screenshot: {filename}")
            
            print(f"Simulation step {i+1}/50 - Drone: [{drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f}]")
            time.sleep(0.2)
        
        print("\nSimulation completed. Press Enter to stop...")
        input()
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'visualizer' in locals():
            visualizer.stop_visualization()
        print("Simulation test completed.")

if __name__ == "__main__":
    print("Voxel Grid Visualization Test")
    print("=" * 40)
    
    # Ask user which test to run
    print("Select test mode:")
    print("1. Test with AirSim (requires AirSim running)")
    print("2. Test with simulated data (no AirSim required)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            test_voxel_visualization()
        elif choice == "2":
            test_visualization_without_airsim()
        else:
            print("Invalid choice. Running simulated test...")
            test_visualization_without_airsim()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Test failed: {e}")
