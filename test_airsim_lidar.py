#!/usr/bin/env python3
"""
Test script to verify AirSim connection and LiDAR data availability.
Run this when AirSim is running to debug visualization data issues.
"""

import airsim
import numpy as np
import time

def test_airsim_connection():
    """Test basic AirSim connectivity"""
    try:
        print("Connecting to AirSim...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✓ Successfully connected to AirSim")
        
        # Test drone state
        print("\nTesting drone state...")
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"✓ Drone position: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")
        
        # Test LiDAR data
        print("\nTesting LiDAR data...")
        lidar_data = client.getLidarData()
        
        if len(lidar_data.point_cloud) == 0:
            print("✗ LiDAR point cloud is empty!")
            print("  - Check if LiDAR is enabled in AirSim settings")
            print("  - Ensure drone is in an environment with obstacles")
            print("  - Try moving drone to different location")
            return False
        
        # Parse point cloud
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        print(f"✓ LiDAR detected {len(points)} points")
        
        # Show point statistics
        if len(points) > 0:
            distances = np.linalg.norm(points, axis=1)
            print(f"  - Distance range: {np.min(distances):.2f}m to {np.max(distances):.2f}m")
            print(f"  - Average distance: {np.mean(distances):.2f}m")
            print(f"  - First 5 points: {points[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ AirSim connection failed: {e}")
        print("  - Make sure AirSim is running")
        print("  - Check if Unreal Engine/AirSim is responding")
        print("  - Verify AirSim settings.json configuration")
        return False

def test_visualization_with_real_data():
    """Test visualization with real AirSim data"""
    try:
        from voxel_visualization import VoxelGridVisualizer
        
        print("\nTesting visualization with real AirSim data...")
        
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        # Create visualizer
        visualizer = VoxelGridVisualizer(
            voxel_size=0.5,
            visualization_range=20.0,
            enable_screenshots=True
        )
        
        visualizer.start_visualization("AirSim Data Test")
        time.sleep(2.0)
        
        print("Updating visualization with real data...")
        
        # Get real drone position
        state = client.getMultirotorState()
        drone_pos = [
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val, 
            state.kinematics_estimated.position.z_val
        ]
        
        print(f"Real drone position: {drone_pos}")
        visualizer.update_drone_position(drone_pos)
        
        # Get real LiDAR data
        lidar_data = client.getLidarData()
        if len(lidar_data.point_cloud) > 0:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            # Convert to global coordinates (add drone position)
            global_points = points + np.array(drone_pos)
            obstacle_list = global_points.tolist()
            
            print(f"Updating with {len(obstacle_list)} real obstacles")
            visualizer.update_obstacles(obstacle_list)
        else:
            print("No LiDAR points - using test obstacles")
            # Add some test obstacles around drone
            test_obstacles = []
            for i in range(8):
                angle = i * np.pi / 4
                x = drone_pos[0] + 5 * np.cos(angle)
                y = drone_pos[1] + 5 * np.sin(angle)
                z = drone_pos[2] + np.random.uniform(-1, 1)
                test_obstacles.append([x, y, z])
            visualizer.update_obstacles(test_obstacles)
        
        # Set target
        target_pos = [drone_pos[0] + 10, drone_pos[1] + 10, drone_pos[2] - 2]
        visualizer.update_target_position(target_pos)
        
        print("Visualization running for 20 seconds...")
        print("You should now see:")
        print("- Blue sphere (drone)")
        print("- Red sphere (target)")
        print("- Orange/yellow obstacles or test markers")
        print("- Coordinate axes and ground grid")
        
        time.sleep(20)
        
        # Save screenshot
        success = visualizer.save_screenshot("airsim_test.png")
        print(f"Screenshot result: {success}")
        
        visualizer.stop_visualization()
        print("Test complete!")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("AirSim LiDAR and Visualization Test")
    print("=" * 40)
    
    # Test 1: Basic connectivity
    if not test_airsim_connection():
        print("\nStopping tests - fix AirSim connection first")
        exit(1)
    
    # Test 2: Visualization with real data
    input("\nPress Enter to test visualization with real data...")
    test_visualization_with_real_data()
    
    print("\nIf you saw blank screens:")
    print("1. Check that basic scene elements (axes, grid, spheres) are visible")
    print("2. Try pressing 'R' key to reset camera view")
    print("3. Use mouse to rotate/zoom the view")
    print("4. Check AirSim LiDAR configuration in settings.json") 