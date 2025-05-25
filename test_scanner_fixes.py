#!/usr/bin/env python
"""
Test script to verify the improved obstacle detection fixes
"""

import os
import sys
import time
import numpy as np
import logging
from datetime import datetime

# Import the scanner class
from data_collection import Scanner
import airsim

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scanner_improvements():
    """Test the improved Scanner obstacle detection"""
    print("\n===== TESTING IMPROVED SCANNER OBSTACLE DETECTION =====\n")
    
    # Connect to AirSim
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
        print("✅ Connected to AirSim")
    except Exception as e:
        print(f"❌ Failed to connect to AirSim: {e}")
        return False
    
    # Initialize and takeoff
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("Taking off...")
    takeoff_future = client.takeoffAsync()
    takeoff_future.join()
    
    # Move to a test altitude
    print("Moving to test altitude...")
    client.moveToZAsync(-5, 3).join()
    time.sleep(2)
    
    # Initialize scanner
    scanner = Scanner(client, scan_range=15.0)
    print("✅ Scanner initialized")
    
    # Test 1: Basic obstacle detection
    print("\n--- Test 1: Basic Obstacle Detection ---")
    obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
    
    print(f"Detected {len(obstacle_positions)} obstacles")
    if obstacle_distances:
        closest_dist = min(obstacle_distances)
        furthest_dist = max(obstacle_distances)
        avg_dist = np.mean(obstacle_distances)
        print(f"Distance range: {closest_dist:.2f}m - {furthest_dist:.2f}m (avg: {avg_dist:.2f}m)")
    
    # Test 2: Path obstacle scanning
    print("\n--- Test 2: Path Obstacle Scanning ---")
    current_pos = client.getMultirotorState().kinematics_estimated.position
    current_pos = [current_pos.x_val, current_pos.y_val, current_pos.z_val]
    
    # Test various paths
    test_paths = [
        ([0, 5, 0], "Forward 5m"),
        ([5, 0, 0], "Right 5m"),
        ([-5, 0, 0], "Left 5m"),
        ([0, 0, -3], "Up 3m"),
        ([0, 0, 3], "Down 3m"),
        ([3, 3, 0], "Diagonal 3m"),
        ([0, 10, 0], "Forward 10m"),
    ]
    
    for relative_pos, description in test_paths:
        target_pos = [
            current_pos[0] + relative_pos[0],
            current_pos[1] + relative_pos[1], 
            current_pos[2] + relative_pos[2]
        ]
        
        obstacle_detected, obs_pos, obs_dist = scanner.scan_path_for_obstacles(
            current_pos, target_pos, safety_radius=2.0
        )
        
        status = "🚫 BLOCKED" if obstacle_detected else "✅ CLEAR"
        dist_info = f" (obstacle at {obs_dist:.2f}m)" if obstacle_detected else ""
        print(f"  {description}: {status}{dist_info}")
    
    # Test 3: Directional cone scanning
    print("\n--- Test 3: Directional Cone Scanning ---")
    test_directions = [
        ([1, 0, 0], "Forward"),
        ([0, 1, 0], "Right"),
        ([-1, 0, 0], "Backward"),
        ([0, -1, 0], "Left"),
        ([0, 0, -1], "Up"),
        ([0, 0, 1], "Down"),
        ([1, 1, 0], "Forward-Right"),
        ([1, -1, 0], "Forward-Left"),
    ]
    
    for direction, description in test_directions:
        cone_obstacles, cone_distances = scanner.scan_in_direction(
            direction, max_distance=8.0, cone_angle=30.0
        )
        
        count = len(cone_obstacles)
        closest = min(cone_distances) if cone_distances else "N/A"
        print(f"  {description}: {count} obstacles (closest: {closest})")
    
    # Test 4: Close obstacle detection (critical for collision avoidance)
    print("\n--- Test 4: Close Obstacle Detection ---")
    
    # Test very close obstacles with small movements
    close_test_paths = [
        ([0, 1, 0], "Forward 1m"),
        ([1, 0, 0], "Right 1m"), 
        ([0, 0, -1], "Up 1m"),
        ([0.5, 0.5, 0], "Diagonal 0.7m"),
    ]
    
    for relative_pos, description in close_test_paths:
        target_pos = [
            current_pos[0] + relative_pos[0],
            current_pos[1] + relative_pos[1],
            current_pos[2] + relative_pos[2]
        ]
        
        obstacle_detected, obs_pos, obs_dist = scanner.scan_path_for_obstacles(
            current_pos, target_pos, safety_radius=1.0  # Smaller safety radius for close tests
        )
        
        status = "🚫 BLOCKED" if obstacle_detected else "✅ CLEAR"
        dist_info = f" (obstacle at {obs_dist:.2f}m)" if obstacle_detected else ""
        print(f"  {description}: {status}{dist_info}")
    
    # Test 5: Large obstacle detection
    print("\n--- Test 5: Large Obstacle Detection (Clustering) ---")
    
    # This test verifies that the clustering algorithm works
    if obstacle_positions:
        clusters = scanner._cluster_obstacles(obstacle_positions, cluster_radius=2.0)
        print(f"Detected {len(clusters)} obstacle clusters from {len(obstacle_positions)} points")
        
        for i, cluster in enumerate(clusters[:5]):  # Show first 5 clusters
            size = len(cluster['obstacles'])
            extent = cluster['max_extent']
            center = cluster['center']
            print(f"  Cluster {i+1}: {size} obstacles, extent {extent:.2f}m at {center}")
    
    print("\n--- Test Results Summary ---")
    print("✅ All obstacle detection tests completed")
    print("🔍 Check the results above for any unexpected behavior")
    print("💡 The improved scanner should:")
    print("   - Detect close obstacles reliably")
    print("   - Handle large obstacles through clustering")
    print("   - Use multi-stage path validation")
    print("   - Better coordinate transformations")
    
    # Land the drone
    print("\nLanding...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    return True

if __name__ == "__main__":
    try:
        success = test_scanner_improvements()
        if success:
            print("\n✅ Scanner improvement tests completed successfully")
            sys.exit(0)
        else:
            print("\n❌ Scanner improvement tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
