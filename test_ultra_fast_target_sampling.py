#!/usr/bin/env python3
"""
Test script for the ultra-fast target sampling function.
This verifies that the optimized sample_visible_target function works correctly
without any drone movement, using only raycasting for validation.
"""

import sys
import os
import time
import logging

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import airsim
from data_collection import sample_visible_target

def test_ultra_fast_target_sampling():
    """Test the ultra-fast target sampling function"""
    print("=== Testing Ultra-Fast Target Sampling ===")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Connect to AirSim
        print("Connecting to AirSim...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        # Enable API control and arm/takeoff the drone (one time setup)
        print("Setting up drone...")
        client.enableApiControl(True)
        client.armDisarm(True)
        print("Taking off...")
        client.takeoffAsync().join()
        
        # Get current position
        current_pose = client.simGetVehiclePose()
        current_pos = [current_pose.position.x_val, current_pose.position.y_val, current_pose.position.z_val]
        print(f"Current drone position: {[round(p, 2) for p in current_pos]}")
          # Test parameters
        distance_range = (20.0, 100.0)  # Updated to 20-100 meter range
        max_attempts = 100
        num_tests = 5
        
        print(f"\nTesting target sampling with:")
        print(f"  Distance range: {distance_range[0]}-{distance_range[1]} meters")
        print(f"  Max attempts per target: {max_attempts}")
        print(f"  Number of targets to find: {num_tests}")
        print(f"  Using ultra-fast raycasting-only approach (no drone movement)")
        print(f"  With improved collision detection timeout")
        
        targets_found = []
        total_time = 0
        
        for i in range(num_tests):
            print(f"\n--- Test {i+1}/{num_tests} ---")
            
            start_time = time.time()
            
            try:
                # Sample target using ultra-fast method (no drone movement)
                target = sample_visible_target(
                    current_pos=current_pos,
                    distance_range=distance_range,
                    client=client,
                    max_attempts=max_attempts,
                    episode_id=i,
                    seed=12345,  # Fixed seed for reproducible results
                    ray_checks=7
                )
                
                end_time = time.time()
                sampling_time = end_time - start_time
                total_time += sampling_time
                
                # Calculate distance
                distance = ((target[0] - current_pos[0])**2 + 
                           (target[1] - current_pos[1])**2 + 
                           (target[2] - current_pos[2])**2)**0.5
                
                targets_found.append({
                    'target': target,
                    'distance': distance,
                    'time': sampling_time
                })
                
                print(f"✓ Found target: {[round(p, 2) for p in target]}")
                print(f"  Distance: {distance:.2f}m")
                print(f"  Sampling time: {sampling_time:.3f}s")
                
                # Verify the target is within distance range
                if distance_range[0] <= distance <= distance_range[1]:
                    print(f"  ✓ Distance within range")
                else:
                    print(f"  ⚠ Distance outside range [{distance_range[0]}-{distance_range[1]}]")
                
            except Exception as e:
                print(f"✗ Error finding target: {e}")
                continue
        
        # Summary
        print(f"\n=== Ultra-Fast Target Sampling Test Results ===")
        print(f"Targets found: {len(targets_found)}/{num_tests}")
        print(f"Average sampling time: {total_time/len(targets_found):.3f}s per target")
        print(f"Total time: {total_time:.3f}s")
        
        if targets_found:
            distances = [t['distance'] for t in targets_found]
            times = [t['time'] for t in targets_found]
            print(f"Distance range: {min(distances):.2f}m - {max(distances):.2f}m")
            print(f"Time range: {min(times):.3f}s - {max(times):.3f}s")
            
            print("\nAll targets found:")
            for i, target_info in enumerate(targets_found):
                print(f"  {i+1}: {[round(p, 2) for p in target_info['target']]} "
                      f"(dist: {target_info['distance']:.2f}m, time: {target_info['time']:.3f}s)")
        
        # Verify drone hasn't moved
        final_pose = client.simGetVehiclePose()
        final_pos = [final_pose.position.x_val, final_pose.position.y_val, final_pose.position.z_val]
        
        position_change = ((final_pos[0] - current_pos[0])**2 + 
                          (final_pos[1] - current_pos[1])**2 + 
                          (final_pos[2] - current_pos[2])**2)**0.5
        
        print(f"\nDrone position verification:")
        print(f"  Initial: {[round(p, 2) for p in current_pos]}")
        print(f"  Final: {[round(p, 2) for p in final_pos]}")
        print(f"  Position change: {position_change:.3f}m")
        
        if position_change < 0.1:  # Less than 10cm movement
            print("  ✓ Drone stayed in place during target sampling")
        else:
            print("  ⚠ Drone moved during target sampling")
          # Test with different distance ranges
        print(f"\n--- Testing Different Distance Ranges ---")
        test_ranges = [(20.0, 40.0), (50.0, 80.0), (80.0, 120.0)]  # Updated ranges for new scale
        
        for dist_range in test_ranges:
            print(f"\nTesting range {dist_range[0]}-{dist_range[1]}m:")
            
            start_time = time.time()
            try:
                target = sample_visible_target(
                    current_pos=current_pos,
                    distance_range=dist_range,
                    client=client,
                    max_attempts=50,
                    episode_id=99,
                    seed=54321,
                    ray_checks=5
                )
                
                sampling_time = time.time() - start_time
                distance = ((target[0] - current_pos[0])**2 + 
                           (target[1] - current_pos[1])**2 + 
                           (target[2] - current_pos[2])**2)**0.5
                
                print(f"  ✓ Target: {[round(p, 2) for p in target]} (dist: {distance:.2f}m, time: {sampling_time:.3f}s)")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        # Clean up
        print(f"\nLanding and cleaning up...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        
        print("\n✓ Ultra-fast target sampling test completed successfully!")
        
        return len(targets_found) == num_tests
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ultra_fast_target_sampling()
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
