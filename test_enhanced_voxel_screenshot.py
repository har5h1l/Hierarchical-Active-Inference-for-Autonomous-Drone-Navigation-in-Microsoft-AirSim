#!/usr/bin/env python3
"""
Test Enhanced Screenshot Functionality in Voxel Visualization

This script tests the enhanced screenshot implementation in the actual voxel visualization system
to verify that the black screenshot issue has been resolved.
"""

import sys
import os
import time
import logging
import numpy as np
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voxel_visualization import VoxelGridVisualizer, create_voxel_visualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_enhanced_screenshot_functionality():
    """Test the enhanced screenshot functionality"""
    print("Testing Enhanced Screenshot Functionality")
    print("="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test 1: Direct VoxelGridVisualizer usage
    print("\n=== Test 1: Direct VoxelGridVisualizer Usage ===")
    
    try:
        # Create visualizer instance
        visualizer = VoxelGridVisualizer(
            voxel_size=0.5,
            grid_size=50,
            visualization_range=25.0,
            update_rate=10.0
        )
        
        # Start visualization
        visualizer.start_visualization("Enhanced Screenshot Test")
        
        # Wait for initialization
        time.sleep(2.0)
        
        # Add some test data
        print("Adding test data...")
        
        # Add drone position
        drone_pos = [0.0, 0.0, -5.0]
        visualizer.update_drone_position(drone_pos)
        
        # Add target position
        target_pos = [10.0, 10.0, -5.0]
        visualizer.update_target_position(target_pos)
        
        # Add some obstacle positions
        obstacle_positions = []
        for i in range(20):
            x = np.random.uniform(-15, 15)
            y = np.random.uniform(-15, 15)
            z = np.random.uniform(-10, 0)
            obstacle_positions.append([x, y, z])
        
        visualizer.update_obstacles(obstacle_positions)
        
        # Wait for rendering to complete
        time.sleep(3.0)
        
        # Test multiple screenshots with the enhanced method
        screenshot_results = []
        
        for i in range(3):
            filename = f"test_enhanced_direct_{i+1}_{timestamp}.png"
            print(f"Taking screenshot {i+1}: {filename}")
            
            success = visualizer.save_screenshot(filename)
            
            if success and os.path.exists(filename):
                file_size = os.path.getsize(filename)
                screenshot_results.append((filename, file_size, file_size > 10000))
                print(f"  Screenshot {i+1}: {file_size} bytes - {'SUCCESS' if file_size > 10000 else 'FAILED'}")
            else:
                screenshot_results.append((filename, 0, False))
                print(f"  Screenshot {i+1}: FAILED to create")
            
            time.sleep(1.0)  # Wait between screenshots
        
        # Stop visualization
        visualizer.stop_visualization()
        
        # Report results
        successful_screenshots = [r for r in screenshot_results if r[2]]
        print(f"\nDirect test results: {len(successful_screenshots)}/{len(screenshot_results)} screenshots successful")
        
        return len(successful_screenshots) > 0
        
    except Exception as e:
        print(f"Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factory_function():
    """Test using the factory function"""
    print("\n=== Test 2: Factory Function Usage ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Create visualizer using factory function
        visualizer = create_voxel_visualizer(
            voxel_size=0.3,
            visualization_range=20.0,
            auto_start=True
        )
        
        if visualizer is None:
            print("Factory function returned None")
            return False
        
        # Wait for initialization
        time.sleep(2.0)
        
        # Add test data similar to real usage
        print("Adding realistic test data...")
        
        # Simulate drone movement path
        path_points = []
        for i in range(10):
            x = i * 2.0
            y = np.sin(i * 0.5) * 3.0
            z = -5.0 + np.cos(i * 0.3) * 2.0
            path_points.append([x, y, z])
            visualizer.update_drone_position([x, y, z])
            time.sleep(0.1)
        
        # Set target
        visualizer.update_target_position([20.0, 5.0, -3.0])
        
        # Add obstacles in a pattern
        obstacles = []
        for x in range(-10, 20, 3):
            for y in range(-10, 10, 3):
                if np.random.random() > 0.7:  # Sparse obstacles
                    z = np.random.uniform(-8, -2)
                    obstacles.append([x, y, z])
        
        visualizer.update_obstacles(obstacles)
        
        # Wait for rendering
        time.sleep(2.0)
        
        # Test screenshots
        screenshot_results = []
        
        for i in range(2):
            filename = f"test_enhanced_factory_{i+1}_{timestamp}.png"
            print(f"Taking factory screenshot {i+1}: {filename}")
            
            success = visualizer.save_screenshot(filename)
            
            if success and os.path.exists(filename):
                file_size = os.path.getsize(filename)
                screenshot_results.append((filename, file_size, file_size > 10000))
                print(f"  Factory screenshot {i+1}: {file_size} bytes - {'SUCCESS' if file_size > 10000 else 'FAILED'}")
            else:
                screenshot_results.append((filename, 0, False))
                print(f"  Factory screenshot {i+1}: FAILED to create")
            
            time.sleep(1.0)
        
        # Stop visualization
        visualizer.stop_visualization()
        
        # Report results
        successful_screenshots = [r for r in screenshot_results if r[2]]
        print(f"\nFactory test results: {len(successful_screenshots)}/{len(screenshot_results)} screenshots successful")
        
        return len(successful_screenshots) > 0
        
    except Exception as e:
        print(f"Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_screenshot_methods():
    """Test individual screenshot methods"""
    print("\n=== Test 3: Individual Screenshot Methods ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Create visualizer
        visualizer = VoxelGridVisualizer(voxel_size=0.4)
        visualizer.start_visualization("Screenshot Methods Test")
        time.sleep(2.0)
        
        # Add minimal test scene
        visualizer.update_drone_position([0, 0, -5])
        visualizer.update_target_position([5, 5, -5])
        visualizer.update_obstacles([[2, 2, -5], [3, 3, -5], [-2, 1, -6]])
        time.sleep(2.0)
        
        # Test individual methods if available
        methods_to_test = [
            ('standard', '_capture_screen_image_standard'),
            ('buffer', '_capture_screen_image_with_buffer'),
            ('fallback', '_capture_screen_image_fallback')
        ]
        
        method_results = {}
        
        for method_name, method_attr in methods_to_test:
            if hasattr(visualizer, method_attr):
                filename = f"test_method_{method_name}_{timestamp}.png"
                print(f"Testing {method_name} method: {filename}")
                
                try:
                    method = getattr(visualizer, method_attr)
                    success = method(filename)
                    
                    if success and os.path.exists(filename):
                        file_size = os.path.getsize(filename)
                        method_results[method_name] = (file_size, file_size > 5000)
                        print(f"  {method_name} method: {file_size} bytes - {'SUCCESS' if file_size > 5000 else 'FAILED'}")
                    else:
                        method_results[method_name] = (0, False)
                        print(f"  {method_name} method: FAILED")
                        
                except Exception as e:
                    method_results[method_name] = (0, False)
                    print(f"  {method_name} method: ERROR - {e}")
            else:
                print(f"  {method_name} method: NOT AVAILABLE")
        
        visualizer.stop_visualization()
        
        # Report method results
        successful_methods = [name for name, (size, success) in method_results.items() if success]
        print(f"\nMethod test results: {len(successful_methods)}/{len(method_results)} methods successful")
        print(f"Working methods: {successful_methods}")
        
        return len(successful_methods) > 0
        
    except Exception as e:
        print(f"Method test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Enhanced Screenshot Functionality Test")
    print("="*60)
    
    # Run all tests
    test_results = []
    
    # Test 1: Direct usage
    result1 = test_enhanced_screenshot_functionality()
    test_results.append(("Direct VoxelGridVisualizer", result1))
    
    # Test 2: Factory function
    result2 = test_factory_function()
    test_results.append(("Factory Function", result2))
    
    # Test 3: Individual methods
    result3 = test_screenshot_methods()
    test_results.append(("Individual Methods", result3))
    
    # Summary
    print("\n" + "="*60)
    print("ENHANCED SCREENSHOT TEST SUMMARY")
    print("="*60)
    
    successful_tests = 0
    for test_name, success in test_results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")
        if success:
            successful_tests += 1
    
    print(f"\nOverall: {successful_tests}/{len(test_results)} tests passed")
    
    # Check if any screenshots were created
    timestamp_pattern = datetime.now().strftime("%Y%m%d")
    screenshot_files = [f for f in os.listdir('.') if f.startswith('test_enhanced_') and timestamp_pattern in f and f.endswith('.png')]
    
    valid_screenshots = 0
    total_size = 0
    
    print(f"\nScreenshot files created: {len(screenshot_files)}")
    for filename in screenshot_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            total_size += size
            if size > 5000:
                valid_screenshots += 1
                print(f"  ✓ {filename}: {size} bytes")
            else:
                print(f"  ✗ {filename}: {size} bytes (too small)")
    
    print(f"\nValid screenshots: {valid_screenshots}/{len(screenshot_files)}")
    print(f"Total size: {total_size} bytes")
    
    if valid_screenshots > 0:
        print("\n🎉 SUCCESS: Enhanced screenshot functionality is working!")
        print("The black screenshot issue has been resolved.")
    else:
        print("\n❌ FAILURE: Enhanced screenshot functionality is not working properly.")
        print("The black screenshot issue persists.")
    
    return valid_screenshots > 0

if __name__ == "__main__":
    main()
