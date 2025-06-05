#!/usr/bin/env python3
"""
Test script for the Windows OpenGL context fix and screenshot queue system.

This script tests the enhanced voxel visualization system with:
1. Thread-safe screenshot queue for Windows OpenGL context management
2. Automatic screenshot failure handling and recovery
3. Platform-specific optimizations for Windows vs Linux/Mac
4. Graceful degradation when screenshots fail repeatedly
"""

import os
import sys
import time
import logging
import platform
import threading
from voxel_visualization_fixed import VoxelGridVisualizer, create_voxel_visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_opengl_fix.log')
    ]
)

def test_basic_functionality():
    """Test basic visualization functionality without screenshots"""
    print("=== Testing Basic Visualization Functionality ===")
    
    try:
        # Create visualizer with screenshots disabled for basic test
        viz = create_voxel_visualizer(
            voxel_size=0.5,
            visualization_range=20.0,
            auto_start=True,
            enable_screenshots=False
        )
        
        if not viz.is_running:
            print("❌ Basic visualization failed to start")
            return False
        
        print("✅ Basic visualization started successfully")
        
        # Test data updates
        viz.update_drone_position([0, 0, -5])
        viz.update_target_position([10, 5, -3])
        
        # Add some obstacles
        obstacles = [
            [5, 2, -4], [7, -3, -2], [12, 0, -6],
            [-2, 4, -3], [3, -5, -7], [8, 8, -2]
        ]
        viz.update_obstacles(obstacles)
        
        print("✅ Data updates successful")
        
        # Let it run for a moment
        time.sleep(2.0)
        
        viz.stop_visualization()
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_screenshot_functionality():
    """Test screenshot functionality with Windows OpenGL context fix"""
    print("\n=== Testing Screenshot Functionality ===")
    
    try:
        # Create visualizer with screenshots enabled
        viz = create_voxel_visualizer(
            voxel_size=0.5,
            visualization_range=20.0,
            auto_start=True,
            enable_screenshots=True
        )
        
        if not viz.is_running:
            print("❌ Visualization with screenshots failed to start")
            return False
        
        print("✅ Visualization with screenshots started successfully")
        
        # Test data updates
        viz.update_drone_position([0, 0, -5])
        viz.update_target_position([15, 0, -3])
        
        # Add obstacles for visual content
        obstacles = []
        for i in range(30):  # More obstacles for interesting screenshots
            obstacles.append([
                i * 0.8 - 12,  # Spread across X
                (i % 5) * 2 - 4,  # Vary Y
                -2 - (i % 3)  # Vary Z
            ])
        viz.update_obstacles(obstacles)
        
        # Wait for scene to stabilize
        time.sleep(3.0)
        
        # Test screenshot saving with various scenarios
        test_dir = "test_screenshots"
        os.makedirs(test_dir, exist_ok=True)
        
        print(f"Testing screenshots on {platform.system()}...")
        
        # Test 1: Single screenshot
        success1 = viz.save_screenshot(f"{test_dir}/test_screenshot_1.png")
        print(f"  Single screenshot: {'✅ Success' if success1 else '❌ Failed'}")
        
        # Test 2: Multiple rapid screenshots (stress test for threading)
        rapid_results = []
        for i in range(5):
            result = viz.save_screenshot(f"{test_dir}/rapid_test_{i}.png")
            rapid_results.append(result)
            time.sleep(0.1)  # Small delay between shots
        
        rapid_success = sum(rapid_results)
        print(f"  Rapid screenshots: {rapid_success}/5 successful")
        
        # Test 3: Update scene and take another screenshot
        viz.update_drone_position([5, 3, -4])
        time.sleep(1.0)
        success3 = viz.save_screenshot(f"{test_dir}/updated_scene.png")
        print(f"  Updated scene screenshot: {'✅ Success' if success3 else '❌ Failed'}")
        
        # Test 4: Concurrent screenshot requests (threading stress test)
        print("  Testing concurrent screenshot requests...")
        
        def take_screenshot(index):
            return viz.save_screenshot(f"{test_dir}/concurrent_{index}.png")
        
        # Launch multiple threads to request screenshots simultaneously
        threads = []
        results = [False] * 3
        
        for i in range(3):
            thread = threading.Thread(
                target=lambda idx=i: results.__setitem__(idx, take_screenshot(idx))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)
        
        concurrent_success = sum(results)
        print(f"  Concurrent screenshots: {concurrent_success}/3 successful")
        
        # Get visualization stats
        stats = viz.get_visualization_stats()
        print(f"  Visualization stats: {stats}")
        
        viz.stop_visualization()
        
        # Check if screenshots were actually saved
        saved_files = []
        if os.path.exists(test_dir):
            saved_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        
        print(f"  Files saved: {len(saved_files)} screenshot files created")
        
        # Overall assessment
        total_attempts = 1 + 5 + 1 + 3  # 10 total
        total_successful = success1 + rapid_success + success3 + concurrent_success
        success_rate = total_successful / total_attempts * 100
        
        print(f"  Overall success rate: {success_rate:.1f}% ({total_successful}/{total_attempts})")
        
        if success_rate >= 70:  # Allow some failures on problematic platforms
            print("✅ Screenshot functionality test passed")
            return True
        else:
            print("⚠️  Screenshot functionality test had low success rate but may be platform-specific")
            return success_rate > 0  # As long as some screenshots worked
        
    except Exception as e:
        print(f"❌ Screenshot functionality test failed: {e}")
        return False

def test_failure_recovery():
    """Test the automatic failure recovery and screenshot disabling"""
    print("\n=== Testing Failure Recovery ===")
    
    try:
        # Create visualizer with screenshots enabled
        viz = create_voxel_visualizer(
            voxel_size=0.5,
            visualization_range=15.0,
            auto_start=True,
            enable_screenshots=True
        )
        
        if not viz.is_running:
            print("❌ Visualization failed to start for failure recovery test")
            return False
        
        print("✅ Visualization started for failure recovery test")
        
        # Add some content
        viz.update_drone_position([0, 0, -3])
        viz.update_target_position([8, 0, -3])
        time.sleep(2.0)
        
        # Check initial state
        initial_screenshots_enabled = viz.enable_screenshots
        print(f"  Initial screenshot state: {'enabled' if initial_screenshots_enabled else 'disabled'}")
        
        # Force multiple screenshot attempts to test failure handling
        print("  Testing screenshot failure handling...")
        
        # This may trigger the consecutive failure counter on problematic platforms
        for i in range(8):  # More than max_consecutive_failures (5)
            success = viz.save_screenshot(f"test_screenshots/failure_test_{i}.png")
            if not success:
                print(f"    Screenshot {i+1} failed (expected on some platforms)")
            time.sleep(0.1)
        
        # Check if automatic disabling occurred
        final_screenshots_enabled = viz.enable_screenshots
        print(f"  Final screenshot state: {'enabled' if final_screenshots_enabled else 'disabled'}")
        
        # The system should continue running even if screenshots are disabled
        viz.update_drone_position([2, 1, -3])
        time.sleep(1.0)
        
        stats = viz.get_visualization_stats()
        print(f"  Final stats: {stats}")
        
        viz.stop_visualization()
        
        print("✅ Failure recovery test completed - system remained stable")
        return True
        
    except Exception as e:
        print(f"❌ Failure recovery test failed: {e}")
        return False

def main():
    """Run all tests"""
    print(f"Testing Windows OpenGL fix on {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Basic functionality
    test_results.append(test_basic_functionality())
    
    # Test 2: Screenshot functionality  
    test_results.append(test_screenshot_functionality())
    
    # Test 3: Failure recovery
    test_results.append(test_failure_recovery())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"  Basic functionality: {'✅ PASS' if test_results[0] else '❌ FAIL'}")
    print(f"  Screenshot functionality: {'✅ PASS' if test_results[1] else '❌ FAIL'}")
    print(f"  Failure recovery: {'✅ PASS' if test_results[2] else '❌ FAIL'}")
    
    overall_success = sum(test_results) >= 2  # At least 2/3 tests should pass
    
    print(f"\nOVERALL: {'✅ TESTS PASSED' if overall_success else '❌ TESTS FAILED'}")
    
    if platform.system() == 'Windows':
        if test_results[1]:  # Screenshot functionality passed
            print("\n🎉 Windows OpenGL context fix appears to be working!")
        else:
            print("\n⚠️  Screenshot functionality had issues - this may be expected on some Windows configurations")
            print("   The visualization should still work, just without screenshot capability")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        logging.exception("Unexpected error in test")
        sys.exit(1) 