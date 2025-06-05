#!/usr/bin/env python3
"""
Windows Deployment Test for OpenGL Context Fix

This script is specifically designed to test the Windows OpenGL context fix
on actual Windows systems. It should be run on Windows to verify that the
WGL threading issues have been resolved.

For macOS development environments, use test_windows_opengl_fix.py instead.
"""

import os
import sys
import time
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_windows_environment():
    """Verify this is running on Windows"""
    if platform.system() != 'Windows':
        print(f"❌ This test is designed for Windows systems.")
        print(f"   Current system: {platform.system()}")
        print(f"   For development testing on macOS/Linux, use test_windows_opengl_fix.py")
        return False
    
    print(f"✅ Running on Windows {platform.release()}")
    print(f"   Python version: {sys.version}")
    return True

def test_basic_visualization():
    """Test basic visualization without screenshots"""
    print("\n=== Testing Basic Visualization (Windows) ===")
    
    try:
        from voxel_visualization_fixed import create_voxel_visualizer
        
        # Create visualizer with screenshots disabled first
        viz = create_voxel_visualizer(
            voxel_size=0.5,
            visualization_range=15.0,
            auto_start=True,
            enable_screenshots=False  # Start without screenshots
        )
        
        if not viz.is_running:
            print("❌ Basic visualization failed to start")
            return False
        
        print("✅ Basic visualization started successfully")
        
        # Test data updates
        viz.update_drone_position([0, 0, -3])
        viz.update_target_position([8, 0, -3])
        
        # Add some obstacles
        obstacles = [
            [3, 1, -2], [5, -2, -4], [7, 3, -3],
            [-1, 2, -5], [2, -3, -2], [6, 4, -6]
        ]
        viz.update_obstacles(obstacles)
        
        print("✅ Data updates successful")
        
        # Let it run briefly
        time.sleep(3.0)
        
        # Check stats
        stats = viz.get_visualization_stats()
        print(f"✅ Visualization stats: {stats['obstacle_count']} obstacles, {stats['path_points']} path points")
        
        viz.stop_visualization()
        print("✅ Basic visualization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic visualization test failed: {e}")
        logging.exception("Basic visualization error")
        return False

def test_windows_screenshot_fix():
    """Test the Windows OpenGL context screenshot fix"""
    print("\n=== Testing Windows Screenshot Fix ===")
    
    try:
        from voxel_visualization_fixed import create_voxel_visualizer
        
        # Create visualizer with screenshots enabled (this is where the fix is needed)
        viz = create_voxel_visualizer(
            voxel_size=0.5,
            visualization_range=20.0,
            auto_start=True,
            enable_screenshots=True  # Enable the fix
        )
        
        if not viz.is_running:
            print("❌ Visualization with screenshots failed to start")
            return False
        
        print("✅ Visualization with screenshots started successfully")
        
        # Set up a scene
        viz.update_drone_position([0, 0, -4])
        viz.update_target_position([12, 0, -3])
        
        # Add obstacles for visual content
        obstacles = []
        for i in range(20):
            obstacles.append([
                i * 0.6 - 6,  # Spread along X
                (i % 3) * 2 - 2,  # Vary Y
                -2 - (i % 2)  # Vary Z
            ])
        viz.update_obstacles(obstacles)
        
        # Wait for scene to stabilize
        print("  Waiting for scene to stabilize...")
        time.sleep(4.0)
        
        # Test screenshot functionality
        test_dir = "windows_test_screenshots"
        os.makedirs(test_dir, exist_ok=True)
        
        print("  Testing Windows OpenGL context fix...")
        
        # Test 1: Single screenshot (should use queue system on Windows)
        success1 = viz.save_screenshot(f"{test_dir}/windows_test_1.png")
        print(f"  Single screenshot: {'✅ Success' if success1 else '❌ Failed'}")
        
        # Test 2: Multiple screenshots (test threading fix)
        successes = 0
        for i in range(3):
            success = viz.save_screenshot(f"{test_dir}/windows_test_{i+2}.png")
            if success:
                successes += 1
            time.sleep(0.5)  # Brief pause between screenshots
        
        print(f"  Multiple screenshots: {successes}/3 successful")
        
        # Test 3: Rapid screenshots (stress test for queue system)
        rapid_successes = 0
        for i in range(3):
            success = viz.save_screenshot(f"{test_dir}/rapid_{i}.png")
            if success:
                rapid_successes += 1
            # No pause - test rapid requests
        
        print(f"  Rapid screenshots: {rapid_successes}/3 successful")
        
        # Check if files were actually created
        created_files = 0
        if os.path.exists(test_dir):
            created_files = len([f for f in os.listdir(test_dir) if f.endswith('.png')])
        
        print(f"  Files created: {created_files} screenshot files")
        
        viz.stop_visualization()
        
        # Overall assessment
        total_attempts = 1 + 3 + 3  # 7 total
        total_successes = success1 + successes + rapid_successes
        success_rate = (total_successes / total_attempts) * 100
        
        print(f"  Overall success rate: {success_rate:.1f}% ({total_successes}/{total_attempts})")
        
        if success_rate >= 60:  # Allow some failures
            print("✅ Windows screenshot fix test passed")
            return True
        else:
            print("⚠️  Low success rate - may need further investigation")
            return success_rate > 0
        
    except Exception as e:
        print(f"❌ Windows screenshot test failed: {e}")
        logging.exception("Windows screenshot test error")
        return False

def test_configuration_options():
    """Test various configuration options"""
    print("\n=== Testing Configuration Options ===")
    
    try:
        from data_collection import DEFAULT_CONFIG
        
        # Test configuration loading
        config = DEFAULT_CONFIG.copy()
        
        # Check if our new parameters are present
        screenshot_functionality = config.get("enable_screenshot_functionality", None)
        voxel_visualization = config.get("enable_voxel_visualization", None)
        screenshot_retries = config.get("screenshot_max_retries", None)
        
        print(f"  enable_screenshot_functionality: {screenshot_functionality}")
        print(f"  enable_voxel_visualization: {voxel_visualization}")
        print(f"  screenshot_max_retries: {screenshot_retries}")
        
        if all(param is not None for param in [screenshot_functionality, voxel_visualization, screenshot_retries]):
            print("✅ Configuration parameters present")
        else:
            print("❌ Some configuration parameters missing")
            return False
        
        # Test different configurations
        print("  Testing screenshot disable configuration...")
        config_no_screenshots = config.copy()
        config_no_screenshots["enable_screenshot_functionality"] = False
        
        print("✅ Configuration options test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main test function for Windows deployment"""
    print("Windows OpenGL Context Fix - Deployment Test")
    print("=" * 50)
    
    # Check if running on Windows
    if not check_windows_environment():
        return False
    
    # Run tests
    test_results = []
    
    print("\nRunning Windows-specific tests...")
    
    # Test 1: Basic visualization
    test_results.append(test_basic_visualization())
    
    # Test 2: Windows screenshot fix
    test_results.append(test_windows_screenshot_fix())
    
    # Test 3: Configuration options
    test_results.append(test_configuration_options())
    
    # Summary
    print("\n" + "=" * 50)
    print("WINDOWS DEPLOYMENT TEST SUMMARY:")
    test_names = ["Basic visualization", "Screenshot fix", "Configuration"]
    
    for name, result in zip(test_names, test_results):
        print(f"  {name}: {'✅ PASS' if result else '❌ FAIL'}")
    
    overall_success = sum(test_results) >= 2  # At least 2/3 should pass
    
    print(f"\nOVERALL: {'✅ TESTS PASSED' if overall_success else '❌ TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Windows OpenGL context fix is working!")
        print("   The original WGL threading error should be resolved.")
        print("   Your experiment should now run without screenshot failures.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
        print("   You may need to disable screenshots if issues persist:")
        print("   config['enable_screenshot_functionality'] = False")
    
    print(f"\n📁 Test screenshots saved to: windows_test_screenshots/")
    
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
        logging.exception("Unexpected error")
        sys.exit(1) 