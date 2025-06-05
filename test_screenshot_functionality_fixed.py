#!/usr/bin/env python3
"""
Test script to verify screenshot functionality works correctly.
This creates a minimal test to check if screenshots are saved properly.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from voxel_visualization import VoxelGridVisualizer
    print("✅ VoxelGridVisualizer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import VoxelGridVisualizer: {e}")
    sys.exit(1)

def test_screenshot_functionality():
    """Test basic screenshot functionality"""
    print("\n=== Testing Screenshot Functionality ===")
    
    # Create test directory
    test_dir = "test_screenshots"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Initialize visualizer (without AirSim client for testing)
        print("Creating visualizer...")
        visualizer = VoxelGridVisualizer(voxel_size=0.5, visualization_range=25.0)
        
        # Start the visualization
        print("Starting visualization...")
        visualizer.start_visualization("Test Screenshot")
        
        # Wait for visualization to initialize
        time.sleep(3.0)
        
        # Add some test obstacles/voxels
        print("Adding test obstacles...")
        test_obstacles = [
            [1.0, 1.0, -5.0],
            [2.0, 1.0, -5.0], 
            [1.0, 2.0, -5.0],
            [2.0, 2.0, -5.0],
            [3.0, 3.0, -6.0]
        ]
        
        visualizer.update_obstacles(test_obstacles)
        
        # Update drone position for context
        print("Setting drone position...")
        visualizer.update_drone_position([0.0, 0.0, -5.0])
        
        # Set target position
        print("Setting target position...")
        visualizer.update_target_position([5.0, 5.0, -5.0])
        
        # Allow time for visualization to update
        time.sleep(1.0)
        
        # Test screenshot saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(test_dir, f"test_screenshot_{timestamp}.png")
        
        print(f"Attempting to save screenshot to: {screenshot_path}")
        success = visualizer.save_screenshot(screenshot_path)
        
        if success and os.path.exists(screenshot_path):
            file_size = os.path.getsize(screenshot_path)
            print(f"✅ Screenshot saved successfully!")
            print(f"   Path: {screenshot_path}")
            print(f"   Size: {file_size} bytes")
            
            # Check if file size is reasonable (not empty)
            if file_size > 1000:  # At least 1KB
                print("✅ Screenshot file size looks reasonable")
                return True
            else:
                print(f"⚠️  Screenshot file size seems too small: {file_size} bytes")
                return False
        else:
            print("❌ Failed to save screenshot")
            return False
            
    except Exception as e:
        print(f"❌ Error during screenshot test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        try:
            if 'visualizer' in locals():
                visualizer.stop_visualization()
                print("Cleaned up visualizer")
        except:
            pass

def test_experiment_directory_structure():
    """Test that experiment directory structure works correctly"""
    print("\n=== Testing Experiment Directory Structure ===")
    
    # Create a mock experiment directory
    test_experiment_dir = f"test_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(test_experiment_dir, exist_ok=True)
    
    # Test screenshot subdirectory creation
    screenshots_dir = os.path.join(test_experiment_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Test screenshot path generation (as used in data_collection.py)
    episode_id = 1
    step = 20
    screenshot_filename = f"episode_{episode_id:03d}_step_{step:03d}.png"
    screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
    
    print(f"Test experiment directory: {test_experiment_dir}")
    print(f"Screenshots directory: {screenshots_dir}")
    print(f"Generated screenshot path: {screenshot_path}")
    
    # Verify structure
    if os.path.exists(test_experiment_dir) and os.path.exists(screenshots_dir):
        print("✅ Directory structure created successfully")
        
        # Test writing a placeholder file
        try:
            with open(screenshot_path, 'w') as f:
                f.write("test")
            
            if os.path.exists(screenshot_path):
                print("✅ File writing to screenshots directory works")
                os.remove(screenshot_path)  # Clean up test file
                return True
            else:
                print("❌ Failed to write file to screenshots directory")
                return False
                
        except Exception as e:
            print(f"❌ Error writing to screenshots directory: {e}")
            return False
    else:
        print("❌ Failed to create directory structure")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing Screenshot Functionality\n")
    
    # Set up logging to see any issues
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    screenshot_test_passed = test_screenshot_functionality()
    directory_test_passed = test_experiment_directory_structure()
    
    print("\n=== Test Results ===")
    print(f"Screenshot functionality: {'✅ PASS' if screenshot_test_passed else '❌ FAIL'}")
    print(f"Directory structure: {'✅ PASS' if directory_test_passed else '❌ FAIL'}")
    
    if screenshot_test_passed and directory_test_passed:
        print("\n🎉 All tests passed! Screenshot system should work during data collection.")
        print("\n📝 Make sure your data collection configuration has:")
        print("   • enable_voxel_visualization: true")
        print("   • save_visualization_screenshots: true")
    else:
        print("\n⚠️  Some tests failed. Check the voxel visualization system.")
        
    return screenshot_test_passed and directory_test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
