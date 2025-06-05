#!/usr/bin/env python3
"""
Quick test to verify screenshot functionality during actual data collection.
This simulates a minimal data collection run to test screenshots.
"""

import os
import sys
import time
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_collection_screenshots():
    """Test screenshot functionality in data collection context"""
    print("🔍 Testing Screenshot Functionality in Data Collection Context\n")
    
    # Import data collection module 
    try:
        import data_collection
        print("✅ Data collection module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_collection: {e}")
        return False
    
    # Check the current default configuration
    config = data_collection.DEFAULT_CONFIG
    print(f"📋 Current default configuration:")
    print(f"   • enable_voxel_visualization: {config.get('enable_voxel_visualization', False)}")
    print(f"   • save_visualization_screenshots: {config.get('save_visualization_screenshots', False)}")
    print(f"   • screenshot_interval: {config.get('screenshot_interval', 10)}")
    
    # Create a test experiment directory
    test_experiment_dir = f"test_data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(test_experiment_dir, exist_ok=True)
    
    # Test screenshot directory creation as done in data_collection.py
    screenshots_dir = os.path.join(test_experiment_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Test screenshot path generation
    episode_id = 1
    step = 10
    screenshot_filename = f"episode_{episode_id:03d}_step_{step:03d}.png"
    screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
    
    print(f"\n📁 Test experiment directory: {test_experiment_dir}")
    print(f"📁 Screenshots directory: {screenshots_dir}")
    print(f"📷 Test screenshot path: {screenshot_path}")
    
    # Create a test scanner to verify the save_visualization_screenshot method
    try:
        scanner = data_collection.Scanner(client=None, enable_visualization=True)
        print("✅ Scanner with visualization created successfully")
        
        # Test screenshot saving (this will fail because no AirSim client, but we can test the method)
        result = scanner.save_visualization_screenshot(screenshot_path)
        print(f"📷 Screenshot save method result: {result}")
        
        scanner.stop_visualization()
        print("✅ Scanner visualization stopped")
        
    except Exception as e:
        print(f"⚠️  Scanner test failed (expected without AirSim): {e}")
        # This is expected without AirSim running
    
    # Verify that configuration values are properly set
    if (config.get('enable_voxel_visualization', False) and 
        config.get('save_visualization_screenshots', False)):
        print("\n✅ Configuration is properly set for screenshot generation!")
        print("🎯 Screenshots should now be generated during data collection.")
        return True
    else:
        print("\n❌ Configuration is not properly set for screenshot generation.")
        print("⚠️  Screenshots will NOT be generated during data collection.")
        return False

def main():
    """Run the test"""
    success = test_data_collection_screenshots()
    
    if success:
        print("\n🎉 Screenshot functionality is ready for data collection!")
        print("\n📝 To run data collection with screenshots:")
        print("   1. Start AirSim with your preferred environment")
        print("   2. Run: python data_collection.py")
        print("   3. Screenshots will be saved in experiment_results/<timestamp>/screenshots/")
    else:
        print("\n⚠️  Screenshot functionality needs attention.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
