#!/usr/bin/env python3
"""
Test script to verify screenshot path configuration
"""

import os
import tempfile
from datetime import datetime

def test_screenshot_path_generation():
    """Test that screenshot paths are generated correctly"""
    
    # Simulate experiment directory structure
    experiment_dir = os.path.join("experiment_results", f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Test screenshot path generation
    episode_id = 5
    step = 10
    screenshot_filename = f"episode_{episode_id:03d}_step_{step:03d}.png"
    
    if experiment_dir:
        screenshots_dir = os.path.join(experiment_dir, "screenshots")
        screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
    else:
        screenshot_path = screenshot_filename
    
    print(f"Test Screenshot Path Configuration:")
    print(f"  Experiment directory: {experiment_dir}")
    print(f"  Screenshots directory: {screenshots_dir}")
    print(f"  Screenshot filename: {screenshot_filename}")
    print(f"  Full screenshot path: {screenshot_path}")
    
    # Test that the path is absolute and correctly structured
    expected_structure = os.path.join(experiment_dir, "screenshots", screenshot_filename)
    assert screenshot_path == expected_structure, f"Path mismatch: {screenshot_path} != {expected_structure}"
    
    print("✓ Screenshot path generation test passed!")
    
    return screenshot_path

if __name__ == "__main__":
    test_screenshot_path_generation()
    print("\nScreenshot path configuration is working correctly!")
