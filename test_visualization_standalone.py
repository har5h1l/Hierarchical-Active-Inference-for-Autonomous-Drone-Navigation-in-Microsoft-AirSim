#!/usr/bin/env python3
"""
Standalone test script to diagnose voxel visualization issues.
This creates a visualization with test data to isolate the problem.
"""

import time
import numpy as np
import logging
from voxel_visualization import VoxelGridVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_visualization():
    """Test the visualization with synthetic data"""
    print("Creating VoxelGridVisualizer...")
    
    # Create visualizer with moderate settings
    visualizer = VoxelGridVisualizer(
        voxel_size=0.5,
        grid_size=50, 
        max_path_points=100,
        visualization_range=30.0,
        update_rate=5.0,
        enable_screenshots=True
    )
    
    print("Starting visualization...")
    visualizer.start_visualization("Diagnostic Test - Voxel Visualization")
    
    # Wait for initialization
    time.sleep(2.0)
    
    print("Setting up test scene...")
    
    # Test 1: Update drone position (should show blue sphere)
    drone_positions = [
        [0, 0, 0],
        [1, 0, 0], 
        [2, 1, 0],
        [3, 2, -1],
        [4, 3, -1]
    ]
    
    # Test 2: Set target position (should show red sphere)
    target_pos = [10, 5, -2]
    visualizer.update_target_position(target_pos)
    print(f"Target position set to: {target_pos}")
    
    # Test 3: Create obstacle data around the drone
    obstacles = []
    # Create a ring of obstacles around the origin
    for angle in np.linspace(0, 2*np.pi, 12):
        x = 5 * np.cos(angle)
        y = 5 * np.sin(angle) 
        z = np.random.uniform(-2, 0)  # Varying heights
        obstacles.append([x, y, z])
    
    # Add some obstacles on the path to target
    for i in range(5):
        x = 2 + i * 1.5
        y = 1 + i * 0.8
        z = np.random.uniform(-1.5, 0.5)
        obstacles.append([x, y, z])
    
    # Add obstacles at different distances
    for _ in range(15):
        x = np.random.uniform(-15, 15)
        y = np.random.uniform(-15, 15)
        z = np.random.uniform(-3, 1)
        obstacles.append([x, y, z])
    
    print(f"Created {len(obstacles)} test obstacles")
    visualizer.update_obstacles(obstacles, voxel_size=0.5)
    
    # Test 4: Animate drone movement and build path
    print("Starting animation loop...")
    for i, pos in enumerate(drone_positions):
        print(f"Moving drone to position {i+1}/{len(drone_positions)}: {pos}")
        visualizer.update_drone_position(pos)
        time.sleep(1.0)
    
    # Test 5: Try to save a screenshot
    print("Attempting to save test screenshot...")
    screenshot_success = visualizer.save_screenshot("diagnostic_test.png")
    print(f"Screenshot save result: {screenshot_success}")
    
    # Test 6: Get visualization stats
    stats = visualizer.get_visualization_stats()
    print("Visualization stats:", stats)
    
    # Keep visualization running for inspection
    print("\nVisualization should now show:")
    print("- Blue sphere (drone) at final position [4, 3, -1]")
    print("- Red sphere (target) at [10, 5, -2]") 
    print("- Green line (drone path) connecting the drone positions")
    print("- Orange/yellow points/voxels (obstacles)")
    print("- Coordinate axes at origin (red=X, green=Y, blue=Z)")
    print("- Gray ground grid")
    print("\nPress keys in visualization window:")
    print("- R: Reset camera view")
    print("- S: Save screenshot") 
    print("- T: Toggle camera following")
    print("- H: Show help")
    print("\nVisualization will run for 30 seconds, then auto-close...")
    
    # Run for 30 seconds
    for remaining in range(30, 0, -1):
        print(f"\rTime remaining: {remaining:2d} seconds", end="", flush=True)
        time.sleep(1.0)
    
    print("\nStopping visualization...")
    visualizer.stop_visualization()
    print("Test complete!")

if __name__ == "__main__":
    try:
        test_visualization()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc() 