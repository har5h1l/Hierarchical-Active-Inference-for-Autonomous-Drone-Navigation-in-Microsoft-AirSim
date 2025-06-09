#!/usr/bin/env python3
"""
Test script to debug camera positioning in Open3D visualization
"""

import open3d as o3d
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_camera_positioning():
    """Test different camera positions and save screenshots"""
    
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Camera Position Test", width=800, height=600, visible=True)
    
    # Create test scene
    print("Creating test scene...")
    
    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    
    # Add drone sphere (blue)
    drone_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    drone_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    drone_sphere.translate([0, 0, -5])  # Position at -5 on Z
    vis.add_geometry(drone_sphere)
    
    # Add target sphere (red)
    target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    target_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    target_sphere.translate([10, 10, -5])  # Position away from drone
    vis.add_geometry(target_sphere)
    
    # Add ground grid
    grid_points = []
    grid_lines = []
    for x in range(-20, 21, 2):
        # Lines parallel to Y axis
        grid_points.extend([[x, -20, 0], [x, 20, 0]])
        grid_lines.append([len(grid_points)-2, len(grid_points)-1])
        
    for y in range(-20, 21, 2):
        # Lines parallel to X axis  
        grid_points.extend([[-20, y, 0], [20, y, 0]])
        grid_lines.append([len(grid_points)-2, len(grid_points)-1])
    
    ground_grid = o3d.geometry.LineSet()
    ground_grid.points = o3d.utility.Vector3dVector(grid_points)
    ground_grid.lines = o3d.utility.Vector2iVector(grid_lines)
    ground_grid.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
    vis.add_geometry(ground_grid)
    
    # Add some obstacle voxels
    obstacles = []
    for i in range(20):
        x = np.random.uniform(-15, 15)
        y = np.random.uniform(-15, 15)
        z = np.random.uniform(-8, -2)
        obstacles.append([x, y, z])
    
    # Create point cloud from obstacles
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obstacles)
    pcd.paint_uniform_color([1.0, 0.5, 0.0])  # Orange
    vis.add_geometry(pcd)
    
    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
    vis.add_geometry(voxel_grid)
    
    print("Scene created. Testing camera positions...")
    
    # Get view control
    view_control = vis.get_view_control()
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.1, 0.1, 0.15])
    
    # Test Case 6: Ultra wide overhead view
    print("Test 6: Ultra wide overhead view")
    view_control.set_front([0, 0, -1])  # Straight down
    view_control.set_lookat([5, 5, -2.5])  # Look at middle of scene
    view_control.set_up([0, 1, 0])  # Y up
    view_control.set_zoom(0.01)  # Ultra wide zoom
    
    time.sleep(0.5)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.5)
    vis.capture_screen_image("test6_ultra_wide.png")
    print("Saved test6_ultra_wide.png")
    
    # Test Case 7: High angled view with ultra wide zoom
    print("Test 7: High angled view with ultra wide zoom")
    view_control.set_front([0.1, 0.1, -0.99])  # Almost straight down but slight angle
    view_control.set_lookat([5, 5, -2.5])  # Center of scene
    view_control.set_up([0, 1, 0])  # Y up
    view_control.set_zoom(0.005)  # Extremely wide
    
    time.sleep(0.5)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.5)
    vis.capture_screen_image("test7_high_angle_ultra_wide.png")
    print("Saved test7_high_angle_ultra_wide.png")
    
    # Test Case 8: Medium height with wide view
    print("Test 8: Medium height with wide view")
    view_control.set_front([0.2, 0.2, -0.96])  # Moderate angle down
    view_control.set_lookat([5, 5, -2.5])  # Center of scene
    view_control.set_up([0, 1, 0])  # Y up
    view_control.set_zoom(0.015)  # Very wide
    
    time.sleep(0.5)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.5)
    vis.capture_screen_image("test8_medium_height_wide.png")
    print("Saved test8_medium_height_wide.png")
    
    # Test Case 9: Using camera_local_translate for positioning
    print("Test 9: Using camera translate for wide view")
    view_control.set_front([0.1, 0.1, -0.99])
    view_control.set_lookat([5, 5, -2.5])
    view_control.set_up([0, 1, 0])
    view_control.set_zoom(0.1)
    
    # Move camera way back
    try:
        view_control.camera_local_translate(forward=-50.0, right=0.0, up=20.0)
    except:
        print("Camera translate failed, using zoom fallback")
        view_control.set_zoom(0.005)
    
    time.sleep(0.5)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.5)
    vis.capture_screen_image("test9_camera_translate.png")
    print("Saved test9_camera_translate.png")
    
    # Test Case 10: Scene-computed optimal view
    print("Test 10: Scene-computed optimal view")
    
    # Compute scene bounds
    all_points = [
        [0, 0, -5],      # Drone
        [10, 10, -5],    # Target
        [-20, -20, 0],   # Grid bounds
        [20, 20, 0],     # Grid bounds
        [-15, -15, -8],  # Obstacle bounds
        [15, 15, -2]     # Obstacle bounds
    ]
    
    points_array = np.array(all_points)
    scene_min = np.min(points_array, axis=0)
    scene_max = np.max(points_array, axis=0)
    scene_center = (scene_min + scene_max) / 2.0
    scene_extent = scene_max - scene_min
    diagonal_length = np.linalg.norm(scene_extent)
    
    print(f"Scene center: {scene_center}")
    print(f"Scene extent: {scene_extent}")
    print(f"Diagonal length: {diagonal_length}")
    
    # Set optimal camera position
    view_control.set_front([0.1, 0.1, -0.99])  # Slight angle for depth
    view_control.set_lookat(scene_center.tolist())
    view_control.set_up([0, 1, 0])
    
    # Calculate zoom based on scene size
    optimal_zoom = 0.02 / (diagonal_length / 60.0)  # Adjusted for scene size
    print(f"Optimal zoom calculated: {optimal_zoom}")
    view_control.set_zoom(optimal_zoom)
    
    time.sleep(0.5)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.5)
    vis.capture_screen_image("test10_optimal_computed.png")
    print("Saved test10_optimal_computed.png")
    
    print("All camera tests complete!")
    print("Check the generated PNG files to see which view works best.")
    print("Press ESC to close the visualization window.")
    
    # Keep window open to manually inspect
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    test_camera_positioning() 