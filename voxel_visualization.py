"""
Voxel Grid Visualization Module using Open3D

This module provides real-time 3D visualization of the drone's environment using voxel grids,
showing obstacles, the drone's position, path history, and target location.
"""

import open3d as o3d
import numpy as np
import time
import threading
import logging
import traceback
import os
from typing import List, Tuple, Optional, Dict, Any
from collections import deque


class VoxelGridVisualizer:
    """Real-time 3D voxel grid visualization for drone navigation"""
    
    def __init__(self, voxel_size=0.5, grid_size=75, max_path_points=200, 
                 visualization_range=50.0, update_rate=8.0, enable_screenshots=True):
        """
        Initialize the voxel grid visualizer
        
        Args:
            voxel_size: Size of each voxel in meters
            grid_size: Number of voxels along each axis
            max_path_points: Maximum number of path points to keep in history
            visualization_range: Range around drone to visualize (meters)
            update_rate: Target update rate in Hz
            enable_screenshots: Whether to enable screenshot functionality
        """
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.max_path_points = max_path_points
        self.visualization_range = visualization_range
        self.update_rate = update_rate
        self.enable_screenshots = enable_screenshots
        
        # Visualization state
        self.vis = None
        self.is_running = False
        self.visualization_thread = None
        self.update_lock = threading.Lock()
        
        # Data storage
        self.drone_path = deque(maxlen=max_path_points)
        self.current_drone_pos = np.array([0.0, 0.0, 0.0])
        self.target_pos = np.array([0.0, 0.0, 0.0])
        self.obstacle_positions = []
        self.voxel_grid = None
        self.last_update_time = 0
        
        # Visualization objects - track geometries for camera calculations
        self.drone_sphere = None
        self.target_sphere = None
        self.path_line = None
        self.coordinate_frame = None
        self.ground_grid = None
        self.obstacles_pcd = None
        self.voxel_grid_vis = None
        self.orientation_frame = None  # For camera reset
        
        # Color scheme
        self.colors = {
            'drone': [0.0, 0.0, 1.0],          # Blue
            'target': [1.0, 0.0, 0.0],        # Red
            'path': [0.0, 1.0, 0.0],          # Green
            'obstacles': [0.8, 0.4, 0.0],     # Orange
            'voxels': [0.6, 0.6, 0.6],        # Gray
            'ground': [0.3, 0.3, 0.3],        # Dark gray
            'axes': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # RGB for XYZ
        }
        
        logging.info(f"Initialized VoxelGridVisualizer with voxel_size={voxel_size}m, "
                    f"grid_size={grid_size}, range={visualization_range}m")
    
    def start_visualization(self, window_name="Drone Navigation - Voxel Grid Visualization"):
        """Start the visualization in a separate thread"""
        if self.is_running:
            logging.warning("Visualization is already running")
            return
        
        self.is_running = True
        self.visualization_thread = threading.Thread(
            target=self._run_visualization,
            args=(window_name,),
            daemon=True
        )
        self.visualization_thread.start()
        
        # Wait a moment for the visualization to initialize
        time.sleep(1.0)
        logging.info("Started voxel grid visualization")
    
    def stop_visualization(self):
        """Stop the visualization"""
        self.is_running = False
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=2.0)
        
        if self.vis:
            self.vis.destroy_window()
            self.vis = None
        
        logging.info("Stopped voxel grid visualization")
    
    def _run_visualization(self, window_name):
        """Main visualization loop running in separate thread"""
        try:
            # Initialize Open3D visualizer with key callbacks
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(window_name=window_name, width=1200, height=800, visible=True)
            
            # Initialize screenshot capability first
            self._initialize_screenshot_capability()
            
            # Register key callbacks
            self._register_key_callbacks()
            
            # Set up the initial scene
            self._setup_initial_scene()
            
            # Configure camera and rendering options with optimal wide-view settings
            self._configure_visualization()
            
            # Main update loop
            last_update = 0
            update_interval = 1.0 / self.update_rate
            
            while self.is_running:
                current_time = time.time()
                
                if current_time - last_update >= update_interval:
                    with self.update_lock:
                        self._update_visualization()
                    last_update = current_time
                
                # Process visualization events
                self.vis.poll_events()
                self.vis.update_renderer()
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            logging.error(f"Error in visualization thread: {e}")
            traceback.print_exc()
        finally:
            if self.vis:
                self.vis.destroy_window()
    
    def _setup_initial_scene(self):
        """Set up the initial 3D scene with ground grid and coordinate axes"""
        try:
            # Add coordinate frame at origin with larger size for visibility
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=5.0, origin=[0, 0, 0]  # Increased size from 2.0 to 5.0
            )
            self.vis.add_geometry(self.coordinate_frame)
            
            # Create ground grid
            self._create_ground_grid()
            
            # Initialize drone representation - make it larger and more visible
            self.drone_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)  # Increased from 0.3
            self.drone_sphere.paint_uniform_color(self.colors['drone'])
            # Position at origin initially
            self.drone_sphere.translate([0, 0, 0])
            self.vis.add_geometry(self.drone_sphere)
            
            # Initialize target representation - make it larger and more visible
            self.target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.2)  # Increased from 0.4
            self.target_sphere.paint_uniform_color(self.colors['target'])
            # Position at a visible default location
            self.target_sphere.translate([10, 10, -2])
            self.vis.add_geometry(self.target_sphere)
            
            # Initialize path line
            self.path_line = o3d.geometry.LineSet()
            self.path_line.paint_uniform_color(self.colors['path'])
            self.vis.add_geometry(self.path_line)
            
            # Initialize obstacles point cloud
            self.obstacles_pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.obstacles_pcd)
            
            # Initialize voxel grid
            self.voxel_grid_vis = o3d.geometry.VoxelGrid()
            self.vis.add_geometry(self.voxel_grid_vis)
            
            # Add some test markers to show the visualization is working
            self._add_debug_markers()
            
            logging.info("Initial 3D scene setup complete with enhanced visibility")
            
        except Exception as e:
            logging.error(f"Error setting up initial scene: {e}")
            traceback.print_exc()
    
    def _add_debug_markers(self):
        """Add debug markers to show the scene is working"""
        try:
            # Add four corner markers to show scene bounds
            marker_positions = [
                [-10, -10, -1], [10, -10, -1], 
                [-10, 10, -1], [10, 10, -1]
            ]
            
            for i, pos in enumerate(marker_positions):
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
                marker.paint_uniform_color([0.8, 0.8, 0.0])  # Yellow markers
                marker.translate(pos)
                self.vis.add_geometry(marker)
                
            logging.debug("Added debug markers at scene corners")
            
        except Exception as e:
            logging.debug(f"Error adding debug markers: {e}")
    
    def _create_ground_grid(self):
        """Create a ground reference grid"""
        try:
            grid_range = min(self.visualization_range, 25.0)  # Cap at 25m for visibility
            grid_spacing = 2.0  # meters
            
            # Create grid lines
            lines = []
            points = []
            
            # Create grid lines parallel to X-axis
            for y in np.arange(-grid_range, grid_range + grid_spacing, grid_spacing):
                start_point = [-grid_range, y, 0]
                end_point = [grid_range, y, 0]
                points.extend([start_point, end_point])
                lines.append([len(points) - 2, len(points) - 1])
            
            # Create grid lines parallel to Y-axis  
            for x in np.arange(-grid_range, grid_range + grid_spacing, grid_spacing):
                start_point = [x, -grid_range, 0]
                end_point = [x, grid_range, 0]
                points.extend([start_point, end_point])
                lines.append([len(points) - 2, len(points) - 1])
            
            # Create ground grid line set
            self.ground_grid = o3d.geometry.LineSet()
            self.ground_grid.points = o3d.utility.Vector3dVector(points)
            self.ground_grid.lines = o3d.utility.Vector2iVector(lines)
            self.ground_grid.paint_uniform_color([0.4, 0.4, 0.4])  # Lighter gray for better visibility
            
            self.vis.add_geometry(self.ground_grid)
            
            logging.info(f"Created ground grid: {grid_range}m range, {grid_spacing}m spacing, {len(lines)} lines")
            
        except Exception as e:
            logging.error(f"Error creating ground grid: {e}")
    
    def _configure_visualization(self):
        """Configure camera and rendering options with optimal wide-view settings"""
        try:
            # Get render option and view control
            render_option = self.vis.get_render_option()
            view_control = self.vis.get_view_control()
            
            # Configure rendering for better visibility
            render_option.background_color = np.asarray([0.15, 0.15, 0.2])  # Lighter background
            render_option.point_size = 5.0
            render_option.line_width = 3.0
            render_option.show_coordinate_frame = True
            render_option.light_on = True  # Ensure lighting is enabled
            
            # Enable mesh shading for better visibility
            if hasattr(render_option, 'mesh_shade_option'):
                render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
            
            # CONSERVATIVE CAMERA CONFIGURATION - Start with default view
            # Use safe, conservative camera settings that ensure visibility
            
            # Set safe default camera position - moderate distance, slight angle
            view_control.set_front([0.3, -0.2, -0.9])         # Gentle angled view (not straight down)
            view_control.set_lookat([0.0, 0.0, -2.0])         # Look at origin/ground level
            view_control.set_up([0.0, 1.0, 0.0])              # Y-up for proper orientation
            view_control.set_zoom(0.6)                        # Conservative zoom (not too wide)
            
            # Try gentle camera positioning - if this fails, fallback to defaults
            try:
                view_control.camera_local_translate(
                    forward=-15.0,    # Moderate distance back (15 meters)
                    right=0.0,        # No lateral movement  
                    up=8.0           # Moderate elevation for perspective
                )
                logging.info("Applied conservative camera configuration: moderate angled view at 15m distance, 8m height")
            except Exception as e:
                logging.warning(f"Camera translate failed, using default positioning: {e}")
                # Fallback to just zoom adjustment if translate fails
                view_control.set_zoom(0.8)  # Even more conservative zoom
                view_control.set_front([0.0, 0.0, -1.0])  # Simple forward view
                view_control.set_lookat([0.0, 0.0, 0.0])  # Look at origin
            
            # Disable camera following initially - let user manually adjust if needed
            self.camera_follow_enabled = False
            
            # Force initial render with new camera settings
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # Wait a moment for camera to stabilize
            time.sleep(0.1)
            
            logging.info("Visualization configuration complete - conservative default view ready")
            
        except Exception as e:
            logging.error(f"Error configuring visualization: {e}")
            traceback.print_exc()
    
    def update_drone_position(self, position: List[float]):
        """Update the drone's current position"""
        with self.update_lock:
            self.current_drone_pos = np.array(position)
            self.drone_path.append(position.copy())
            self.last_update_time = time.time()
            logging.debug(f"Updated drone position to: {position}")
    
    def update_target_position(self, position: List[float]):
        """Update the target position"""
        with self.update_lock:
            self.target_pos = np.array(position)
            logging.debug(f"Updated target position to: {position}")
    
    def update_obstacles(self, obstacle_positions: List[List[float]], 
                        voxel_size: Optional[float] = None):
        """Update obstacle positions and create voxel grid"""
        with self.update_lock:
            self.obstacle_positions = obstacle_positions
            if voxel_size:
                self.voxel_size = voxel_size
            self._create_voxel_grid()
            logging.info(f"Updated obstacles: {len(obstacle_positions)} obstacles, voxel_size={self.voxel_size}")
    
    def _create_voxel_grid(self):
        """Create voxel grid from obstacle positions"""
        try:
            if not self.obstacle_positions:
                return
            
            # Create point cloud from obstacles
            obstacle_points = np.array(self.obstacle_positions)
            
            # Filter points within visualization range of drone
            drone_pos = self.current_drone_pos
            distances = np.linalg.norm(obstacle_points - drone_pos, axis=1)
            in_range_mask = distances <= self.visualization_range
            
            if not np.any(in_range_mask):
                return
            
            filtered_points = obstacle_points[in_range_mask]
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            
            # Create voxel grid from point cloud
            self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                pcd, voxel_size=self.voxel_size
            )
            
            # Color voxels based on density/distance
            self._color_voxels()
            
        except Exception as e:
            logging.debug(f"Error creating voxel grid: {e}")
    
    def _color_voxels(self):
        """Color voxels based on distance from drone and density"""
        try:
            if not self.voxel_grid or not hasattr(self.voxel_grid, 'get_voxels'):
                return
            
            voxels = self.voxel_grid.get_voxels()
            drone_pos = self.current_drone_pos
            
            for voxel in voxels:
                # Get voxel center position
                voxel_pos = np.array([voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]])
                voxel_world_pos = voxel_pos * self.voxel_size + self.voxel_grid.origin
                
                # Calculate distance to drone
                distance_to_drone = np.linalg.norm(voxel_world_pos - drone_pos)
                
                # Color based on distance (closer = more red, farther = more blue)
                max_dist = self.visualization_range
                normalized_dist = min(distance_to_drone / max_dist, 1.0)
                
                # Create color gradient from red (close) to blue (far)
                red_component = 1.0 - normalized_dist
                blue_component = normalized_dist
                green_component = 0.3
                
                voxel.color = [red_component, green_component, blue_component]
                
        except Exception as e:
            logging.debug(f"Error coloring voxels: {e}")
    
    def _update_visualization(self):
        """Update all visualization elements"""
        try:
            # Update drone position
            if self.drone_sphere:
                center = self.drone_sphere.get_center()
                translation = self.current_drone_pos - np.asarray(center)
                self.drone_sphere.translate(translation)
                self.vis.update_geometry(self.drone_sphere)
            
            # Update target position
            if self.target_sphere:
                center = self.target_sphere.get_center()
                translation = self.target_pos - np.asarray(center)
                self.target_sphere.translate(translation)
                self.vis.update_geometry(self.target_sphere)
            
            # Update path
            if len(self.drone_path) > 1 and self.path_line:
                path_points = list(self.drone_path)
                path_lines = [[i, i + 1] for i in range(len(path_points) - 1)]
                
                self.path_line.points = o3d.utility.Vector3dVector(path_points)
                self.path_line.lines = o3d.utility.Vector2iVector(path_lines)
                
                # Create gradient colors for path (older = darker, newer = brighter)
                path_colors = []
                for i in range(len(path_lines)):
                    intensity = (i + 1) / len(path_lines)  # 0 to 1
                    color = [0, intensity, 0]  # Green gradient
                    path_colors.append(color)
                
                self.path_line.colors = o3d.utility.Vector3dVector(path_colors)
                self.vis.update_geometry(self.path_line)
            
            # Update obstacles point cloud
            if self.obstacle_positions and self.obstacles_pcd:
                obstacle_points = np.array(self.obstacle_positions)
                
                # Filter points within visualization range
                drone_pos = self.current_drone_pos
                distances = np.linalg.norm(obstacle_points - drone_pos, axis=1)
                in_range_mask = distances <= self.visualization_range
                
                if np.any(in_range_mask):
                    filtered_points = obstacle_points[in_range_mask]
                    filtered_distances = distances[in_range_mask]
                    
                    self.obstacles_pcd.points = o3d.utility.Vector3dVector(filtered_points)
                    
                    # Color obstacles based on distance
                    colors = []
                    max_dist = self.visualization_range
                    for dist in filtered_distances:
                        normalized_dist = min(dist / max_dist, 1.0)
                        # Orange to yellow gradient based on distance
                        red = 1.0
                        green = 0.4 + 0.6 * (1.0 - normalized_dist)
                        blue = 0.0
                        colors.append([red, green, blue])
                    
                    self.obstacles_pcd.colors = o3d.utility.Vector3dVector(colors)
                    self.vis.update_geometry(self.obstacles_pcd)
            
            # Update voxel grid
            if self.voxel_grid and self.voxel_grid_vis:
                # Remove old voxel grid and add new one
                self.vis.remove_geometry(self.voxel_grid_vis, reset_bounding_box=False)
                self.voxel_grid_vis = self.voxel_grid
                self.vis.add_geometry(self.voxel_grid_vis, reset_bounding_box=False)
            
            # Dynamically adjust camera to follow drone
            self._update_camera_view()
            
        except Exception as e:
            logging.debug(f"Error updating visualization: {e}")
    
    def _update_camera_view(self):
        """Update camera to follow the drone while maintaining good overview"""
        try:
            if not self.vis:
                return
            
            view_control = self.vis.get_view_control()
            
            # Simple camera following - keep the working approach but with proper distance
            if len(self.drone_path) > 0:
                # Look at current drone position with smooth following
                current_lookat = np.array(view_control.get_lookat())
                target_lookat = self.current_drone_pos
                
                # Smooth interpolation to follow drone
                interpolation_factor = 0.05
                new_lookat = current_lookat + interpolation_factor * (target_lookat - current_lookat)
                
                view_control.set_lookat(new_lookat)
                # Keep the working camera settings with proper distance
                view_control.set_front([0.2, 0.2, -1.0])
                view_control.set_up([0, 1, 0])
                
                # Maintain camera distance for wide view - translate back if needed
                # This ensures we stay at a good viewing distance as we follow the drone
                current_params = view_control.convert_to_pinhole_camera_parameters()
                camera_position = current_params.extrinsic[:3, 3]
                lookat_position = new_lookat
                
                # Calculate distance from camera to lookat point
                distance = np.linalg.norm(camera_position - lookat_position)
                
                # If camera is too close (less than desired distance), move it back
                desired_distance = 60.0  # meters - good distance for wide overview
                if distance < desired_distance:
                    # Calculate how much to move back
                    move_back = desired_distance - distance
                    view_control.camera_local_translate(forward=-move_back, right=0.0, up=0.0)
            
        except Exception as e:
            logging.debug(f"Error updating camera view: {e}")
    
    def _initialize_screenshot_capability(self):
        """Initialize and test screenshot capability"""
        try:
            if not self.enable_screenshots:
                logging.info("Screenshots disabled by configuration")
                return
                
            # Force initial render and allow window to stabilize
            time.sleep(0.2)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.1)
            
            # Test screenshot capability
            test_filename = "test_screenshot_capability.png"
            try:
                self.vis.capture_screen_image(test_filename)
                
                # Verify the file was created and has reasonable size
                if os.path.exists(test_filename):
                    file_size = os.path.getsize(test_filename)
                    if file_size > 1024:  # At least 1KB
                        logging.info(f"Screenshot capability confirmed - test file: {file_size} bytes")
                        os.remove(test_filename)  # Clean up test file
                        return
                    else:
                        logging.warning(f"Test screenshot file too small: {file_size} bytes")
                else:
                    logging.warning("Test screenshot file was not created")
                    
            except Exception as e:
                logging.warning(f"Screenshot capability test failed: {e}")
                
        except Exception as e:
            logging.error(f"Error initializing screenshot capability: {e}")
    
    def save_screenshot(self, filename: str) -> bool:
        """Save a screenshot of the current visualization with enhanced error handling"""
        try:
            if not self.vis or not self.enable_screenshots:
                logging.debug("Screenshot not available - visualizer not initialized or screenshots disabled")
                return False
                
            # Enhanced rendering preparation
            for _ in range(3):  # Multiple render cycles for stability
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.05)  # Allow time for rendering to complete
            
            # Additional stabilization
            time.sleep(0.1)
            
            # Capture screenshot
            self.vis.capture_screen_image(filename)
            
            # Verify file was created and has reasonable size
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                if file_size > 1024:  # At least 1KB for a valid image
                    logging.info(f"Saved visualization screenshot: {filename} ({file_size} bytes)")
                    return True
                else:
                    logging.warning(f"Screenshot file created but too small: {filename} ({file_size} bytes)")
                    return False
            else:
                logging.error(f"Screenshot file was not created: {filename}")
                return False
                
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")
            traceback.print_exc()
            return False
    
    def toggle_camera_follow(self):
        """Toggle camera following mode"""
        if hasattr(self, 'camera_follow_enabled'):
            self.camera_follow_enabled = not self.camera_follow_enabled
            logging.info(f"Camera follow mode: {'ON' if self.camera_follow_enabled else 'OFF'}")
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """Get current visualization statistics"""
        return {
            'is_running': self.is_running,
            'path_points': len(self.drone_path),
            'obstacle_count': len(self.obstacle_positions),
            'voxel_size': self.voxel_size,
            'visualization_range': self.visualization_range,
            'drone_position': self.current_drone_pos.tolist(),
            'target_position': self.target_pos.tolist(),
            'last_update': self.last_update_time
        }
    
    def reset_camera_view(self):
        """
        Reset camera to optimal bird's eye view showing all scene geometry.
        Centers the camera on all geometries with proper zoom and orientation.
        """
        try:
            if not self.vis:
                logging.warning("Visualizer not initialized - cannot reset camera")
                return
            
            view_control = self.vis.get_view_control()
            
            # Collect all meaningful points in the scene for bounding box calculation
            all_points = []
            
            # Add drone position
            if hasattr(self, 'current_drone_pos') and self.current_drone_pos is not None:
                all_points.append(self.current_drone_pos)
            
            # Add target position
            if hasattr(self, 'target_pos') and self.target_pos is not None:
                all_points.append(self.target_pos)
            
            # Add path points
            if hasattr(self, 'drone_path') and len(self.drone_path) > 0:
                all_points.extend(list(self.drone_path))
            
            # Add obstacle positions
            if hasattr(self, 'obstacle_positions') and self.obstacle_positions:
                all_points.extend(self.obstacle_positions)
            
            # If we have no points, use default scene
            if not all_points:
                all_points = [
                    [-20, -20, -10], [20, 20, 10],  # Default bounding box
                    [0, 0, 0]  # Origin
                ]
            
            # Convert to numpy array and compute scene bounds
            points_array = np.array(all_points)
            scene_min = np.min(points_array, axis=0)
            scene_max = np.max(points_array, axis=0)
            scene_center = (scene_min + scene_max) / 2.0
            scene_extent = scene_max - scene_min
            
            # Ensure minimum scene size for very small scenes
            min_extent = 10.0
            for i in range(3):
                if scene_extent[i] < min_extent:
                    padding = (min_extent - scene_extent[i]) / 2.0
                    scene_min[i] -= padding
                    scene_max[i] += padding
                    scene_extent[i] = min_extent
            
            # Recalculate center after padding
            scene_center = (scene_min + scene_max) / 2.0
            diagonal_length = np.linalg.norm(scene_extent)
            
            # Set camera to bird's eye view
            view_control.set_lookat(scene_center.tolist())
            view_control.set_up([0, 1, 0])  # Y-up for proper orientation
            
            # Set camera orientation - diagonal top-down view
            # Front vector points from camera to lookat - negative means camera looks down
            view_control.set_front([0.3, -0.5, -0.8])  # Slightly angled bird's eye view
            
            # Set appropriate zoom - larger values = wider view
            # Start with a conservative zoom that shows everything
            view_control.set_zoom(0.3)
            
            # Move camera back to ensure everything is visible
            # Use diagonal length to determine appropriate distance
            camera_distance = diagonal_length * 1.8  # 1.8x diagonal for safe margin
            
            # Move camera backward (negative forward value)
            try:
                view_control.camera_local_translate(
                    forward=-camera_distance, 
                    right=0.0, 
                    up=camera_distance * 0.3  # Slightly elevated
                )
            except Exception as e:
                logging.debug(f"Camera translate failed, using alternative method: {e}")
                # Alternative: directly set zoom to achieve wide view
                view_control.set_zoom(0.1)  # Very wide zoom as fallback
            
            # Add/update orientation coordinate frame at scene center
            self._add_orientation_frame(scene_center)
            
            # Force render update
            self.vis.poll_events()
            self.vis.update_renderer()
            
            logging.info(f"Camera reset - Center: {scene_center}, "
                        f"Extent: {scene_extent}, "
                        f"Diagonal: {diagonal_length:.1f}m, "
                        f"Camera distance: {camera_distance:.1f}m")
            
        except Exception as e:
            logging.error(f"Error resetting camera view: {e}")
            traceback.print_exc()
    
    def _add_orientation_frame(self, position):
        """Add or update coordinate frame at specified position for orientation reference"""
        try:
            # Remove existing orientation frame if present
            if self.orientation_frame:
                self.vis.remove_geometry(self.orientation_frame, reset_bounding_box=False)
            
            # Create new coordinate frame at scene center
            frame_size = max(5.0, np.linalg.norm(self.current_drone_pos - position) * 0.1)
            self.orientation_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=frame_size, 
                origin=position.tolist()
            )
            
            # Add to scene
            self.vis.add_geometry(self.orientation_frame, reset_bounding_box=False)
            
        except Exception as e:
            logging.debug(f"Error adding orientation frame: {e}")
    
    def _register_key_callbacks(self):
        """Register keyboard shortcuts for visualization control"""
        try:
            # R key - Reset camera view
            self.vis.register_key_callback(ord("R"), 
                lambda vis: self.reset_camera_view())
            
            # C key - Toggle camera following
            self.vis.register_key_callback(ord("C"), 
                lambda vis: self.toggle_camera_follow())
            
            # S key - Save screenshot  
            self.vis.register_key_callback(ord("S"), 
                lambda vis: self._save_screenshot_callback())
            
            # H key - Show help
            self.vis.register_key_callback(ord("H"), 
                lambda vis: self._show_help())
            
            logging.info("Registered key callbacks: R(reset camera), C(toggle follow), S(screenshot), H(help)")
            
        except Exception as e:
            logging.error(f"Error registering key callbacks: {e}")
    
    def _save_screenshot_callback(self):
        """Callback for screenshot key press"""
        timestamp = int(time.time())
        filename = f"visualization_screenshot_{timestamp}.png"
        success = self.save_screenshot(filename)
        if success:
            logging.info(f"Screenshot saved: {filename}")
        else:
            logging.warning("Screenshot failed")
    
    def _show_help(self):
        """Display help information about key controls"""
        help_text = """
        Visualization Controls:
        R - Reset camera to bird's eye view
        C - Toggle camera following drone
        S - Save screenshot
        H - Show this help
        Mouse - Rotate, zoom, pan view
        """
        print(help_text)
        logging.info("Help displayed")


def create_voxel_visualizer(voxel_size=0.5, visualization_range=25.0, auto_start=True, enable_screenshots=True):
    """
    Create and optionally start a voxel grid visualizer
    
    Args:
        voxel_size: Size of voxels in meters
        visualization_range: Range to visualize around drone
        auto_start: Whether to automatically start visualization
        enable_screenshots: Whether to enable screenshot functionality
    
    Returns:
        VoxelGridVisualizer instance
    """
    visualizer = VoxelGridVisualizer(
        voxel_size=voxel_size,
        visualization_range=visualization_range,
        enable_screenshots=enable_screenshots
    )
    
    if auto_start:
        visualizer.start_visualization()
    
    return visualizer


def reset_camera_view(vis: o3d.visualization.Visualizer, geometries=None, 
                     camera_height_factor=1.5, zoom_factor=0.3, add_coord_frame=True):
    """
    Reset camera to optimal bird's eye view showing all scene geometry.
    
    This function works with any Open3D visualizer and automatically computes
    the best camera position to show all geometries in the scene.
    
    Args:
        vis: Open3D Visualizer instance
        geometries: List of geometries to consider for bounding box calculation.
                   If None, attempts to estimate from scene or uses default bounds.
        camera_height_factor: Multiplier for camera height above scene (default: 1.5)
        zoom_factor: Zoom level (smaller = more zoomed in, larger = more zoomed out)
        add_coord_frame: Whether to add coordinate frame at scene center
    
    Returns:
        dict: Information about the camera setup (center, extent, distance)
    """
    try:
        view_control = vis.get_view_control()
        
        # Collect points for bounding box calculation
        all_points = []
        
        if geometries:
            # Use provided geometries
            for geom in geometries:
                if hasattr(geom, 'get_axis_aligned_bounding_box'):
                    bbox = geom.get_axis_aligned_bounding_box()
                    # Add bbox corners to points
                    min_bound = np.asarray(bbox.min_bound)
                    max_bound = np.asarray(bbox.max_bound)
                    all_points.extend([min_bound, max_bound])
                elif hasattr(geom, 'points'):
                    # Point cloud
                    points = np.asarray(geom.points)
                    if len(points) > 0:
                        all_points.extend(points)
        
        # If no points collected, use reasonable defaults
        if not all_points:
            all_points = [
                [-25, -25, -15], [25, 25, 15],  # Default scene bounds
                [0, 0, 0]  # Origin
            ]
            logging.info("Using default scene bounds for camera reset")
        
        # Compute scene bounding box
        points_array = np.array(all_points)
        scene_min = np.min(points_array, axis=0)
        scene_max = np.max(points_array, axis=0)
        scene_center = (scene_min + scene_max) / 2.0
        scene_extent = scene_max - scene_min
        
        # Ensure minimum scene size
        min_extent = 10.0
        for i in range(3):
            if scene_extent[i] < min_extent:
                padding = (min_extent - scene_extent[i]) / 2.0
                scene_min[i] -= padding
                scene_max[i] += padding
                scene_extent[i] = min_extent
        
        # Recalculate after padding
        scene_center = (scene_min + scene_max) / 2.0
        diagonal_length = np.linalg.norm(scene_extent)
        
        # Set camera position and orientation
        view_control.set_lookat(scene_center.tolist())
        view_control.set_up([0, 1, 0])  # Y-up convention
        
        # Bird's eye view with slight angle for depth perception
        view_control.set_front([0.2, -0.4, -0.9])  # Angled downward view
        
        # Set zoom
        view_control.set_zoom(zoom_factor)
        
        # Calculate camera distance and position
        camera_distance = diagonal_length * camera_height_factor
        camera_height = camera_distance * 0.4  # Elevated view
        
        # Move camera to optimal position
        try:
            view_control.camera_local_translate(
                forward=-camera_distance,
                right=0.0,
                up=camera_height
            )
        except Exception as e:
            logging.debug(f"Camera translate failed: {e}")
            # Fallback: adjust zoom instead
            view_control.set_zoom(zoom_factor * 0.5)
        
        # Add coordinate frame for orientation
        if add_coord_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=max(3.0, diagonal_length * 0.1),
                origin=scene_center.tolist()
            )
            vis.add_geometry(coord_frame, reset_bounding_box=False)
        
        # Force render update
        vis.poll_events()
        vis.update_renderer()
        
        camera_info = {
            'center': scene_center.tolist(),
            'extent': scene_extent.tolist(),
            'diagonal_length': diagonal_length,
            'camera_distance': camera_distance
        }
        
        logging.info(f"Camera reset completed - Center: {scene_center}, "
                    f"Extent: {scene_extent}, Diagonal: {diagonal_length:.1f}m")
        
        return camera_info
        
    except Exception as e:
        logging.error(f"Error in reset_camera_view: {e}")
        traceback.print_exc()
        return None


def setup_visualization_key_callbacks(vis):
    """
    Set up standard key callbacks for visualization control.
    
    Args:
        vis: VisualizerWithKeyCallback instance
    
    Key bindings:
        R - Reset camera view
        H - Show help
        ESC - Exit visualization
    """
    try:
        if not isinstance(vis, o3d.visualization.VisualizerWithKeyCallback):
            logging.warning("Visualizer must be VisualizerWithKeyCallback for key callbacks")
            return
        
        # R key - Reset camera
        vis.register_key_callback(ord("R"), 
            lambda vis: reset_camera_view(vis))
        
        # H key - Show help
        def show_help(vis):
            help_text = """
            Visualization Controls:
            R - Reset camera to bird's eye view
            H - Show this help
            ESC - Exit visualization
            Mouse - Rotate, zoom, pan view
            """
            print(help_text)
            return False
        
        vis.register_key_callback(ord("H"), show_help)
        
        # ESC key - Close visualization
        vis.register_key_callback(256, lambda vis: vis.close())  # ESC key code
        
        logging.info("Key callbacks registered: R(reset), H(help), ESC(exit)")
        
    except Exception as e:
        logging.error(f"Error setting up key callbacks: {e}")


def create_visualization_with_auto_camera(window_name="Open3D Visualization", 
                                        width=1200, height=800):
    """
    Create a visualization window with automatic camera reset functionality.
    
    Returns:
        Configured VisualizerWithKeyCallback instance
    """
    try:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=window_name, width=width, height=height)
        
        # Set up key callbacks
        setup_visualization_key_callbacks(vis)
        
        # Configure render options
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([0.1, 0.1, 0.15])
        render_option.point_size = 3.0
        render_option.line_width = 2.0
        render_option.show_coordinate_frame = True
        
        logging.info(f"Created visualization window: {window_name}")
        return vis
        
    except Exception as e:
        logging.error(f"Error creating visualization: {e}")
        return None


if __name__ == "__main__":
    # Demo/test visualization with enhanced error handling
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create visualizer
        viz = create_voxel_visualizer(voxel_size=0.3, visualization_range=30.0)
        
        # Wait for visualization to start
        time.sleep(2.0)
        
        # Simulate some data updates
        import random
        
        # Initial positions to ensure something is visible
        viz.update_drone_position([0, 0, -5])
        viz.update_target_position([10, 0, -5])
        
        # Add some initial obstacles
        initial_obstacles = []
        for i in range(30):
            obs_pos = [
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-8, -2)
            ]
            initial_obstacles.append(obs_pos)
        viz.update_obstacles(initial_obstacles)
        
        # Run for demonstration
        time.sleep(17.0)
        
        # Clean shutdown
        viz.stop_visualization()
        
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        traceback.print_exc()
