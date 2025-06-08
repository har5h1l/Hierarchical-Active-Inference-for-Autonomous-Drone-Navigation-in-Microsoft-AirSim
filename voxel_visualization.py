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
        
        # Visualization objects
        self.drone_sphere = None
        self.target_sphere = None
        self.path_line = None
        self.coordinate_frame = None
        self.ground_grid = None
        self.obstacles_pcd = None
        self.voxel_grid_vis = None
        
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
            # Initialize Open3D visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=window_name, width=1200, height=800)
            
            # Set up the initial scene
            self._setup_initial_scene()
            
            # Configure camera and rendering options
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
            # Add coordinate frame at origin
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=2.0, origin=[0, 0, 0]
            )
            self.vis.add_geometry(self.coordinate_frame)
            
            # Create ground grid
            self._create_ground_grid()
            
            # Initialize drone representation
            self.drone_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            self.drone_sphere.paint_uniform_color(self.colors['drone'])
            self.vis.add_geometry(self.drone_sphere)
            
            # Initialize target representation
            self.target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
            self.target_sphere.paint_uniform_color(self.colors['target'])
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
            
            logging.debug("Initial 3D scene setup complete")
            
        except Exception as e:
            logging.error(f"Error setting up initial scene: {e}")
            traceback.print_exc()
    
    def _create_ground_grid(self):
        """Create a ground reference grid"""
        try:
            grid_range = self.visualization_range
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
            self.ground_grid.paint_uniform_color(self.colors['ground'])
            
            self.vis.add_geometry(self.ground_grid)
            
        except Exception as e:
            logging.error(f"Error creating ground grid: {e}")
    
    def _configure_visualization(self):
        """Configure camera and rendering options"""
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
            
            # Set initial camera position (bird's eye view) - PROPER DISTANCE VERSION
            view_control.set_zoom(0.5)  # Use reasonable zoom value
            view_control.set_front([0.2, 0.2, -1.0])  # Slightly angled view
            view_control.set_lookat([0, 0, -10])       # Look at higher altitude for better overview
            view_control.set_up([0, 1, 0])            # Y is up
            
            # Now use camera_local_translate to move camera further back for wide view
            # Negative forward value moves camera backwards (further away)
            view_control.camera_local_translate(forward=-50.0, right=0.0, up=0.0)
            
            # Force initial render
            self.vis.poll_events()
            self.vis.update_renderer()
            
            logging.debug("Visualization configuration complete")
            
        except Exception as e:
            logging.error(f"Error configuring visualization: {e}")
    
    def update_drone_position(self, position: List[float]):
        """Update the drone's current position"""
        with self.update_lock:
            self.current_drone_pos = np.array(position)
            self.drone_path.append(position.copy())
            self.last_update_time = time.time()
    
    def update_target_position(self, position: List[float]):
        """Update the target position"""
        with self.update_lock:
            self.target_pos = np.array(position)
    
    def update_obstacles(self, obstacle_positions: List[List[float]], 
                        voxel_size: Optional[float] = None):
        """Update obstacle positions and create voxel grid"""
        with self.update_lock:
            self.obstacle_positions = obstacle_positions
            if voxel_size:
                self.voxel_size = voxel_size
            self._create_voxel_grid()
    
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
    
    def save_screenshot(self, filename: str) -> bool:
        """Save a screenshot of the current visualization"""
        try:
            if self.vis and self.enable_screenshots:
                # Ensure the window is rendered before capturing
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.1)  # Small delay to ensure rendering is complete
                
                self.vis.capture_screen_image(filename)
                logging.info(f"Saved visualization screenshot to {filename}")
                return True
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")
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


def create_voxel_visualizer(voxel_size=0.5, visualization_range=25.0, auto_start=True, enable_screenshots=True):
    """
    Create and optionally start a voxel grid visualizer
    
    Args:
        voxel_size: Size of voxels in meters
        visualization_range: Range around drone to visualize
        auto_start: Whether to automatically start the visualization
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
