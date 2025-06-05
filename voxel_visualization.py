"""
Voxel Grid Visualization Module using Open3D

This module provides real-time 3D visualization of the drone's environment using voxel grids,
showing obstacles, the drone's position, path history, and target location.

This is now a wrapper that uses the robust visualization implementation to handle OpenGL context failures.
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

# Import the robust visualization system
try:
    from voxel_visualization_robust import RobustVoxelGridVisualizer
    ROBUST_AVAILABLE = True
    logging.info("Using robust voxel visualization system")
except ImportError:
    ROBUST_AVAILABLE = False
    logging.warning("Robust visualization not available, using original implementation")

# Optional imports for enhanced screenshot functionality
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available - some screenshot methods may not work")


class VoxelGridVisualizer:
    """Real-time 3D voxel grid visualization for drone navigation"""
    
    def __init__(self, voxel_size=0.5, grid_size=50, max_path_points=200, 
                 visualization_range=25.0, update_rate=10.0):
        """
        Initialize the voxel grid visualizer
        
        Args:
            voxel_size: Size of each voxel in meters
            grid_size: Number of voxels along each axis
            max_path_points: Maximum number of path points to keep in history
            visualization_range: Range around drone to visualize (meters)
            update_rate: Target update rate in Hz
        """
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.max_path_points = max_path_points
        self.visualization_range = visualization_range
        self.update_rate = update_rate
        
        # Use robust implementation if available
        if ROBUST_AVAILABLE:
            logging.info("Using robust visualization implementation")
            self._robust_viz = RobustVoxelGridVisualizer(voxel_size, visualization_range)
            self._use_robust = True
        else:
            logging.warning("Using fallback implementation")
            self._use_robust = False
            self._initialize_fallback()
        
        logging.info(f"Initialized VoxelGridVisualizer with voxel_size={voxel_size}m, "
                    f"grid_size={grid_size}, range={visualization_range}m")
    
    def _initialize_fallback(self):
        """Initialize fallback visualization components"""
        # Visualization state
        self.vis = None
        self.is_running = False
        self.visualization_thread = None
        self.update_lock = threading.Lock()
        
        # Data storage
        self.drone_path = deque(maxlen=self.max_path_points)
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
    
    def start_visualization(self, window_name="Drone Navigation - Voxel Grid Visualization"):
        """Start the visualization in a separate thread"""
        if self._use_robust:
            return self._robust_viz.start_visualization(window_name)
        
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
        if self._use_robust:
            return self._robust_viz.stop_visualization()
        
        # Fallback implementation
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
            
            # Set initial camera position (bird's eye view)
            view_control.set_zoom(0.4)
            view_control.set_front([0.2, 0.2, -1.0])  # Slightly angled view
            view_control.set_lookat([0, 0, -3])       # Look at drone level
            view_control.set_up([0, 1, 0])            # Y is up
            
            # Force initial render
            self.vis.poll_events()
            self.vis.update_renderer()
            
            logging.debug("Visualization configuration complete")
            
        except Exception as e:
            logging.error(f"Error configuring visualization: {e}")
    
    def update_obstacles(self, obstacles):
        """Update obstacles in the visualization"""
        if self._use_robust:
            return self._robust_viz.update_obstacles(obstacles)
        
        # Fallback implementation
        with self.update_lock:
            self.obstacle_positions = obstacles.copy() if obstacles else []
    
    def update_drone_position(self, position):
        """Update drone position"""
        if self._use_robust:
            return self._robust_viz.update_drone_position(position)
        
        # Fallback implementation
        with self.update_lock:
            self.current_drone_pos = np.array(position)
            self.drone_path.append(position.copy())
    
    def update_target_position(self, position):
        """Update target position"""
        if self._use_robust:
            return self._robust_viz.update_target_position(position)
        
        # Fallback implementation
        with self.update_lock:
            self.target_pos = np.array(position)
    
    def save_screenshot(self, filepath):
        """Save screenshot using robust fallback methods"""
        if self._use_robust:
            return self._robust_viz.save_screenshot(filepath)
        
        # Fallback implementation
        logging.warning("Using fallback screenshot method - basic Open3D capture")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if self.vis:
                return self.vis.capture_screen_image(filepath)
            else:
                logging.error("Visualization not initialized for screenshot")
                return False
        except Exception as e:
            logging.error(f"Fallback screenshot failed: {e}")
            return False
    
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
                self.drone_sphere.translate(translation, relative=False)
                self.vis.update_geometry(self.drone_sphere)
            
            # Update target position
            if self.target_sphere:
                center = self.target_sphere.get_center()
                translation = self.target_pos - np.asarray(center)
                self.target_sphere.translate(translation, relative=False)
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
            if not hasattr(self, 'camera_follow_enabled'):
                self.camera_follow_enabled = True
            
            if self.camera_follow_enabled:
                view_control = self.vis.get_view_control()
                
                # Calculate new camera position (follow drone with offset)
                drone_pos = self.current_drone_pos
                
                # Look at point slightly ahead of drone
                look_at_pos = drone_pos + np.array([2, 0, 0])
                
                # Update camera smoothly using compatible API
                try:
                    # Use the newer API if available
                    camera_params = view_control.convert_to_pinhole_camera_parameters()
                    
                    # Smoothly interpolate to new look-at position
                    if hasattr(self, '_last_lookat'):
                        new_lookat = 0.9 * self._last_lookat + 0.1 * look_at_pos
                    else:
                        new_lookat = look_at_pos
                    
                    self._last_lookat = new_lookat
                    view_control.set_lookat(new_lookat)
                    
                except AttributeError:
                    # Fallback for older Open3D versions
                    try:
                        current_lookat = view_control.get_lookat()
                        new_lookat = 0.9 * np.array(current_lookat) + 0.1 * look_at_pos
                        view_control.set_lookat(new_lookat)
                    except:
                        pass  # Skip camera updates if not supported
        except Exception as e:
            logging.debug(f"Error updating camera view: {e}")
    
    def save_screenshot(self, filename: str):
        """Save a screenshot of the current visualization with enhanced OpenGL context handling"""
        try:
            if not self.vis or not hasattr(self.vis, 'capture_screen_image'):
                logging.error("Visualizer not available or doesn't support screenshot capture")
                return False
            
            # Ensure the visualization is running and window is valid
            if not self.is_running:
                logging.error("Visualization is not running")
                return False
            
            max_attempts = 5
            success = False
            
            for attempt in range(max_attempts):
                try:
                    logging.debug(f"Screenshot attempt {attempt + 1}/{max_attempts}")
                    
                    # Force window to foreground and ensure it's visible
                    try:
                        # Try to bring window to front (may not work in all environments)
                        if hasattr(self.vis, 'get_window_name'):
                            logging.debug("Attempting to bring window to foreground")
                    except:
                        pass
                    
                    # Extensive rendering preparation
                    for render_cycle in range(10):
                        try:
                            # Poll events to handle any pending window operations
                            if not self.vis.poll_events():
                                logging.warning("Window may be closing or invalid")
                                break
                            
                            # Force a complete scene update
                            with self.update_lock:
                                self._update_visualization()
                            
                            # Update renderer
                            self.vis.update_renderer()
                            
                            # Progressive delay to allow rendering to complete
                            time.sleep(0.1 + (render_cycle * 0.01))
                            
                        except Exception as render_error:
                            logging.debug(f"Render cycle {render_cycle} error: {render_error}")
                            continue
                    
                    # Additional stabilization delay
                    time.sleep(0.5)
                    
                    # Ensure we have valid geometry to render
                    if not any([self.drone_sphere, self.target_sphere, self.obstacles_pcd]):
                        logging.warning("No geometry available to render")
                        # Try to force add basic geometry
                        self._ensure_basic_geometry()
                    
                    # Try alternative screenshot methods
                    screenshot_methods = [
                        self._capture_screen_image_standard,
                        self._capture_screen_image_with_buffer,
                        self._capture_screen_image_fallback
                    ]
                    
                    for method_idx, method in enumerate(screenshot_methods):
                        try:
                            logging.debug(f"Trying screenshot method {method_idx + 1}")
                            success = method(filename)
                            if success:
                                break
                        except Exception as method_error:
                            logging.debug(f"Screenshot method {method_idx + 1} failed: {method_error}")
                            continue
                    
                    if success:
                        # Verify the screenshot
                        if os.path.exists(filename):
                            file_size = os.path.getsize(filename)
                            if file_size > 10000:  # Increased threshold to 10KB
                                logging.info(f"Successfully saved screenshot to {filename} ({file_size} bytes)")
                                return True
                            else:
                                logging.warning(f"Screenshot file too small ({file_size} bytes), likely empty")
                                success = False
                    
                    if not success and attempt < max_attempts - 1:
                        logging.debug(f"Screenshot attempt {attempt + 1} failed, retrying in 1 second...")
                        time.sleep(1.0)
                    
                except Exception as attempt_error:
                    logging.warning(f"Screenshot attempt {attempt + 1} failed with error: {attempt_error}")
                    if attempt < max_attempts - 1:
                        time.sleep(1.0)
                    continue
            
            logging.error(f"Failed to capture screenshot after {max_attempts} attempts")
            return False
                
        except Exception as e:
            logging.error(f"Critical error in save_screenshot: {e}")
            traceback.print_exc()
            return False
    def _capture_screen_image_standard(self, filename: str) -> bool:
        """Standard Open3D screenshot method"""
        try:
            self.vis.capture_screen_image(filename)
            return True
        except Exception as e:
            logging.debug(f"Standard capture failed: {e}")
            return False
    
    def _capture_screen_image_with_buffer(self, filename: str) -> bool:
        """Screenshot using float buffer method"""
        try:
            if not PIL_AVAILABLE:
                logging.debug("PIL not available for buffer-based screenshot")
                return False
                
            # Try using the float buffer method
            buffer = self.vis.capture_screen_float_buffer()
            if buffer is not None and len(buffer) > 0:
                # Convert float buffer to image and save
                import numpy as np
                
                # Convert float buffer to numpy array
                height = self.vis.get_render_option().window_height if hasattr(self.vis.get_render_option(), 'window_height') else 800
                width = self.vis.get_render_option().window_width if hasattr(self.vis.get_render_option(), 'window_width') else 1200
                
                # Try to get actual window size
                try:
                    # This may not be available in all Open3D versions
                    if hasattr(self.vis, 'get_window_size'):
                        width, height = self.vis.get_window_size()
                except:
                    pass
                
                # Reshape buffer to image
                img_array = np.array(buffer).reshape((height, width, 3))
                img_array = (img_array * 255).astype(np.uint8)
                
                # Flip vertically (OpenGL convention)
                img_array = np.flipud(img_array)
                
                # Save using PIL
                img = Image.fromarray(img_array, 'RGB')
                img.save(filename)
                return True
            return False
        except Exception as e:
            logging.debug(f"Float buffer capture failed: {e}")
            return False
    
    def _capture_screen_image_fallback(self, filename: str) -> bool:
        """Fallback screenshot method with alternative approach"""
        try:
            # Force one more complete render cycle
            for _ in range(3):
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.1)
            
            # Try the standard method one more time
            self.vis.capture_screen_image(filename)
            return True
        except Exception as e:
            logging.debug(f"Fallback capture failed: {e}")
            return False
    
    def _ensure_basic_geometry(self):
        """Ensure basic geometry exists for rendering"""
        try:
            # Make sure we have at least basic geometry visible
            if self.drone_sphere is None:
                self.drone_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
                self.drone_sphere.paint_uniform_color(self.colors['drone'])
                self.drone_sphere.translate(self.current_drone_pos)
                self.vis.add_geometry(self.drone_sphere)
            
            if self.target_sphere is None:
                self.target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
                self.target_sphere.paint_uniform_color(self.colors['target'])
                self.target_sphere.translate(self.target_pos)
                self.vis.add_geometry(self.target_sphere)
                
            logging.debug("Ensured basic geometry is available")
        except Exception as e:
            logging.debug(f"Error ensuring basic geometry: {e}")
    
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


# Convenience function for quick visualization setup
def create_voxel_visualizer(voxel_size=0.5, visualization_range=25.0, auto_start=True):
    """
    Create and optionally start a voxel grid visualizer
    
    Args:
        voxel_size: Size of voxels in meters
        visualization_range: Range around drone to visualize
        auto_start: Whether to automatically start the visualization
    
    Returns:
        VoxelGridVisualizer instance
    """
    visualizer = VoxelGridVisualizer(
        voxel_size=voxel_size,
        visualization_range=visualization_range
    )
    
    if auto_start:
        visualizer.start_visualization()
    
    return visualizer


if __name__ == "__main__":
    # Demo/test visualization with enhanced error handling
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create visualizer
        viz = create_voxel_visualizer(voxel_size=0.3, visualization_range=20.0)
        
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
        
        # Simulate drone movement
        for i in range(50):
            # Update drone position (circular motion)
            t = i * 0.2
            drone_pos = [5 * np.cos(t), 5 * np.sin(t), -5 + 2 * np.sin(t * 2)]
            viz.update_drone_position(drone_pos)
            
            # Update target occasionally
            if i % 20 == 0:
                target_pos = [10 + 5 * np.cos(t/2), 5 * np.sin(t/2), -5]
                viz.update_target_position(target_pos)
            
            # Add some dynamic obstacles
            if i % 15 == 0:  # Update obstacles every 15 iterations
                obstacles = []
                for _ in range(25):
                    obs_pos = [
                        random.uniform(-15, 15),
                        random.uniform(-15, 15),
                        random.uniform(-10, 0)
                    ]
                    obstacles.append(obs_pos)
                viz.update_obstacles(obstacles)
            
            # Save occasional screenshots
            if i % 10 == 0:
                viz.save_screenshot(f"demo_screenshot_{i:02d}.png")
            
            time.sleep(0.2)
        
        # Keep visualization running
        print("\nVisualization demo complete!")
        print("You should see a 3D window with:")
        print("- Blue sphere: Drone position")
        print("- Red sphere: Target position")
        print("- Green line: Drone path")
        print("- Orange/yellow points: Obstacles")
        print("- Gray grid: Ground reference")
        print("- Colored voxels: 3D obstacle grid")
        print("\nPress Enter to stop visualization...")
        input()
        
    except Exception as e:
        logging.error(f"Error in demo: {e}")
        traceback.print_exc()
    finally:
        if 'viz' in locals():
            viz.stop_visualization()
