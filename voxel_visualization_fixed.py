"""
Voxel Grid Visualization Module using Open3D

This module provides real-time 3D visualization of the drone's environment using voxel grids,
showing obstacles, the drone's position, path history, and target location.

Enhanced with Windows OpenGL context fix and thread-safe screenshot management.
"""

import open3d as o3d
import numpy as np
import time
import threading
import logging
import traceback
import os
import platform
import queue
from typing import List, Tuple, Optional, Dict, Any
from collections import deque


def check_visualization_support():
    """Check if Open3D visualization is supported on this platform"""
    try:
        # Try creating a minimal visualizer to test support
        test_vis = o3d.visualization.Visualizer()
        test_vis.create_window(width=100, height=100, visible=False)
        test_vis.destroy_window()
        return True, "GUI"
    except Exception as gui_error:
        try:
            # Try offscreen rendering
            test_renderer = o3d.visualization.rendering.OffscreenRenderer(100, 100)
            del test_renderer
            return True, "Offscreen"
        except Exception as offscreen_error:
            return False, f"GUI failed: {gui_error}, Offscreen failed: {offscreen_error}"


class VoxelGridVisualizer:
    """Real-time 3D voxel grid visualization for drone navigation with Windows OpenGL fix"""
    
    def __init__(self, voxel_size=0.5, grid_size=50, max_path_points=200, 
                 visualization_range=25.0, update_rate=10.0, enable_screenshots=True):
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
        
        # Screenshot queue and management for thread-safe Windows OpenGL handling
        self.screenshot_queue = queue.Queue()
        self.screenshot_results = {}
        self.screenshot_enabled = enable_screenshots and platform.system() == 'Windows'
        self.screenshot_consecutive_failures = 0
        self.max_consecutive_failures = 5  # Disable screenshots after this many failures
        
        # Data storage
        self.drone_path = deque(maxlen=max_path_points)
        self.current_drone_pos = np.array([0.0, 0.0, -3.0])  # Default NED position
        self.target_pos = np.array([10.0, 0.0, -3.0])
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
        
        # Color scheme - improved for better visibility
        self.colors = {
            'drone': [0.2, 0.8, 1.0],          # Bright blue
            'target': [1.0, 0.2, 0.2],        # Bright red
            'path': [0.2, 1.0, 0.2],          # Bright green
            'obstacles': [1.0, 0.6, 0.0],     # Orange
            'voxels': [0.7, 0.7, 0.7],        # Light gray
            'ground': [0.4, 0.4, 0.4],        # Medium gray
            'axes': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # RGB for XYZ
        }
        
        # Camera follow settings
        self.camera_follow_enabled = True
        self._last_lookat = None
        self._camera_distance = 15.0
        self._camera_height_offset = 8.0
        
        logging.info(f"Initialized VoxelGridVisualizer with voxel_size={voxel_size}m, "
                    f"grid_size={grid_size}, range={visualization_range}m, "
                    f"screenshots={'enabled' if self.enable_screenshots else 'disabled'}")
    
    def start_visualization(self, window_name="Drone Navigation - Voxel Grid Visualization"):
        """Start the visualization in a separate thread"""
        if self.is_running:
            logging.warning("Visualization is already running")
            return
        
        # Check if visualization is supported before starting
        supported, mode_or_error = check_visualization_support()
        if not supported:
            logging.warning(f"Open3D visualization not supported on this platform: {mode_or_error}")
            logging.info("Visualization will be disabled - continuing without 3D display")
            return
        else:
            logging.info(f"Open3D visualization supported via {mode_or_error}")
        
        self.is_running = True
        self.visualization_thread = threading.Thread(
            target=self._run_visualization,
            args=(window_name,),
            daemon=True
        )
        self.visualization_thread.start()
        
        # Wait longer for the visualization to initialize properly
        time.sleep(2.0)
        logging.info("Started voxel grid visualization")
    
    def stop_visualization(self):
        """Stop the visualization"""
        self.is_running = False
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=3.0)
        
        if self.vis:
            try:
                self.vis.destroy_window()
            except:
                pass
            self.vis = None
        
        logging.info("Stopped voxel grid visualization")
    
    def _run_visualization(self, window_name):
        """Main visualization loop running in separate thread with screenshot queue processing"""
        try:
            # Determine the best visualization mode for this platform
            is_windows = platform.system() == 'Windows'
            headless_env = os.environ.get('DISPLAY') is None or os.environ.get('HEADLESS') == '1'
            
            # On Windows, prefer GUI mode as headless often doesn't work
            # On Linux/Mac, can try headless first if no display
            gui_success = False
            
            # Try GUI mode first (especially on Windows)
            try:
                logging.info("Attempting GUI visualization mode")
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(window_name=window_name, width=1200, height=800, visible=False)
                gui_success = True
                logging.info("GUI visualization mode initialized successfully")
            except Exception as gui_error:
                logging.warning(f"GUI mode failed: {gui_error}")
                gui_success = False
            
            # If GUI failed and we're not on Windows, try offscreen rendering
            if not gui_success and not is_windows:
                try:
                    logging.info("Attempting headless mode - using offscreen renderer")
                    self.vis = o3d.visualization.rendering.OffscreenRenderer(1200, 800)
                    self._setup_offscreen_scene()
                    logging.info("Headless visualization mode initialized successfully")
                    return  # Exit early for headless mode
                except Exception as headless_error:
                    logging.warning(f"Headless mode also failed: {headless_error}")
            
            # If both modes failed, disable visualization
            if not gui_success:
                logging.warning("All visualization modes failed - disabling visualization")
                logging.info("Experiment will continue without 3D visualization")
                self.is_running = False
                return
            
            # Set up the GUI scene if we got here
            if gui_success:
                # Set up the initial scene
                self._setup_initial_scene()
                
                # Configure camera and rendering options
                self._configure_visualization()
                
                # Main update loop for GUI mode with screenshot queue processing
                last_update = 0
                update_interval = 1.0 / self.update_rate
                
                while self.is_running:
                    current_time = time.time()
                    
                    try:
                        # Process screenshot queue first (in the main visualization thread for Windows)
                        self._process_screenshot_queue()
                        
                        if current_time - last_update >= update_interval:
                            with self.update_lock:
                                self._update_visualization()
                                self.vis.poll_events()
                                self.vis.update_renderer()
                            last_update = current_time
                        
                        # Small sleep to prevent excessive CPU usage
                        time.sleep(0.02)  # 50 FPS max
                        
                    except Exception as e:
                        logging.debug(f"Error in visualization loop: {e}")
                        time.sleep(0.1)
        
        except Exception as e:
            logging.error(f"Critical error in visualization thread: {e}")
            traceback.print_exc()
        finally:
            self.is_running = False

    def _process_screenshot_queue(self):
        """Process pending screenshot requests in the main visualization thread"""
        while not self.screenshot_queue.empty():
            try:
                request_id, filename, max_retries = self.screenshot_queue.get_nowait()
                success = self._capture_screenshot_internal(filename, max_retries)
                self.screenshot_results[request_id] = success
                
                # Clean up old results to prevent memory leak
                if len(self.screenshot_results) > 10:
                    oldest_keys = list(self.screenshot_results.keys())[:-5]
                    for key in oldest_keys:
                        del self.screenshot_results[key]
                        
            except queue.Empty:
                break
            except Exception as e:
                logging.warning(f"Error processing screenshot queue: {e}")

    def _capture_screenshot_internal(self, filename: str, max_retries: int = 3) -> bool:
        """Internal screenshot capture method that runs in the main visualization thread"""
        if not self.vis or not self.is_running:
            return False
            
        if not self.enable_screenshots:
            logging.debug("Screenshots disabled")
            return False
            
        # Check if we've had too many consecutive failures
        if self.screenshot_consecutive_failures >= self.max_consecutive_failures:
            logging.debug(f"Screenshots disabled due to {self.screenshot_consecutive_failures} consecutive failures")
            return False
        
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except Exception as e:
            logging.warning(f"Could not create directory for screenshot: {e}")
            return False
        
        if hasattr(self.vis, 'capture_screen_image'):
            # For GUI visualizer with enhanced Windows OpenGL handling
            for attempt in range(max_retries):
                try:
                    # Pre-capture preparation - ensure scene is stable
                    for _ in range(2):  # Minimal renders to avoid context conflicts
                        if not self.is_running:
                            return False
                        self.vis.poll_events()
                        self.vis.update_renderer()
                        time.sleep(0.05)  # Short delay
                    
                    # Additional stabilization delay for Windows
                    if platform.system() == 'Windows':
                        time.sleep(0.15)
                    
                    # Capture screenshot
                    success = self.vis.capture_screen_image(filename)
                    if success:
                        logging.info(f"Saved visualization screenshot to {filename}")
                        self.screenshot_consecutive_failures = 0  # Reset failure counter
                        return True
                    else:
                        logging.warning(f"Screenshot capture returned False (attempt {attempt + 1}/{max_retries})")
                        
                except Exception as capture_error:
                    if "WGL" in str(capture_error) or "GLFW" in str(capture_error):
                        logging.warning(f"OpenGL context error on attempt {attempt + 1}/{max_retries}: {capture_error}")
                        # Longer wait for OpenGL context issues
                        time.sleep(0.3 + attempt * 0.2)
                    else:
                        logging.warning(f"Screenshot capture error on attempt {attempt + 1}/{max_retries}: {capture_error}")
                        time.sleep(0.1)
            
            # All attempts failed
            self.screenshot_consecutive_failures += 1
            logging.warning(f"Failed to capture screenshot to {filename} after {max_retries} attempts "
                          f"(consecutive failures: {self.screenshot_consecutive_failures})")
            
            # Disable screenshots if too many failures
            if self.screenshot_consecutive_failures >= self.max_consecutive_failures:
                logging.warning(f"Disabling screenshots due to {self.screenshot_consecutive_failures} consecutive failures")
                self.enable_screenshots = False
            
            return False
                
        elif hasattr(self.vis, 'render_to_image'):
            # For offscreen renderer
            try:
                image = self.vis.render_to_image()
                o3d.io.write_image(filename, image)
                logging.info(f"Saved offscreen screenshot to {filename}")
                self.screenshot_consecutive_failures = 0
                return True
            except Exception as render_error:
                logging.warning(f"Offscreen rendering failed: {render_error}")
                self.screenshot_consecutive_failures += 1
                return False
        else:
            logging.debug("No screenshot capability available")
            return False

    def _setup_offscreen_scene(self):
        """Set up scene for offscreen rendering"""
        try:
            # Create a simple scene for offscreen rendering
            self.scene = o3d.visualization.rendering.Open3DScene(self.vis)
            
            # Add lighting
            self.scene.scene.enable_sun_light(True)
            self.scene.scene.set_sun_light_direction([0.3, -1.0, -0.3])
            
            # Set background
            self.scene.scene.set_background([0.1, 0.1, 0.15, 1.0])
            
            # Setup camera for offscreen rendering
            bounds = o3d.geometry.AxisAlignedBoundingBox([-20, -20, -10], [20, 20, 5])
            self.scene.setup_camera(60.0, bounds, [0, 0, 0])
            
            logging.info("Offscreen scene setup complete")
            
        except Exception as e:
            logging.error(f"Error setting up offscreen scene: {e}")
    
    def _setup_initial_scene(self):
        """Set up the initial 3D scene with ground grid and coordinate axes"""
        try:
            # Add coordinate frame at origin
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=3.0, origin=[0, 0, 0]
            )
            self.vis.add_geometry(self.coordinate_frame)
            
            # Create ground grid
            self._create_ground_grid()
            
            # Initialize drone representation - larger for better visibility
            self.drone_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            self.drone_sphere.paint_uniform_color(self.colors['drone'])
            self.drone_sphere.translate(self.current_drone_pos)
            self.vis.add_geometry(self.drone_sphere)
            
            # Initialize target representation
            self.target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.6)
            self.target_sphere.paint_uniform_color(self.colors['target'])
            self.target_sphere.translate(self.target_pos)
            self.vis.add_geometry(self.target_sphere)
            
            # Initialize path line
            self.path_line = o3d.geometry.LineSet()
            self.path_line.paint_uniform_color(self.colors['path'])
            self.vis.add_geometry(self.path_line)
            
            # Initialize obstacles point cloud
            self.obstacles_pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.obstacles_pcd)
            
            # Initialize voxel grid placeholder
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
            grid_spacing = 5.0  # Larger spacing for less clutter
            
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
            
            # Create the LineSet
            self.ground_grid = o3d.geometry.LineSet()
            self.ground_grid.points = o3d.utility.Vector3dVector(points)
            self.ground_grid.lines = o3d.utility.Vector2iVector(lines)
            self.ground_grid.paint_uniform_color(self.colors['ground'])
            
            self.vis.add_geometry(self.ground_grid)
            
        except Exception as e:
            logging.error(f"Error creating ground grid: {e}")
    
    def _configure_visualization(self):
        """Configure camera and rendering options for optimal visibility"""
        try:
            # Get render option and view control
            render_option = self.vis.get_render_option()
            view_control = self.vis.get_view_control()
            
            # Configure rendering for better visibility and less black screens
            render_option.background_color = np.asarray([0.1, 0.1, 0.15])  # Dark blue background
            render_option.point_size = 8.0  # Larger points
            render_option.line_width = 4.0  # Thicker lines
            render_option.show_coordinate_frame = True
            render_option.light_on = True
            
            # Enhanced lighting settings
            if hasattr(render_option, 'mesh_shade_option'):
                render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
            
            # Set optimal camera position for drone navigation view
            # Position camera behind and above the drone's starting position
            drone_pos = self.current_drone_pos
            
            # Calculate camera position: behind the drone, elevated
            camera_pos = drone_pos + np.array([-self._camera_distance, 0, self._camera_height_offset])
            
            # Set camera parameters
            view_control.set_zoom(0.3)  # Wider field of view
            view_control.set_front([0.7, 0.0, -0.7])  # Angled down view
            view_control.set_lookat(drone_pos)  # Look at drone
            view_control.set_up([0, 0, 1])  # Z is up in NED coordinates
            
            # Force multiple renders to ensure proper initialization
            for _ in range(3):
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.1)
            
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
        """Create voxel grid from obstacle positions with improved density handling"""
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
            
            # Reduce point density to avoid clutter - subsample if too many points
            if len(filtered_points) > 1000:
                # Keep every nth point to reduce clutter
                step = len(filtered_points) // 1000
                filtered_points = filtered_points[::step]
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            
            # Create voxel grid from point cloud - use larger voxel size to reduce clutter
            effective_voxel_size = max(self.voxel_size, 0.5)  # Minimum 0.5m voxels
            self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                pcd, voxel_size=effective_voxel_size
            )
            
            # Color voxels based on density/distance
            self._color_voxels()
            
        except Exception as e:
            logging.debug(f"Error creating voxel grid: {e}")
    
    def _color_voxels(self):
        """Color voxels based on distance from drone"""
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
                
                # Color based on distance - improved color scheme
                max_dist = self.visualization_range
                normalized_dist = min(distance_to_drone / max_dist, 1.0)
                
                # Create more visible color gradient
                if normalized_dist < 0.3:  # Very close - bright red
                    voxel.color = [1.0, 0.0, 0.0]
                elif normalized_dist < 0.6:  # Medium distance - orange
                    voxel.color = [1.0, 0.5, 0.0]
                else:  # Far - yellow
                    voxel.color = [1.0, 1.0, 0.0]
                
        except Exception as e:
            logging.debug(f"Error coloring voxels: {e}")
    
    def _update_visualization(self):
        """Update all visualization elements"""
        try:
            # Update drone position
            if self.drone_sphere:
                # Remove and recreate sphere to ensure proper positioning
                self.vis.remove_geometry(self.drone_sphere, reset_bounding_box=False)
                
                self.drone_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
                self.drone_sphere.paint_uniform_color(self.colors['drone'])
                self.drone_sphere.translate(self.current_drone_pos)
                
                self.vis.add_geometry(self.drone_sphere, reset_bounding_box=False)
            
            # Update target position
            if self.target_sphere:
                self.vis.remove_geometry(self.target_sphere, reset_bounding_box=False)
                
                self.target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.6)
                self.target_sphere.paint_uniform_color(self.colors['target'])
                self.target_sphere.translate(self.target_pos)
                
                self.vis.add_geometry(self.target_sphere, reset_bounding_box=False)
            
            # Update path
            if len(self.drone_path) > 1 and self.path_line:
                path_points = list(self.drone_path)
                path_lines = [[i, i + 1] for i in range(len(path_points) - 1)]
                
                self.path_line.points = o3d.utility.Vector3dVector(path_points)
                self.path_line.lines = o3d.utility.Vector2iVector(path_lines)
                
                # Create gradient colors for path
                path_colors = []
                for i in range(len(path_lines)):
                    intensity = (i + 1) / len(path_lines)
                    color = [0, intensity, 0]  # Green gradient
                    path_colors.append(color)
                
                self.path_line.colors = o3d.utility.Vector3dVector(path_colors)
                self.vis.update_geometry(self.path_line)
            
            # Update obstacles point cloud with reduced clutter
            if self.obstacle_positions and self.obstacles_pcd:
                obstacle_points = np.array(self.obstacle_positions)
                
                # Filter and subsample obstacles to reduce clutter
                drone_pos = self.current_drone_pos
                distances = np.linalg.norm(obstacle_points - drone_pos, axis=1)
                in_range_mask = distances <= self.visualization_range
                
                if np.any(in_range_mask):
                    filtered_points = obstacle_points[in_range_mask]
                    filtered_distances = distances[in_range_mask]
                    
                    # Subsample if too many points
                    if len(filtered_points) > 500:
                        step = len(filtered_points) // 500
                        filtered_points = filtered_points[::step]
                        filtered_distances = filtered_distances[::step]
                    
                    self.obstacles_pcd.points = o3d.utility.Vector3dVector(filtered_points)
                    
                    # Color obstacles based on distance with better visibility
                    colors = []
                    max_dist = self.visualization_range
                    for dist in filtered_distances:
                        normalized_dist = min(dist / max_dist, 1.0)
                        # Bright orange to yellow gradient
                        red = 1.0
                        green = 0.5 + 0.5 * (1.0 - normalized_dist)
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
            
            # Update camera to follow drone
            self._update_camera_view()
            
        except Exception as e:
            logging.debug(f"Error updating visualization: {e}")
    
    def _update_camera_view(self):
        """Update camera to follow the drone with improved positioning"""
        try:
            if not self.camera_follow_enabled or not hasattr(self.vis, 'get_view_control'):
                return
            
            view_control = self.vis.get_view_control()
            drone_pos = self.current_drone_pos
            
            # Calculate optimal camera position
            # Position camera behind and above the drone
            camera_offset = np.array([-self._camera_distance, 0, self._camera_height_offset])
            camera_target = drone_pos + np.array([5, 0, 0])  # Look ahead of drone
            
            # Smooth camera transition
            if self._last_lookat is not None:
                # Interpolate to new position
                new_lookat = 0.8 * self._last_lookat + 0.2 * camera_target
            else:
                new_lookat = camera_target
            
            self._last_lookat = new_lookat
            
            # Update camera
            try:
                view_control.set_lookat(new_lookat)
                view_control.set_front([0.7, 0.0, -0.7])
                view_control.set_up([0, 0, 1])
            except:
                pass  # Skip if camera update fails
                        
        except Exception as e:
            logging.debug(f"Error updating camera view: {e}")
    
    def save_screenshot(self, filename: str, max_retries: int = 3, timeout: float = 2.0) -> bool:
        """
        Save a screenshot of the current visualization using thread-safe queue system
        
        Args:
            filename: Path where to save the screenshot
            max_retries: Maximum number of capture attempts
            timeout: Maximum time to wait for screenshot completion
            
        Returns:
            bool: True if screenshot was saved successfully, False otherwise
        """
        if not self.vis or not self.is_running:
            logging.debug("Visualizer not initialized or not running - cannot save screenshot")
            return False
            
        if not self.enable_screenshots:
            logging.debug("Screenshots disabled")
            return False
        
        # For non-Windows platforms or offscreen rendering, use direct capture
        if platform.system() != 'Windows' or hasattr(self.vis, 'render_to_image'):
            return self._capture_screenshot_internal(filename, max_retries)
        
        # For Windows GUI mode, use queue system to avoid threading issues
        request_id = f"{time.time()}_{filename}"
        self.screenshot_queue.put((request_id, filename, max_retries))
        
        # Wait for the screenshot to be processed
        start_time = time.time()
        while request_id not in self.screenshot_results:
            if time.time() - start_time > timeout:
                logging.warning(f"Screenshot request timed out after {timeout}s")
                return False
            time.sleep(0.05)
        
        # Get the result and clean up
        result = self.screenshot_results.pop(request_id)
        return result
    
    def toggle_camera_follow(self):
        """Toggle camera following mode"""
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
            'last_update': self.last_update_time,
            'camera_follow': self.camera_follow_enabled
        }


# Convenience function for quick visualization setup
def create_voxel_visualizer(voxel_size=0.5, visualization_range=25.0, auto_start=True, enable_screenshots=True):
    """
    Create and optionally start a voxel grid visualizer
    
    Args:
        voxel_size: Size of voxels in meters
        visualization_range: Range around drone to visualize
        auto_start: Whether to automatically start the visualization
        enable_screenshots: Whether to enable screenshot functionality
    
    Returns:
        VoxelGridVisualizer instance (may have is_running=False if visualization failed)
    """
    visualizer = VoxelGridVisualizer(
        voxel_size=voxel_size,
        visualization_range=visualization_range,
        enable_screenshots=enable_screenshots
    )
    
    if auto_start:
        visualizer.start_visualization()
        # If visualization failed to start, log it but don't raise an error
        if not visualizer.is_running:
            logging.info("Visualization could not be started - continuing without 3D display")
    
    return visualizer


if __name__ == "__main__":
    # Demo/test visualization with enhanced error handling
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create visualizer
        viz = create_voxel_visualizer(voxel_size=0.5, visualization_range=20.0)
        
        # Wait for visualization to start
        time.sleep(3.0)
        
        # Simulate some data updates
        import random
        
        # Initial positions to ensure something is visible
        viz.update_drone_position([0, 0, -5])
        viz.update_target_position([15, 5, -3])
        
        # Add some initial obstacles
        initial_obstacles = []
        for i in range(20):  # Reduced number for less clutter
            obs_pos = [
                random.uniform(-15, 15),
                random.uniform(-10, 10),
                random.uniform(-8, -1)
            ]
            initial_obstacles.append(obs_pos)
        viz.update_obstacles(initial_obstacles)
        
        # Simulate drone movement
        for i in range(30):
            # Update drone position (path toward target)
            t = i * 0.5
            drone_pos = [t * 0.5, 2 * np.sin(t * 0.3), -5 + np.sin(t * 0.2)]
            viz.update_drone_position(drone_pos)
            
            # Update target occasionally
            if i % 10 == 0:
                target_pos = [15 + 3 * np.cos(t/3), 5 + 2 * np.sin(t/3), -3]
                viz.update_target_position(target_pos)
            
            # Add some dynamic obstacles occasionally
            if i % 8 == 0:
                obstacles = []
                for _ in range(15):  # Fewer obstacles
                    obs_pos = [
                        random.uniform(-20, 20),
                        random.uniform(-15, 15),
                        random.uniform(-8, 0)
                    ]
                    obstacles.append(obs_pos)
                viz.update_obstacles(obstacles)
            
            # Save screenshots more frequently for testing
            if i % 5 == 0:
                viz.save_screenshot(f"demo_screenshot_{i:02d}.png")
            
            time.sleep(0.5)
        
        # Keep visualization running
        print("\nVisualization demo complete!")
        print("You should see a 3D window with:")
        print("- Bright blue sphere: Drone position")
        print("- Bright red sphere: Target position")
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
