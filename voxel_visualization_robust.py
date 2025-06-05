#!/usr/bin/env python3
"""
Robust Voxel Grid Visualizer with OpenGL Context Failure Resilience
This implementation provides multiple fallback strategies for screenshot capture.
"""

import numpy as np
import open3d as o3d
import logging
import os
import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import cv2
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustVoxelGridVisualizer:
    """
    A robust voxel grid visualizer with multiple screenshot capture methods
    and OpenGL context failure resilience.
    """
    
    def __init__(self, voxel_size=0.5, visualization_range=25.0):
        """Initialize the robust voxel grid visualizer"""
        self.voxel_size = voxel_size
        self.visualization_range = visualization_range
        self.grid_size = int(2 * visualization_range / voxel_size)
        
        # Core visualization components
        self.vis = None
        self.voxel_grid = None
        self.drone_marker = None
        self.target_marker = None
        self.path_line = None
        
        # State tracking
        self.is_running = False
        self.current_obstacles = []
        self.current_drone_pos = [0, 0, 0]
        self.current_target_pos = [10, 10, -5]
        self.flight_path = []
        
        # Screenshot capture methods
        self.screenshot_methods = [
            self._capture_matplotlib_fallback,
            self._capture_opencv_fallback,
            self._capture_data_export,
            self._capture_simplified_open3d,
            self._capture_headless_open3d
        ]
        
        # Threading for visualization
        self.vis_thread = None
        self.lock = threading.Lock()
        
        logger.info(f"Initialized RobustVoxelGridVisualizer with voxel_size={voxel_size}m, "
                   f"grid_size={self.grid_size}, range={visualization_range}m")
    
    def start_visualization(self, window_name="Robust Voxel Grid Visualization"):
        """Start the visualization in a separate thread"""
        if self.is_running:
            logger.warning("Visualization already running")
            return
        
        self.window_name = window_name
        self.is_running = True
        
        # Start visualization thread
        self.vis_thread = threading.Thread(target=self._run_visualization)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        
        # Wait for initialization
        time.sleep(2.0)
        logger.info("Started robust voxel grid visualization")
    
    def _run_visualization(self):
        """Run the Open3D visualization loop"""
        try:
            # Create visualization window
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.window_name, width=1200, height=800)
            
            # Initialize empty voxel grid
            self._initialize_voxel_grid()
            
            # Add coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            self.vis.add_geometry(coordinate_frame)
            
            # Set up camera
            self._setup_camera()
            
            # Main visualization loop
            while self.is_running:
                with self.lock:
                    self._update_visualization()
                
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                time.sleep(0.05)  # ~20 FPS
                
        except Exception as e:
            logger.error(f"Visualization error: {e}")
        finally:
            if self.vis:
                self.vis.destroy_window()
    
    def _initialize_voxel_grid(self):
        """Initialize the voxel grid structure"""
        self.voxel_grid = o3d.geometry.VoxelGrid()
        
        # Create drone marker
        self.drone_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
        self.drone_marker.paint_uniform_color([0, 1, 0])  # Green
        
        # Create target marker
        self.target_marker = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        self.target_marker.paint_uniform_color([1, 0, 0])  # Red
        
        # Add geometries to visualizer
        self.vis.add_geometry(self.voxel_grid)
        self.vis.add_geometry(self.drone_marker)
        self.vis.add_geometry(self.target_marker)
    
    def _setup_camera(self):
        """Set up the camera view"""
        view_control = self.vis.get_view_control()
        view_control.set_lookat([0, 0, -5])
        view_control.set_up([0, 0, 1])
        view_control.set_front([1, 1, 1])
        view_control.set_zoom(0.5)
    
    def _update_visualization(self):
        """Update the visualization with current data"""
        try:
            # Update voxel grid with obstacles
            if self.current_obstacles:
                self._update_voxel_grid()
            
            # Update drone position
            self._update_drone_marker()
            
            # Update target position
            self._update_target_marker()
            
            # Update flight path
            self._update_flight_path()
            
        except Exception as e:
            logger.warning(f"Visualization update error: {e}")
    
    def _update_voxel_grid(self):
        """Update voxel grid with current obstacles"""
        # Clear existing voxels
        self.voxel_grid.clear()
        
        # Add obstacle voxels
        for obstacle in self.current_obstacles:
            voxel_center = np.array(obstacle)
            voxel = o3d.geometry.VoxelGrid.Voxel(
                grid_index=self._world_to_grid(voxel_center),
                color=[1, 0, 0]  # Red for obstacles
            )
            self.voxel_grid.add_voxel(voxel)
        
        self.vis.update_geometry(self.voxel_grid)
    
    def _update_drone_marker(self):
        """Update drone marker position"""
        translation = np.array(self.current_drone_pos)
        self.drone_marker.translate(translation, relative=False)
        self.vis.update_geometry(self.drone_marker)
    
    def _update_target_marker(self):
        """Update target marker position"""
        translation = np.array(self.current_target_pos)
        self.target_marker.translate(translation, relative=False)
        self.vis.update_geometry(self.target_marker)
    
    def _update_flight_path(self):
        """Update flight path visualization"""
        if len(self.flight_path) > 1 and self.path_line is not None:
            points = np.array(self.flight_path)
            lines = [[i, i+1] for i in range(len(points)-1)]
            
            self.path_line.points = o3d.utility.Vector3dVector(points)
            self.path_line.lines = o3d.utility.Vector2iVector(lines)
            self.vis.update_geometry(self.path_line)
    
    def _world_to_grid(self, world_pos):
        """Convert world position to grid coordinates"""
        grid_pos = np.array(world_pos) / self.voxel_size
        grid_pos = grid_pos + self.grid_size // 2  # Center the grid
        return tuple(grid_pos.astype(int))
    
    def update_obstacles(self, obstacles):
        """Update obstacles in the visualization"""
        with self.lock:
            self.current_obstacles = obstacles.copy() if obstacles else []
    
    def update_drone_position(self, position):
        """Update drone position"""
        with self.lock:
            self.current_drone_pos = position.copy()
            self.flight_path.append(position.copy())
            
            # Limit path history
            if len(self.flight_path) > 100:
                self.flight_path = self.flight_path[-100:]
    
    def update_target_position(self, position):
        """Update target position"""
        with self.lock:
            self.current_target_pos = position.copy()
    
    def save_screenshot(self, filepath):
        """
        Save screenshot using multiple fallback methods.
        Returns True if successful, False otherwise.
        """
        logger.info(f"Attempting to save screenshot to: {filepath}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Try each screenshot method until one succeeds
        for i, method in enumerate(self.screenshot_methods):
            try:
                logger.info(f"Trying screenshot method {i+1}: {method.__name__}")
                success = method(filepath)
                
                if success and os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
                    logger.info(f"Screenshot saved successfully using {method.__name__}")
                    return True
                else:
                    logger.warning(f"Method {method.__name__} failed or produced small file")
                    
            except Exception as e:
                logger.warning(f"Screenshot method {method.__name__} failed: {e}")
                continue
        
        logger.error("All screenshot methods failed")
        return False
    
    def _capture_matplotlib_fallback(self, filepath):
        """Fallback screenshot using matplotlib"""
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot obstacles as red cubes
            if self.current_obstacles:
                for obs in self.current_obstacles:
                    ax.scatter(obs[0], obs[1], obs[2], c='red', s=100, marker='s', alpha=0.7)
            
            # Plot drone as green sphere
            ax.scatter(self.current_drone_pos[0], self.current_drone_pos[1], 
                      self.current_drone_pos[2], c='green', s=200, marker='o', alpha=0.9)
            
            # Plot target as red sphere
            ax.scatter(self.current_target_pos[0], self.current_target_pos[1], 
                      self.current_target_pos[2], c='red', s=300, marker='o', alpha=0.9)
            
            # Plot flight path
            if len(self.flight_path) > 1:
                path_array = np.array(self.flight_path)
                ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                       'b-', alpha=0.6, linewidth=2)
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Voxel Grid Visualization (Matplotlib Fallback)')
            
            # Set equal aspect ratio
            max_range = self.visualization_range
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, 0])
            
            # Add legend
            red_patch = mpatches.Patch(color='red', label='Obstacles/Target')
            green_patch = mpatches.Patch(color='green', label='Drone')
            blue_patch = mpatches.Patch(color='blue', label='Flight Path')
            ax.legend(handles=[red_patch, green_patch, blue_patch])
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Matplotlib fallback failed: {e}")
            return False
    
    def _capture_opencv_fallback(self, filepath):
        """Fallback screenshot using OpenCV for 2D visualization"""
        try:
            # Create a 800x600 image
            img = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background
            
            # Convert 3D coordinates to 2D (top-down view)
            scale = 15  # pixels per meter
            center_x, center_y = 400, 300
            
            # Draw obstacles
            for obs in self.current_obstacles:
                x = int(center_x + obs[0] * scale)
                y = int(center_y + obs[1] * scale)
                if 0 <= x < 800 and 0 <= y < 600:
                    cv2.rectangle(img, (x-5, y-5), (x+5, y+5), (0, 0, 255), -1)  # Red
            
            # Draw flight path
            if len(self.flight_path) > 1:
                points = []
                for pos in self.flight_path[-50:]:  # Last 50 points
                    x = int(center_x + pos[0] * scale)
                    y = int(center_y + pos[1] * scale)
                    if 0 <= x < 800 and 0 <= y < 600:
                        points.append((x, y))
                
                for i in range(len(points)-1):
                    cv2.line(img, points[i], points[i+1], (255, 0, 0), 2)  # Blue
            
            # Draw drone
            drone_x = int(center_x + self.current_drone_pos[0] * scale)
            drone_y = int(center_y + self.current_drone_pos[1] * scale)
            if 0 <= drone_x < 800 and 0 <= drone_y < 600:
                cv2.circle(img, (drone_x, drone_y), 8, (0, 255, 0), -1)  # Green
            
            # Draw target
            target_x = int(center_x + self.current_target_pos[0] * scale)
            target_y = int(center_y + self.current_target_pos[1] * scale)
            if 0 <= target_x < 800 and 0 <= target_y < 600:
                cv2.circle(img, (target_x, target_y), 12, (0, 0, 255), -1)  # Red
            
            # Add title and legend
            cv2.putText(img, "Voxel Grid Visualization (Top-Down View)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img, "Green: Drone, Red: Target/Obstacles, Blue: Path", 
                       (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Add coordinate system
            cv2.arrowedLine(img, (50, 550), (100, 550), (0, 0, 0), 2)  # X axis
            cv2.arrowedLine(img, (50, 550), (50, 500), (0, 0, 0), 2)  # Y axis
            cv2.putText(img, "X", (105, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(img, "Y", (45, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Save the image
            success = cv2.imwrite(filepath, img)
            return success
            
        except Exception as e:
            logger.error(f"OpenCV fallback failed: {e}")
            return False
    
    def _capture_data_export(self, filepath):
        """Export visualization data as JSON for external rendering"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'drone_position': self.current_drone_pos,
                'target_position': self.current_target_pos,
                'obstacles': self.current_obstacles,
                'flight_path': self.flight_path[-50:],  # Last 50 points
                'voxel_size': self.voxel_size,
                'visualization_range': self.visualization_range
            }
            
            json_filepath = filepath.replace('.png', '.json')
            with open(json_filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Also create a simple text visualization
            text_filepath = filepath.replace('.png', '.txt')
            with open(text_filepath, 'w') as f:
                f.write("VOXEL GRID VISUALIZATION DATA\n")
                f.write("=" * 40 + "\n")
                f.write(f"Timestamp: {data['timestamp']}\n")
                f.write(f"Drone Position: {data['drone_position']}\n")
                f.write(f"Target Position: {data['target_position']}\n")
                f.write(f"Number of Obstacles: {len(data['obstacles'])}\n")
                f.write(f"Flight Path Length: {len(data['flight_path'])}\n")
                f.write(f"Voxel Size: {data['voxel_size']}m\n")
                f.write(f"Visualization Range: {data['visualization_range']}m\n")
            
            logger.info(f"Data exported to {json_filepath} and {text_filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return False
    
    def _capture_simplified_open3d(self, filepath):
        """Simplified Open3D screenshot without complex geometries"""
        try:
            # Create a minimal visualizer
            simple_vis = o3d.visualization.Visualizer()
            simple_vis.create_window(width=800, height=600, visible=False)
            
            # Add only basic geometries
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
            simple_vis.add_geometry(coordinate_frame)
            
            # Simple point cloud for obstacles
            if self.current_obstacles:
                points = np.array(self.current_obstacles)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color([1, 0, 0])  # Red
                simple_vis.add_geometry(pcd)
            
            # Take screenshot
            simple_vis.poll_events()
            simple_vis.update_renderer()
            time.sleep(0.5)  # Wait for rendering
            
            success = simple_vis.capture_screen_image(filepath)
            simple_vis.destroy_window()
            
            return success
            
        except Exception as e:
            logger.error(f"Simplified Open3D failed: {e}")
            return False
    
    def _capture_headless_open3d(self, filepath):
        """Headless Open3D rendering"""
        try:
            # Create headless renderer
            render = o3d.visualization.rendering.OffscreenRenderer(800, 600)
            
            # Set up scene
            render.scene.set_background([1, 1, 1, 1])  # White background
            
            # Add coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
            render.scene.add_geometry("coord_frame", coordinate_frame, 
                                    o3d.visualization.rendering.MaterialRecord())
            
            # Add obstacles as spheres
            for i, obs in enumerate(self.current_obstacles[:10]):  # Limit to 10 obstacles
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
                sphere.translate(obs)
                sphere.paint_uniform_color([1, 0, 0])  # Red
                render.scene.add_geometry(f"obstacle_{i}", sphere,
                                        o3d.visualization.rendering.MaterialRecord())
            
            # Set camera
            render.setup_camera(60.0, [0, 0, 0], [10, 10, 10], [0, 0, 1])
            
            # Render and save
            image = render.render_to_image()
            o3d.io.write_image(filepath, image)
            
            return True
            
        except Exception as e:
            logger.error(f"Headless Open3D failed: {e}")
            return False
    
    def stop_visualization(self):
        """Stop the visualization"""
        self.is_running = False
        if self.vis_thread and self.vis_thread.is_alive():
            self.vis_thread.join(timeout=2.0)
        logger.info("Stopped robust voxel grid visualization")

# Backward compatibility
VoxelGridVisualizer = RobustVoxelGridVisualizer

def main():
    """Test the robust visualization system"""
    print("Testing Robust Voxel Grid Visualizer...")
    
    # Create visualizer
    viz = RobustVoxelGridVisualizer(voxel_size=0.5, visualization_range=25.0)
    
    # Start visualization
    viz.start_visualization("Robust Test")
    
    # Add test data
    obstacles = [[1, 1, -5], [2, 2, -5], [3, 3, -6]]
    viz.update_obstacles(obstacles)
    viz.update_drone_position([0, 0, -5])
    viz.update_target_position([10, 10, -5])
    
    # Test screenshot
    test_dir = "test_robust_screenshots"
    os.makedirs(test_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(test_dir, f"robust_test_{timestamp}.png")
    
    success = viz.save_screenshot(screenshot_path)
    print(f"Screenshot test: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Keep running for a few seconds
    time.sleep(5)
    
    # Stop visualization
    viz.stop_visualization()
    print("Test completed")

if __name__ == "__main__":
    main()
