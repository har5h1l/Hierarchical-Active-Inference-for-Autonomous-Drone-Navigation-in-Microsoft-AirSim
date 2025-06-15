"""
Depth Camera Integration for AirSim Drone Navigation
This module provides depth camera functionality to integrate with the existing LiDAR-based Scanner class.

Usage:
1. Add the methods from this file to your Scanner class in data_collection.py
2. Update the Scanner.__init__ method to include the depth camera parameters
3. Modify fetch_density_distances to use get_fused_sensor_data instead

Author: AI Assistant for AirSim Integration
"""

import airsim
import numpy as np
import logging

class DepthCameraProcessor:
    """
    Depth camera processing functionality for AirSim integration
    This class demonstrates the methods that should be added to the Scanner class
    """
    
    def __init__(self, client, depth_camera_name="front_center"):
        self.client = client
        self.depth_camera_name = depth_camera_name
        
        # Sensor positioning from settings.json
        # Camera position relative to drone center (25cm forward)
        self.camera_offset = np.array([0.25, 0.0, 0.0])
        # LiDAR position relative to drone center (10cm down) 
        self.lidar_offset = np.array([0.0, 0.0, -0.1])
        
        # Depth camera parameters (from settings.json)
        self.depth_image_width = 640
        self.depth_image_height = 480
        self.depth_camera_fov = 90.0  # Field of view in degrees
        
        # Calculate camera intrinsic parameters
        self.fx = self.depth_image_width / (2.0 * np.tan(np.radians(self.depth_camera_fov / 2.0)))
        self.fy = self.fx  # Assuming square pixels
        self.cx = self.depth_image_width / 2.0
        self.cy = self.depth_image_height / 2.0
        
        logging.info(f"DepthCameraProcessor initialized with fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def get_depth_camera_data(self, scan_range=20.0):
        """
        Get depth camera data from AirSim and convert to point cloud
        
        Args:
            scan_range: Maximum distance for depth processing (meters)
            
        Returns:
            np.ndarray: Point cloud data in camera coordinate frame (N x 3)
        """
        try:
            # Method 1: Try DepthPlanar first (more accurate)
            try:
                response = self.client.simGetImage(self.depth_camera_name, airsim.ImageType.DepthPlanar)
                
                if response is not None:
                    # Convert response to numpy array (float32)
                    depth_img = airsim.get_pfm_array(response)
                    
                    if depth_img is not None and depth_img.size > 0:
                        # Filter valid depths
                        valid_mask = (depth_img > 0.1) & (depth_img < scan_range) & np.isfinite(depth_img)
                        
                        # Get pixel coordinates
                        v_coords, u_coords = np.where(valid_mask)
                        depths = depth_img[valid_mask]
                        
                        if len(depths) > 0:
                            # Convert to 3D points using camera intrinsics
                            x_cam = (u_coords - self.cx) * depths / self.fx
                            y_cam = (v_coords - self.cy) * depths / self.fy
                            z_cam = depths
                            
                            points_camera = np.column_stack((x_cam, y_cam, z_cam))
                            
                            logging.debug(f"Generated {len(points_camera)} points from depth camera (DepthPlanar)")
                            return points_camera
            
            except Exception as e:
                logging.debug(f"DepthPlanar method failed: {e}, trying DepthPerspective")
            
            # Method 2: Fallback to DepthPerspective
            response = self.client.simGetImage(self.depth_camera_name, airsim.ImageType.DepthPerspective)
            
            if response is None or len(response) == 0:
                logging.warning("Depth camera response is None or empty")
                return np.array([]).reshape(0, 3)
            
            # Convert to numpy array
            img_1d = np.frombuffer(response, dtype=np.uint8)
            if len(img_1d) == 0:
                logging.warning("Empty depth image data")
                return np.array([]).reshape(0, 3)
            
            # Reshape to 2D image - depth is stored as grayscale
            depth_img = img_1d.reshape(self.depth_image_height, self.depth_image_width)
            depth_img = depth_img.astype(np.float32)
            
            # Convert to actual depth values (AirSim depth is stored as grayscale 0-255)
            # Scale to reasonable depth range (0-100m)
            depth_img = depth_img / 255.0 * 100.0
            
            # Filter out invalid depths (too close or too far)
            valid_mask = (depth_img > 0.1) & (depth_img < scan_range)
            
            # Get pixel coordinates for valid depths
            v_coords, u_coords = np.where(valid_mask)
            depths = depth_img[valid_mask]
            
            if len(depths) == 0:
                return np.array([]).reshape(0, 3)
            
            # Convert to 3D points using camera intrinsics
            # Camera coordinate system: X-right, Y-down, Z-forward
            x_cam = (u_coords - self.cx) * depths / self.fx
            y_cam = (v_coords - self.cy) * depths / self.fy  
            z_cam = depths
            
            # Stack to form point cloud
            points_camera = np.column_stack((x_cam, y_cam, z_cam))
            
            logging.debug(f"Generated {len(points_camera)} points from depth camera (DepthPerspective)")
            return points_camera
            
        except Exception as e:
            logging.error(f"Error getting depth camera data: {e}")
            return np.array([]).reshape(0, 3)

    def transform_camera_points_to_global(self, camera_points, drone_pos, drone_quat):
        """
        Transform depth camera points to global coordinate system
        Accounts for camera position offset relative to drone center
        
        Args:
            camera_points: Point cloud in camera coordinate frame (N x 3)
            drone_pos: Drone position in global frame (3,)
            drone_quat: Drone orientation quaternion [w, x, y, z] (4,)
            
        Returns:
            np.ndarray: Points in global coordinate frame (N x 3)
        """
        if len(camera_points) == 0:
            return np.array([]).reshape(0, 3)
        
        try:
            # Camera coordinate system transformation
            # AirSim camera: X-right, Y-down, Z-forward
            # Convert to drone body frame: X-forward, Y-right, Z-down
            
            # Rotation matrix from camera frame to drone body frame
            # Camera: [X-right, Y-down, Z-forward] -> Drone: [X-forward, Y-right, Z-down]
            camera_to_drone = np.array([
                [0, 0, 1],   # Drone X = Camera Z (forward)
                [1, 0, 0],   # Drone Y = Camera X (right)  
                [0, 1, 0]    # Drone Z = Camera Y (down)
            ])
            
            # Transform points to drone body frame
            points_drone_body = camera_points @ camera_to_drone.T
            
            # Add camera offset (camera is 0.25m forward of drone center)
            points_drone_body += self.camera_offset
            
            # Quaternion utility functions
            def quaternion_multiply(q1, q2):
                w1, x1, y1, z1 = q1
                w2, x2, y2, z2 = q2
                return np.array([
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                ])
            
            def quaternion_conjugate(q):
                return np.array([q[0], -q[1], -q[2], -q[3]])
            
            def rotate_point_by_quaternion(point, quat):
                point_quat = np.array([0.0, point[0], point[1], point[2]])
                quat_conj = quaternion_conjugate(quat)
                temp = quaternion_multiply(quat, point_quat)
                rotated_quat = quaternion_multiply(temp, quat_conj)
                return rotated_quat[1:4]
            
            # Transform points from drone body frame to global frame
            global_points = []
            for point in points_drone_body:
                # Rotate by drone orientation
                rotated_point = rotate_point_by_quaternion(point, drone_quat)
                # Translate by drone position
                global_point = rotated_point + drone_pos
                global_points.append(global_point)
            
            return np.array(global_points)
            
        except Exception as e:
            logging.error(f"Error transforming camera points to global frame: {e}")
            return np.array([]).reshape(0, 3)

    def get_fused_sensor_data(self, lidar_data, drone_pos, drone_quat, scan_range=20.0):
        """
        Get fused data from both LiDAR and depth camera
        
        Args:
            lidar_data: Raw LiDAR data from AirSim
            drone_pos: Drone position in global frame
            drone_quat: Drone orientation quaternion
            scan_range: Maximum scan range in meters
            
        Returns:
            tuple: (global_points, point_sources) where each point has an associated source label
        """
        try:
            all_points = []
            all_sources = []
            
            # Process LiDAR data
            if lidar_data and len(lidar_data.point_cloud) >= 3:
                try:
                    lidar_points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
                    
                    # Transform LiDAR points to global coordinates
                    for point in lidar_points:
                        if not (np.any(np.isnan(point)) or np.any(np.isinf(point))):
                            # Add LiDAR offset (10cm down from drone center)
                            point_with_offset = point + self.lidar_offset
                            
                            # Transform to global coordinates
                            def rotate_point_by_quaternion(point, quat):
                                point_quat = np.array([0.0, point[0], point[1], point[2]])
                                quat_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
                                def quaternion_multiply(q1, q2):
                                    w1, x1, y1, z1 = q1
                                    w2, x2, y2, z2 = q2
                                    return np.array([
                                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                                    ])
                                temp = quaternion_multiply(quat, point_quat)
                                rotated_quat = quaternion_multiply(temp, quat_conj)
                                return rotated_quat[1:4]
                            
                            global_point = rotate_point_by_quaternion(point_with_offset, drone_quat) + drone_pos
                            distance = np.linalg.norm(global_point - drone_pos)
                            
                            if 0.1 < distance <= scan_range:
                                all_points.append(global_point)
                                all_sources.append('lidar')
                    
                    logging.debug(f"Processed {len([s for s in all_sources if s == 'lidar'])} LiDAR points")
                except Exception as e:
                    logging.debug(f"Error processing LiDAR data: {e}")
            
            # Process depth camera data
            try:
                camera_points = self.get_depth_camera_data(scan_range)
                
                if len(camera_points) > 0:
                    # Transform to global coordinates
                    global_camera_points = self.transform_camera_points_to_global(
                        camera_points, drone_pos, drone_quat
                    )
                    
                    # Filter by distance and add to collection
                    for point in global_camera_points:
                        distance = np.linalg.norm(point - drone_pos)
                        if 0.1 < distance <= scan_range:
                            all_points.append(point)
                            all_sources.append('camera')
                    
                    logging.debug(f"Processed {len([s for s in all_sources if s == 'camera'])} camera points")
            
            except Exception as e:
                logging.debug(f"Error processing depth camera data: {e}")
            
            return np.array(all_points), all_sources
            
        except Exception as e:
            logging.error(f"Error in fused sensor data processing: {e}")
            return np.array([]).reshape(0, 3), []


# === INTEGRATION INSTRUCTIONS ===
"""
To integrate this depth camera functionality into your existing Scanner class:

1. Add these parameters to Scanner.__init__():
   ```python
   def __init__(self, client, scan_range=20.0, enable_visualization=False, voxel_size=0.5, 
                enable_screenshots=True, enable_depth_camera=True, depth_camera_name="front_center"):
       # ... existing code ...
       self.enable_depth_camera = enable_depth_camera
       self.depth_camera_name = depth_camera_name
       
       # Sensor positioning from settings.json
       self.camera_offset = np.array([0.25, 0.0, 0.0])  # 25cm forward
       self.lidar_offset = np.array([0.0, 0.0, -0.1])   # 10cm down
       
       # Depth camera parameters
       self.depth_image_width = 640
       self.depth_image_height = 480
       self.depth_camera_fov = 90.0
       
       # Camera intrinsics
       self.fx = self.depth_image_width / (2.0 * np.tan(np.radians(self.depth_camera_fov / 2.0)))
       self.fy = self.fx
       self.cx = self.depth_image_width / 2.0
       self.cy = self.depth_image_height / 2.0
   ```

2. Add the depth camera methods to your Scanner class:
   - get_depth_camera_data()
   - transform_camera_points_to_global()
   - get_fused_sensor_data()

3. Modify fetch_density_distances() to use sensor fusion:
   In the LiDAR data processing section, replace:
   ```python
   lidar_data = self.client.getLidarData()
   # ... process lidar_data ...
   ```
   
   With:
   ```python
   lidar_data = self.client.getLidarData()
   
   if self.enable_depth_camera:
       # Use fused sensor data
       all_points, point_sources = self.get_fused_sensor_data(
           lidar_data, drone_pos, drone_quat, self.scan_range
       )
       
       # Convert to the expected format for voxelization
       all_global_points = [
           {'position': all_points[i], 'distance': np.linalg.norm(all_points[i] - drone_pos), 'source': point_sources[i]}
           for i in range(len(all_points))
       ]
   else:
       # Use only LiDAR data (existing code)
       # ... existing LiDAR processing ...
   ```

4. Update the Scanner creation calls to enable depth camera:
   ```python
   scanner = Scanner(client, scan_range=20.0, enable_depth_camera=True)
   ```

This integration will:
- Automatically fuse LiDAR and depth camera data
- Account for sensor position differences (LiDAR 10cm down, camera 25cm forward)
- Convert depth images to 3D point clouds
- Apply the same voxelization process to combined sensor data
- Maintain compatibility with existing visualization and clustering code

The fused data will provide better obstacle detection, especially:
- Camera provides dense data in the forward direction
- LiDAR provides 360° coverage
- Combined data improves obstacle detection accuracy and coverage
""" 