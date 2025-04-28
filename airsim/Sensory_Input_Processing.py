import airsim
import numpy as np
import time
from math import pi
import msgpackrpc
from sklearn.cluster import DBSCAN
from collections import defaultdict

class EnvironmentScanner:
    def __init__(self, client=None):
        self.client = client or airsim.MultirotorClient()
        try:
            self.client.confirmConnection()
        except Exception as e:
            raise Exception(f"Failed to connect to AirSim: {str(e)}")

    def get_obstacle_coordinates(self):
        """Main function to perform sensor sweep and return obstacle coordinates"""
        try:
            points = self.collect_sensor_data()
            
            if points is None or len(points) == 0:
                raise Exception("No valid point cloud data collected")
                
            voxel_coordinates = self.create_obstacle_voxel_grid(points)
            
            if voxel_coordinates is None or len(voxel_coordinates) == 0:
                raise Exception("No valid voxels created")
            
            nearest_obstacles = self.find_nearest_obstacles(voxel_coordinates)
            
            # Create lists of coordinates for each obstacle without printing
            obstacle_lists = []
            for obstacle in nearest_obstacles:
                coords = [(round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2)) 
                         for p in obstacle['points']]
                obstacle_lists.append(coords)
                
            return obstacle_lists
            
        except Exception as e:
            print(f"Error getting obstacle coordinates: {str(e)}")
            return []

    def fetch_density_distances(self):
        """
        Returns the number of obstacles detected and their distances from the drone.
        
        Returns:
            tuple: (number of obstacles, list of distances)
        """
        try:
            obstacle_lists = self.get_obstacle_coordinates()
            num_obstacles = len(obstacle_lists)
            
            # Use find_nearest_obstacles to get distances since it's already calculated there
            obstacle_info = self.find_nearest_obstacles(np.array([coord for obstacle in obstacle_lists for coord in obstacle]))
            distances = [obs['distance'] for obs in obstacle_info]
            
            return num_obstacles, distances
            
        except Exception as e:
            print(f"Error getting obstacle summary: {str(e)}")
            return 0, []

    def collect_sensor_data(self):
        """Collect both LiDAR and camera data from current position with enhanced vertical detection"""
        try:
            # Get drone's pose
            drone_pose = self.client.simGetVehiclePose(vehicle_name="Drone1")
            
            # Get LiDAR data
            lidar_data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name="Drone1")
            
            # Get camera depth data
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True)
            ], vehicle_name="Drone1")
            
            if len(lidar_data.point_cloud) < 3:
                return None
                
            if not responses:
                return None
            
            # Process LiDAR points - transform to correct frame
            lidar_points = np.array(lidar_data.point_cloud).reshape((-1, 3))
            
            # Filter LiDAR points by FOV - but with expanded vertical range to better capture trees
            # Expand vertical FOV from [-10,10] to [-15,20] to better capture tall trees
            lidar_angles = np.arctan2(lidar_points[:, 1], lidar_points[:, 0]) * 180 / np.pi
            vertical_angles = np.arctan2(lidar_points[:, 2], np.sqrt(lidar_points[:, 0]**2 + lidar_points[:, 1]**2)) * 180 / np.pi
            
            valid_lidar = np.logical_and(
                np.abs(lidar_angles) < 90,
                np.logical_and(vertical_angles >= -15, vertical_angles <= 20)  # Expanded vertical FOV
            )
            lidar_points = lidar_points[valid_lidar]
            
            # Process depth image with improved point density
            depth_response = responses[0]
            depth_img = airsim.list_to_2d_float_array(depth_response.image_data_float,
                                                     depth_response.width,
                                                     depth_response.height)
            
            # Convert depth to 3D points in camera frame with denser sampling for vertical structures
            height, width = depth_img.shape
            fov = 90  # from settings.json
            aspect = float(width) / height
            f = width / (2 * np.tan(fov * pi / 360))
            
            # Vertical scan lines to better detect trees
            # Sample denser in vertical direction with every other row, but sparser horizontally
            camera_points = []
            vertical_lines = []
            
            # First pass - create regular grid but denser than before
            step_h = 2  # Sample every other row (higher vertical resolution)
            step_w = 2  # Sample every other column
            
            for i in range(0, height, step_h):
                for j in range(0, width, step_w):
                    depth = depth_img[i, j]
                    if depth > 0 and depth < 100:
                        x = depth
                        y = -(j - width/2) * depth / f
                        z = -(i - height/2) * depth / (f/aspect)
                        
                        if x > 0:
                            camera_points.append([x, y, z])
            
            # Second pass - detect vertical structures
            # Look for columns with multiple depth values - indicating vertical structures
            # Sample a few vertical lines directly
            n_vert_lines = 10
            for j in range(width//4, 3*width//4, width//(n_vert_lines+1)):  # Sample across middle section of image
                j = int(j)
                col_points = []
                for i in range(0, height, 1):  # Dense vertical sampling (every row)
                    depth = depth_img[i, j]
                    if depth > 0 and depth < 100:
                        x = depth
                        y = -(j - width/2) * depth / f
                        z = -(i - height/2) * depth / (f/aspect)
                        
                        if x > 0:
                            col_points.append([x, y, z])
                
                # Add points from this column to our vertical scan data
                if col_points:
                    vertical_lines.extend(col_points)
            
            # Convert to numpy arrays
            camera_points = np.array(camera_points) if camera_points else np.empty((0, 3))
            vertical_lines = np.array(vertical_lines) if vertical_lines else np.empty((0, 3))
            
            # Filter out points that are too far or too close
            max_distance = 40.0
            min_distance = 0.5
            
            if len(camera_points) > 0:
                camera_points = camera_points[np.logical_and(
                    np.linalg.norm(camera_points, axis=1) > min_distance,
                    np.linalg.norm(camera_points, axis=1) < max_distance
                )]
            
            if len(vertical_lines) > 0:
                vertical_lines = vertical_lines[np.logical_and(
                    np.linalg.norm(vertical_lines, axis=1) > min_distance,
                    np.linalg.norm(vertical_lines, axis=1) < max_distance
                )]
                print(f"Added {len(vertical_lines)} points from vertical scan lines")
            
            if len(lidar_points) > 0:
                lidar_points = lidar_points[np.logical_and(
                    np.linalg.norm(lidar_points, axis=1) > min_distance,
                    np.linalg.norm(lidar_points, axis=1) < max_distance
                )]
            
            # Combine points
            all_points = []
            if len(lidar_points) > 0:
                all_points.append(lidar_points)
            if len(camera_points) > 0:
                all_points.append(camera_points)
            if len(vertical_lines) > 0:
                all_points.append(vertical_lines)  # Add our vertical scan data
                
            if not all_points:
                raise Exception("No valid points collected")
                
            combined_points = np.vstack(all_points)
            print(f"Collected {len(combined_points)} points: {len(lidar_points)} LiDAR, {len(camera_points)} camera, {len(vertical_lines)} vertical")
            return combined_points
            
        except Exception as e:
            print(f"Error collecting sensor data: {str(e)}")
            return None

    def create_obstacle_voxel_grid(self, points, voxel_size=0.4):
        """Create voxel grid and identify obstacle voxels with enhanced vertical structure detection"""
        try:
            # Define grid parameters with expanded vertical range to better capture trees
            grid_bounds = {
                'x_min': -25, 'x_max': 25,  # Expanded horizontal range
                'y_min': -25, 'y_max': 25,  # Expanded horizontal range
                'z_min': -7, 'z_max': 20    # Expanded vertical range to better capture tall trees
            }
            
            # Initialize voxel storage using dictionary to merge points in same voxel
            voxel_dict = {}
            
            # Add tracking for vertical structures to better detect trees
            vertical_structure_dict = defaultdict(int)
            
            # Convert points to voxel coordinates
            for point in points:
                x_idx = int((point[0] - grid_bounds['x_min']) / voxel_size)
                y_idx = int((point[1] - grid_bounds['y_min']) / voxel_size)
                z_idx = int((point[2] - grid_bounds['z_min']) / voxel_size)
                
                # Store both voxel indices and real-world coordinates with rounding
                if (0 <= x_idx and 0 <= y_idx and 0 <= z_idx):
                    voxel_key = (x_idx, y_idx, z_idx)
                    voxel_coord = (
                        round(x_idx * voxel_size + grid_bounds['x_min'] + voxel_size/2, 2),
                        round(y_idx * voxel_size + grid_bounds['y_min'] + voxel_size/2, 2),
                        round(z_idx * voxel_size + grid_bounds['z_min'] + voxel_size/2, 2)
                    )
                    # Only store one coordinate per voxel space
                    voxel_dict[voxel_key] = voxel_coord
                    
                    # Track vertical structures by counting points in the same xy column
                    # This helps identify trees and other vertical obstacles
                    column_key = (x_idx, y_idx)
                    vertical_structure_dict[column_key] += 1
            
            # Count vertical structures identified
            vertical_columns = 0
            for column_key, count in vertical_structure_dict.items():
                if count >= 3:  # Consider columns with 3+ points as potential vertical structures
                    vertical_columns += 1
            
            if vertical_columns > 0:
                print(f"Detected {vertical_columns} potential vertical structures (trees/poles)")
                
            # Create a list of regular voxel coordinates first
            voxel_coordinates = list(voxel_dict.values())
            
            # Add additional points to emphasize vertical structures like trees
            # This makes them more likely to be detected as obstacles
            enhanced_voxels = []
            
            for voxel_key, voxel_coord in voxel_dict.items():
                x_idx, y_idx, _ = voxel_key
                column_key = (x_idx, y_idx)
                
                # If this column has multiple voxels stacked vertically (likely a tree or pole)
                if vertical_structure_dict[column_key] >= 3:
                    # Double the weight of this voxel by adding it again
                    enhanced_voxels.append(voxel_coord)
                    
                    # For very tall structures, add even more emphasis
                    if vertical_structure_dict[column_key] >= 5:
                        enhanced_voxels.append(voxel_coord)
            
            # Add the enhanced voxels to our coordinate list
            if enhanced_voxels:
                voxel_coordinates.extend(enhanced_voxels)
                print(f"Enhanced {len(enhanced_voxels)} voxels for better vertical structure detection")
                
            return np.array(voxel_coordinates)
            
        except Exception as e:
            print(f"Error creating voxel grid: {str(e)}")
            return None

    def find_nearest_obstacles(self, voxel_coordinates, n_obstacles=3):
        """Find the nearest n obstacles using clustering and direct distance calculation
        with improved vertical structure detection"""
        try:
            if len(voxel_coordinates) == 0:
                return []
                
            # Use DBSCAN with smaller eps for finer clustering to better separate individual trees
            clustering = DBSCAN(eps=0.7, min_samples=3).fit(voxel_coordinates)
            labels = clustering.labels_
            
            # Count number of clusters found
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"Found {n_clusters} potential obstacle clusters and {n_noise} noise points")
            
            # Group voxels by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                if label != -1:  # Ignore noise points
                    clusters[label].append(voxel_coordinates[i])
            
            # Calculate cluster info with direct distance calculation
            obstacle_info = []
            min_voxels = 8  # Reduced from 10 to catch thinner obstacles
            
            for label, points in clusters.items():
                points_array = np.array(points)
                
                if len(points_array) < min_voxels:
                    continue
                
                # Check if this cluster represents a vertical structure (like a tree)
                # by examining height distribution along z-axis
                z_values = points_array[:, 2]
                height_range = max(z_values) - min(z_values) if len(z_values) > 0 else 0
                
                # Consider an object a vertical structure if it spans at least 1.5m in height
                is_vertical = height_range > 1.5
                
                # Use a smaller reduction threshold for vertical objects to preserve their structure
                reduction_threshold = 0.5 if is_vertical else 0.6
                reduced_points = self.reduce_points(points_array, reduction_threshold=reduction_threshold)
                
                if len(reduced_points) < 4:  # Lower threshold for vertical structures
                    continue
                
                # Get horizontal and vertical extent of the obstacle
                x_range = max(points_array[:, 0]) - min(points_array[:, 0]) if len(points_array) > 0 else 0
                y_range = max(points_array[:, 1]) - min(points_array[:, 1]) if len(points_array) > 0 else 0
                
                # Calculate distance to this obstacle
                distances = np.linalg.norm(reduced_points, axis=1)
                min_distance = round(float(np.min(distances)), 2)
                
                # Calculate approximate volume (useful for classification)
                volume = x_range * y_range * height_range if x_range > 0 and y_range > 0 and height_range > 0 else 0
                
                # If this is a vertical structure, log it for debugging
                if is_vertical:
                    print(f"Identified vertical structure: height={height_range:.1f}m, distance={min_distance:.2f}m")
                
                obstacle_info.append({
                    'distance': min_distance,
                    'points': reduced_points,
                    'is_vertical': is_vertical,
                    'height': height_range,
                    'width': max(x_range, y_range),
                    'volume': volume,
                    'point_count': len(points_array)
                })
            
            if not obstacle_info:
                return []
            
            # Sort obstacles primarily by distance
            obstacle_info.sort(key=lambda x: x['distance'])
            
            # Increased n_obstacles from 2 to 3 to capture more obstacles
            return obstacle_info[:n_obstacles]
            
        except Exception as e:
            print(f"Error finding nearest obstacles: {str(e)}")
            return []

    def reduce_points(self, points, reduction_threshold=0.6):
        """Reduce number of points by combining those that are very close together"""
        if len(points) == 0:
            return points
            
        reduced = []
        used = set()
        
        for i in range(len(points)):
            if i in used:
                continue
                
            current = points[i]
            cluster = [current]
            used.add(i)
            
            # Find points close to current point
            for j in range(i + 1, len(points)):
                if j not in used and np.linalg.norm(current - points[j]) < reduction_threshold:
                    cluster.append(points[j])
                    used.add(j)
            
            # Average the cluster points
            reduced.append(np.mean(cluster, axis=0))
        
        return np.array(reduced)