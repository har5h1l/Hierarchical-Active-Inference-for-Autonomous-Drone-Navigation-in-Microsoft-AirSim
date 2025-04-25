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
        """Collect both LiDAR and camera data from current position"""
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
            
            # Filter LiDAR points by FOV
            lidar_angles = np.arctan2(lidar_points[:, 1], lidar_points[:, 0]) * 180 / np.pi
            vertical_angles = np.arctan2(lidar_points[:, 2], np.sqrt(lidar_points[:, 0]**2 + lidar_points[:, 1]**2)) * 180 / np.pi
            
            valid_lidar = np.logical_and(
                np.abs(lidar_angles) < 90,
                np.logical_and(vertical_angles >= -10, vertical_angles <= 10)
            )
            lidar_points = lidar_points[valid_lidar]
            
            # Process depth image
            depth_response = responses[0]
            depth_img = airsim.list_to_2d_float_array(depth_response.image_data_float,
                                                     depth_response.width,
                                                     depth_response.height)
            
            # Convert depth to 3D points in camera frame
            height, width = depth_img.shape
            fov = 90  # from settings.json
            aspect = float(width) / height
            f = width / (2 * np.tan(fov * pi / 360))
            
            camera_points = []
            for i in range(0, height, 2):
                for j in range(0, width, 2):
                    depth = depth_img[i, j]
                    if depth > 0 and depth < 100:
                        x = depth
                        y = -(j - width/2) * depth / f
                        z = -(i - height/2) * depth / (f/aspect)
                        
                        if x > 0:
                            camera_points.append([x, y, z])
            
            camera_points = np.array(camera_points)
            
            # Filter out points that are too far or too close
            max_distance = 40.0
            min_distance = 0.5
            
            if len(camera_points) > 0:
                camera_points = camera_points[np.logical_and(
                    np.linalg.norm(camera_points, axis=1) > min_distance,
                    np.linalg.norm(camera_points, axis=1) < max_distance
                )]
            
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
                
            if not all_points:
                raise Exception("No valid points collected")
                
            combined_points = np.vstack(all_points)
            return combined_points
            
        except Exception as e:
            print(f"Error collecting sensor data: {str(e)}")
            return None

    def create_obstacle_voxel_grid(self, points, voxel_size=0.6):
        """Create voxel grid and identify obstacle voxels"""
        try:
            # Define grid parameters
            grid_bounds = {
                'x_min': -20, 'x_max': 20,
                'y_min': -20, 'y_max': 20,
                'z_min': -5, 'z_max': 15
            }
            
            # Initialize voxel storage using dictionary to merge points in same voxel
            voxel_dict = {}
            
            # Convert points to voxel coordinates silently
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
            
            # Convert dictionary values to array
            voxel_coordinates = np.array(list(voxel_dict.values()))
            return voxel_coordinates
            
        except Exception as e:
            print(f"Error creating voxel grid: {str(e)}")
            return None

    def find_nearest_obstacles(self, voxel_coordinates, n_obstacles=2):
        """Find the nearest n obstacles using clustering and direct distance calculation"""
        try:
            if len(voxel_coordinates) == 0:
                return []
                
            # Use DBSCAN with larger eps for coarser clustering
            clustering = DBSCAN(eps=0.8, min_samples=3).fit(voxel_coordinates)
            labels = clustering.labels_
            
            # Group voxels by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                if label != -1:  # Ignore noise points
                    clusters[label].append(voxel_coordinates[i])
            
            # Calculate cluster info with direct distance calculation
            obstacle_info = []
            min_voxels = 10  # Minimum number of voxels to consider as valid obstacle
            
            for label, points in clusters.items():
                points_array = np.array(points)
                
                if len(points_array) < min_voxels:
                    continue
                
                reduced_points = self.reduce_points(points_array, reduction_threshold=0.6)
                
                if len(reduced_points) < 5:
                    continue
                
                distances = np.linalg.norm(reduced_points, axis=1)
                min_distance = round(float(np.min(distances)), 2)
                
                obstacle_info.append({
                    'distance': min_distance,
                    'points': reduced_points
                })
            
            if not obstacle_info:
                return []
            
            obstacle_info.sort(key=lambda x: x['distance'])
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