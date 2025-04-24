module AirSimInterface

export initialize_airsim, get_sensor_data, send_action

using PyCall
using StaticArrays
using ..StateSpace

"""
    initialize_airsim()

Initialize the AirSim client via Python.
Returns a Python AirSim client object.
"""
function initialize_airsim()
    # Import Python modules
    py"""
    import sys
    import os
    import numpy as np
    import time
    
    # Import AirSim
    try:
        import airsim
    except ImportError:
        print("ERROR: AirSim module not found. Please install it using:")
        print("pip install airsim")
        sys.exit(1)
    
    class AirSimDroneClient:
        def __init__(self):
            # Connect to AirSim
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            # Initialize drone
            self.reset()
            
            # Create data directory for saving sensor data if needed
            os.makedirs("data/camera", exist_ok=True)
            os.makedirs("data/lidar", exist_ok=True)
        
        def reset(self):
            # Reset drone position
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
            time.sleep(1)
        
        def get_position(self):
            # Get current position as numpy array
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            return np.array([pos.x_val, pos.y_val, pos.z_val])
        
        def move_by_velocity(self, vx, vy, vz, duration=1):
            # Move drone by velocity vector
            self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
        
        def move_to_position(self, x, y, z, velocity=5):
            # Move drone to absolute position
            self.client.moveToPositionAsync(x, y, z, velocity).join()
        
        def move_by_relative_position(self, dx, dy, dz):
            # Move drone by relative displacement
            current = self.get_position()
            self.client.moveToPositionAsync(
                current[0] + dx, 
                current[1] + dy, 
                current[2] + dz, 
                2
            ).join()
        
        def get_lidar_data(self):
            # Get LiDAR point cloud
            lidar_data = self.client.getLidarData()
            
            if len(lidar_data.point_cloud) < 3:
                return np.array([]).reshape(0, 3)
                
            points = np.array(lidar_data.point_cloud).reshape(-1, 3)
            return points
        
        def get_rgb_image(self):
            # Get RGB camera image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene)
            ])
            
            if not responses:
                return None
                
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
            return img_rgb
        
        def get_depth_image(self):
            # Get depth image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis)
            ])
            
            if not responses:
                return None
                
            response = responses[0]
            
            if response.pixels_as_float:
                img1d = np.array(response.image_data_float, dtype=np.float32)
                img_depth = img1d.reshape(response.height, response.width)
            else:
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_depth = img1d.reshape(response.height, response.width)
            
            return img_depth
        
        def get_sensor_data(self):
            # Get all sensor data
            position = self.get_position()
            lidar_points = self.get_lidar_data()
            rgb_image = self.get_rgb_image()
            depth_image = self.get_depth_image()
            
            return {
                "position": position,
                "lidar_points": lidar_points,
                "rgb_image": rgb_image,
                "depth_image": depth_image
            }
    
    # Create global instance
    drone_client = AirSimDroneClient()
    """
    
    # Return the Python client
    return py"drone_client"
end

"""
    get_point_cloud(client)

Get the point cloud data from AirSim.
"""
function get_point_cloud(client)
    # Call the Python method
    py_points = client.get_lidar_data()
    
    # Convert to Julia array of StaticVectors
    if py_points.size == 0
        return SVector{3, Float64}[]
    end
    
    # Convert each point to SVector
    point_cloud = [SVector{3, Float64}(py_points[i,:]) for i in 1:size(py_points, 1)]
    
    return point_cloud
end

"""
    get_sensor_data(client, target_position)

Get sensor data and convert to DroneObservation.
"""
function get_sensor_data(client, target_position)
    # Get raw sensor data
    data = client.get_sensor_data()
    
    # Extract components
    position = SVector{3, Float64}(data["position"])
    point_cloud = get_point_cloud(client)
    
    # Create observation
    observation = StateSpace.create_observation_from_sensors(
        position,
        target_position,
        point_cloud
    )
    
    return observation
end

"""
    send_action(client, action)

Send an action (3D vector) to AirSim for execution.
"""
function send_action(client, action)
    # Extract components
    dx, dy, dz = action
    
    # Execute the action
    client.move_by_relative_position(dx, dy, dz)
    
    # Get updated position
    new_position = client.get_position()
    
    return SVector{3, Float64}(new_position)
end

end # module
