#!/usr/bin/env python3
"""
AirSim Drone Control Script
Provides functions to move drone to target and collect LiDAR & camera data
"""

import os
import sys
import time
import numpy as np
import airsim
import cv2
from utils import save_image, save_lidar, get_target_position

class DroneController:
    def __init__(self, data_path='../data'):
        """Initialize AirSim client and connect to the drone"""
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Ensure data directory exists
        self.data_path = data_path
        os.makedirs(os.path.join(self.data_path, 'camera'), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, 'lidar'), exist_ok=True)
        
        # Initialize drone
        self.reset_drone()
        
    def reset_drone(self):
        """Reset drone to starting position"""
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        
    def move_to_position(self, x, y, z, velocity=5):
        """Move drone to specified position with given velocity"""
        self.client.moveToPositionAsync(x, y, z, velocity).join()
        
    def move_to_target(self, target_position):
        """Move drone towards target position"""
        current = self.client.getMultirotorState().kinematics_estimated.position
        target = airsim.Vector3r(target_position[0], 
                                target_position[1], 
                                target_position[2])
        
        # Calculate direction vector
        direction = airsim.Vector3r(target.x_val - current.x_val,
                                  target.y_val - current.y_val,
                                  target.z_val - current.z_val)
        
        # Normalize and scale by step size
        length = np.sqrt(direction.x_val**2 + direction.y_val**2 + direction.z_val**2)
        if length > 0.1:  # Only move if there's meaningful distance
            step_size = min(2.0, length)  # Don't overshoot
            direction.x_val *= step_size / length
            direction.y_val *= step_size / length
            direction.z_val *= step_size / length
            
            # Move incrementally
            self.move_to_position(current.x_val + direction.x_val,
                                current.y_val + direction.y_val,
                                current.z_val + direction.z_val)
            return False  # Not arrived yet
        else:
            return True  # Arrived at target
    
    def collect_sensor_data(self, timestamp=None):
        """Collect LiDAR and camera data and save to disk"""
        if timestamp is None:
            timestamp = int(time.time() * 1000)  # milliseconds
            
        # Get camera images
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene),
            airsim.ImageRequest("0", airsim.ImageType.DepthVis)
        ])
        
        # Save RGB image
        if responses[0].pixels_as_float:
            img_rgb = airsim.list_to_2d_float_array(responses[0].image_data_float, 
                                                  responses[0].width, 
                                                  responses[0].height)
        else:
            img_rgb = cv2.imdecode(np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8), 
                                 cv2.IMREAD_COLOR)
        save_image(img_rgb, os.path.join(self.data_path, 'camera', f"{timestamp}_rgb.png"))
        
        # Save depth image
        if responses[1].pixels_as_float:
            img_depth = airsim.list_to_2d_float_array(responses[1].image_data_float, 
                                                    responses[1].width, 
                                                    responses[1].height)
            save_image(img_depth, os.path.join(self.data_path, 'camera', f"{timestamp}_depth.png"))
        
        # Get LiDAR data
        lidar_data = self.client.getLidarData()
        if len(lidar_data.point_cloud) > 3:
            points = np.array(lidar_data.point_cloud).reshape(-1, 3)
            save_lidar(points, os.path.join(self.data_path, 'lidar', f"{timestamp}_lidar.npy"))
        
        # Create status file to signal new data is available
        with open(os.path.join(self.data_path, 'status.txt'), 'w') as f:
            f.write(f"{timestamp}\n")
        
        return timestamp

def main():
    """Main function for testing drone control independently"""
    controller = DroneController()
    print("Drone controller initialized")
    
    # Get target from command line or use default
    target_position = get_target_position()
    print(f"Moving to target: {target_position}")
    
    # Control loop
    arrived = False
    while not arrived:
        arrived = controller.move_to_target(target_position)
        timestamp = controller.collect_sensor_data()
        print(f"Collected data at {timestamp}, distance to target: {not arrived}")
        time.sleep(0.1)
    
    print("Arrived at target")
    controller.client.armDisarm(False)
    controller.client.enableApiControl(False)

if __name__ == "__main__":
    main()
