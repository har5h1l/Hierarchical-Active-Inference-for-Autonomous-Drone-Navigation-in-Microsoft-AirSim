#!/usr/bin/env python3
"""
Utility functions for AirSim drone control
"""

import os
import sys
import numpy as np
import cv2
import argparse

def save_image(img, filepath):
    """Save image to disk"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Handle different image types
    if isinstance(img, np.ndarray) and img.dtype == np.float32:
        # Normalize float images for visualization
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    
    # Save image
    cv2.imwrite(filepath, img)
    return filepath

def save_lidar(points, filepath):
    """Save LiDAR point cloud to disk"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save points as numpy array
    np.save(filepath, points)
    return filepath

def load_lidar(filepath):
    """Load LiDAR point cloud from disk"""
    if os.path.exists(filepath):
        return np.load(filepath)
    return None

def load_image(filepath):
    """Load image from disk"""
    if os.path.exists(filepath):
        return cv2.imread(filepath)
    return None

def get_target_position(args=None):
    """Get target position from command line or return default"""
    parser = argparse.ArgumentParser(description='Drone target position')
    parser.add_argument('--x', type=float, default=10.0,
                        help='X coordinate of target')
    parser.add_argument('--y', type=float, default=10.0,
                        help='Y coordinate of target')
    parser.add_argument('--z', type=float, default=-5.0,
                        help='Z coordinate of target')
    
    # Parse arguments or use provided args
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    return [args.x, args.y, args.z]

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_angle(point1, point2, reference_vector=[1, 0, 0]):
    """Calculate angle between direction vector and reference vector"""
    direction = np.array(point2) - np.array(point1)
    direction_2d = np.array([direction[0], direction[1], 0])
    reference = np.array(reference_vector)
    
    # Normalize vectors
    direction_2d = direction_2d / np.linalg.norm(direction_2d) if np.linalg.norm(direction_2d) > 0 else direction_2d
    reference = reference / np.linalg.norm(reference)
    
    # Calculate angle using dot product
    dot_product = np.dot(direction_2d[:2], reference[:2])
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Determine sign using cross product
    cross = np.cross(reference[:2], direction_2d[:2])
    if cross < 0:
        angle = -angle
    
    return angle
