module Fusion

export fuse_sensor_data, create_point_cloud, voxelize_point_cloud

using Images
using StaticArrays
using LinearAlgebra

"""
    create_point_cloud(depth_image, camera_params)

Create a 3D point cloud from a depth image using camera parameters.
Returns a matrix of points (Nx3).
"""
function create_point_cloud(depth_image::AbstractMatrix{<:Real}; 
                           fov_h=90.0, 
                           fov_v=60.0, 
                           near_clip=0.1, 
                           far_clip=100.0)
    height, width = size(depth_image)
    
    # Camera parameters
    fx = width / (2 * tan(deg2rad(fov_h) / 2))
    fy = height / (2 * tan(deg2rad(fov_v) / 2))
    cx = width / 2
    cy = height / 2
    
    # Initialize point cloud arrays
    points = Vector{SVector{3, Float32}}()
    sizehint!(points, height * width)
    
    # Generate point cloud
    for v in 1:height
        for u in 1:width
            depth = depth_image[v, u]
            
            # Skip invalid depth values
            if depth <= near_clip || depth >= far_clip
                continue
            end
            
            # Calculate 3D coordinates
            x = depth * (u - cx) / fx
            y = depth * (v - cy) / fy
            z = depth
            
            push!(points, SVector{3, Float32}(x, y, z))
        end
    end
    
    # Convert to matrix for easier handling
    return length(points) > 0 ? hcat(points...) : zeros(Float32, 3, 0)
end

"""
    transform_point_cloud(points, transform)

Apply a 4x4 transformation matrix to a point cloud.
"""
function transform_point_cloud(points::AbstractMatrix, transform::AbstractMatrix)
    if size(points, 1) != 3
        error("Points must be a 3xN matrix")
    end
    
    # Extract rotation and translation from transformation matrix
    R = transform[1:3, 1:3]
    t = transform[1:3, 4]
    
    # Apply transformation
    transformed_points = R * points .+ t
    
    return transformed_points
end

"""
    fuse_sensor_data(rgb_image, depth_image, lidar_points)

Fuse camera and LiDAR data into a colored point cloud.
Returns a tuple of (points, colors) where:
- points is a 3xN matrix of 3D coordinates
- colors is a 3xN matrix of RGB values for each point
"""
function fuse_sensor_data(rgb_image, depth_image, lidar_points)
    if isnothing(rgb_image) || isnothing(depth_image) || isnothing(lidar_points)
        @warn "Missing sensor data, cannot fuse"
        return nothing, nothing
    end
    
    # Create point cloud from depth image
    camera_points = create_point_cloud(depth_image)
    
    # Transform LiDAR points to camera frame (assuming already calibrated)
    # In real applications, you would need to apply a proper calibration matrix
    lidar_to_camera = [
        1.0 0.0 0.0 0.0;
        0.0 1.0 0.0 0.0;
        0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 1.0
    ]
    
    transformed_lidar = transform_point_cloud(
        lidar_points', lidar_to_camera
    )
    
    # Combine point clouds
    combined_points = hcat(camera_points, transformed_lidar)
    
    # Extract colors for camera points from RGB image
    colors = zeros(Float32, 3, size(camera_points, 2) + size(transformed_lidar, 2))
    
    # Process camera points
    height, width = size(rgb_image)
    camera_count = size(camera_points, 2)
    
    for i in 1:camera_count
        x, y, z = camera_points[:, i]
        
        # Project back to image coordinates
        u = round(Int, x * fx / z + cx)
        v = round(Int, y * fy / z + cy)
        
        # Check if within image bounds
        if 1 <= u <= width && 1 <= v <= height
            pixel = rgb_image[v, u]
            colors[:, i] = [red(pixel), green(pixel), blue(pixel)]
        end
    end
    
    # LiDAR points get a default color (e.g., white)
    colors[:, (camera_count+1):end] .= 1.0
    
    return combined_points, colors
end

"""
    voxelize_point_cloud(points, colors; voxel_size=0.1)

Voxelize a point cloud to reduce density while preserving structure.
Returns a tuple of (voxelized_points, voxelized_colors).
"""
function voxelize_point_cloud(points::AbstractMatrix, colors::AbstractMatrix; voxel_size=0.1)
    if size(points, 2) == 0
        return points, colors
    end
    
    # Calculate voxel indices for each point
    voxel_indices = floor.(Int, points ./ voxel_size)
    
    # Create dictionary to store voxels
    voxels = Dict{NTuple{3, Int}, Tuple{SVector{3, Float32}, SVector{3, Float32}, Int}}()
    
    # Aggregate points into voxels
    for i in 1:size(points, 2)
        idx = (voxel_indices[1, i], voxel_indices[2, i], voxel_indices[3, i])
        point = SVector{3, Float32}(points[:, i])
        color = SVector{3, Float32}(colors[:, i])
        
        if haskey(voxels, idx)
            p, c, count = voxels[idx]
            voxels[idx] = (p + point, c + color, count + 1)
        else
            voxels[idx] = (point, color, 1)
        end
    end
    
    # Calculate average point and color for each voxel
    voxelized_points = zeros(Float32, 3, length(voxels))
    voxelized_colors = zeros(Float32, 3, length(voxels))
    
    for (i, (_, (point, color, count))) in enumerate(voxels)
        voxelized_points[:, i] = point / count
        voxelized_colors[:, i] = color / count
    end
    
    return voxelized_points, voxelized_colors
end

end # module
