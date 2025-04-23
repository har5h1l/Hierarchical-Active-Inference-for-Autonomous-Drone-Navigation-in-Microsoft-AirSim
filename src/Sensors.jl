module Sensors

export read_camera_data, read_lidar_data, get_latest_data_timestamp

using Images, FileIO
using PyCall

"""
    get_latest_data_timestamp()

Read the status file to get the timestamp of the latest sensor data.
Returns the timestamp as an integer if available, otherwise returns nothing.
"""
function get_latest_data_timestamp(status_file="../data/status.txt")
    if isfile(status_file)
        try
            timestamp = parse(Int, strip(read(status_file, String)))
            return timestamp
        catch e
            @warn "Failed to parse timestamp from status file: $e"
        end
    end
    return nothing
end

"""
    read_camera_data(timestamp=nothing)

Read the camera data (RGB and depth) from files.
If timestamp is nothing, use the latest available data.
Returns a tuple of (rgb_image, depth_image) or (nothing, nothing) if not available.
"""
function read_camera_data(timestamp=nothing; data_dir="../data/camera")
    if timestamp === nothing
        timestamp = get_latest_data_timestamp()
        if timestamp === nothing
            @warn "No timestamp available, cannot read camera data"
            return nothing, nothing
        end
    end
    
    rgb_path = joinpath(data_dir, "$(timestamp)_rgb.png")
    depth_path = joinpath(data_dir, "$(timestamp)_depth.png")
    
    rgb_img = nothing
    depth_img = nothing
    
    if isfile(rgb_path)
        try
            rgb_img = load(rgb_path)
        catch e
            @warn "Failed to load RGB image: $e"
        end
    end
    
    if isfile(depth_path)
        try
            depth_img = load(depth_path)
            # Convert depth image to float matrix if needed
            if typeof(depth_img) <: AbstractArray{<:ColorTypes.Color}
                depth_img = Float32.(Gray.(depth_img))
            end
        catch e
            @warn "Failed to load depth image: $e"
        end
    end
    
    return rgb_img, depth_img
end

"""
    read_lidar_data(timestamp=nothing)

Read the LiDAR point cloud data from file.
If timestamp is nothing, use the latest available data.
Returns a matrix of points (Nx3) or nothing if not available.
"""
function read_lidar_data(timestamp=nothing; data_dir="../data/lidar")
    if timestamp === nothing
        timestamp = get_latest_data_timestamp()
        if timestamp === nothing
            @warn "No timestamp available, cannot read LiDAR data"
            return nothing
        end
    end
    
    lidar_path = joinpath(data_dir, "$(timestamp)_lidar.npy")
    
    if isfile(lidar_path)
        try
            # Use PyCall to load numpy files
            np = pyimport("numpy")
            points = np.load(lidar_path)
            # Convert to Julia array
            return Array(points)
        catch e
            @warn "Failed to load LiDAR data: $e"
        end
    end
    
    return nothing
end

"""
    read_sensor_data(timestamp=nothing)

Read both camera and LiDAR data for a given timestamp.
If timestamp is nothing, use the latest available data.
Returns a tuple of (rgb_image, depth_image, lidar_points) or (nothing, nothing, nothing) if not available.
"""
function read_sensor_data(timestamp=nothing)
    if timestamp === nothing
        timestamp = get_latest_data_timestamp()
    end
    
    rgb_img, depth_img = read_camera_data(timestamp)
    lidar_points = read_lidar_data(timestamp)
    
    return rgb_img, depth_img, lidar_points, timestamp
end

end # module
