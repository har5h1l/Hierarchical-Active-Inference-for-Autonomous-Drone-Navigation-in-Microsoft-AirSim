module State

export State, extract_state, get_distance_to_target, get_angle_to_target, detect_obstacles, get_obstacle_voxels

using LinearAlgebra
using StaticArrays

"""
State structure representing the drone's state relative to environment
Aligned with the active inference model states:
- S1: Distance from target (scalar)
- S2: Angle/direction vector to target (3D vector)
- S3: Obstacle coordinates from voxel grid
"""
struct State
    # Basic position and orientation
    position::SVector{3, Float64}
    orientation::SVector{3, Float64}  # Roll, pitch, yaw in radians
    
    # S1: Target information - distance
    target_position::SVector{3, Float64}
    distance_to_target::Float64
    
    # S2: Target information - direction vector
    direction_to_target::SVector{3, Float64}
    angle_to_target::Float64  # Simplified angle in horizontal plane
    
    # S3: Obstacle information as voxel representation
    nearest_obstacle::SVector{3, Float64}
    distance_to_obstacle::Float64
    obstacles::Vector{SVector{3, Float64}}  # Voxel grid centroids
    voxel_grid::Dict{Tuple{Int, Int, Int}, Bool}  # Discretized voxel grid
    
    # Uncertainty estimates
    position_uncertainty::Float64
    obstacle_uncertainty::Float64
end

"""
    extract_state(point_cloud, drone_position, target_position; voxel_size=0.5)

Extract the latent state from sensor data including:
- S1: Distance to target
- S2: Direction vector to target
- S3: Obstacle voxel grid

Returns a State struct.
"""
function extract_state(point_cloud, drone_position::AbstractVector, target_position::AbstractVector; voxel_size=0.5)
    # Default values
    default_pos = SVector{3, Float64}(0.0, 0.0, 0.0)
    default_orientation = SVector{3, Float64}(0.0, 0.0, 0.0)
    
    # Use provided positions or defaults
    position = length(drone_position) == 3 ? SVector{3, Float64}(drone_position) : default_pos
    target = length(target_position) == 3 ? SVector{3, Float64}(target_position) : default_pos
    
    # S1: Calculate distance to target
    distance_to_target = norm(target - position)
    
    # S2: Calculate direction vector and angle to target
    direction = target - position
    direction_to_target = norm(direction) > 0.001 ? normalize(direction) : SVector{3, Float64}(0.0, 0.0, 0.0)
    
    # Calculate angle in the horizontal plane (yaw angle)
    angle_to_target = atan(direction[2], direction[1])  # atan2(y, x)
    
    # S3: Detect obstacles from point cloud and create voxel grid
    obstacles, nearest_obstacle, distance_to_obstacle = detect_obstacles(point_cloud, position)
    voxel_grid = create_voxel_grid(obstacles, voxel_size)
    
    # Simple uncertainty estimates
    position_uncertainty = 0.1  # Fixed value for now
    obstacle_uncertainty = isnothing(point_cloud) ? 1.0 : 
        min(1.0, 10.0 / (isempty(obstacles) ? 1 : length(obstacles)))
    
    return State(
        position,
        default_orientation,
        target,
        distance_to_target,
        direction_to_target,
        angle_to_target,
        nearest_obstacle,
        distance_to_obstacle,
        obstacles,
        voxel_grid,
        position_uncertainty,
        obstacle_uncertainty
    )
end

"""
    create_voxel_grid(points, voxel_size)

Convert a set of 3D points into a discretized voxel grid.
Returns a dictionary mapping voxel coordinates to occupancy.
"""
function create_voxel_grid(points::Vector{SVector{3, Float64}}, voxel_size::Float64)
    grid = Dict{Tuple{Int, Int, Int}, Bool}()
    
    for point in points
        # Convert point to voxel coordinates
        voxel_x = floor(Int, point[1] / voxel_size)
        voxel_y = floor(Int, point[2] / voxel_size)
        voxel_z = floor(Int, point[3] / voxel_size)
        
        # Mark voxel as occupied
        grid[(voxel_x, voxel_y, voxel_z)] = true
    end
    
    return grid
end

"""
    get_obstacle_voxels(state::State; max_voxels=10)

Get the occupied voxels from the state's voxel grid.
Returns a vector of voxel coordinates, limited to max_voxels.
"""
function get_obstacle_voxels(state::State; max_voxels=10)
    voxel_coords = collect(keys(state.voxel_grid))
    
    # Limit number of voxels
    if length(voxel_coords) > max_voxels
        # Sort voxels by distance to drone
        sorted_voxels = sort(voxel_coords, by = voxel -> 
            sum((voxel[i] - state.position[i])^2 for i in 1:3))
        
        voxel_coords = sorted_voxels[1:max_voxels]
    end
    
    return voxel_coords
end

"""
    get_distance_to_target(state::State)

Get the distance to the target from the current state (S1).
"""
function get_distance_to_target(state::State)
    return state.distance_to_target
end

"""
    get_angle_to_target(state::State)

Get the angle to the target from the current state (part of S2).
"""
function get_angle_to_target(state::State)
    return state.angle_to_target
end

"""
    get_direction_to_target(state::State)

Get the direction vector to the target (S2).
"""
function get_direction_to_target(state::State)
    return state.direction_to_target
end

"""
    detect_obstacles(point_cloud, position; radius=10.0)

Detect obstacles from the point cloud within a certain radius around the current position.
Returns a tuple of (obstacles, nearest_obstacle, distance_to_nearest)
"""
function detect_obstacles(point_cloud, position; radius=10.0)
    if isnothing(point_cloud) || size(point_cloud, 2) == 0
        # No point cloud data available
        return SVector{3, Float64}[], 
               SVector{3, Float64}(Inf, Inf, Inf),
               Inf
    end
    
    obstacles = Vector{SVector{3, Float64}}()
    nearest_idx = 0
    min_distance = Inf
    
    for i in 1:size(point_cloud, 2)
        point = SVector{3, Float64}(point_cloud[:, i])
        
        # Calculate distance to drone
        distance = norm(point - position)
        
        # Consider points within radius as obstacles
        # Exclude points too close (likely from the drone itself)
        if 0.5 < distance < radius
            push!(obstacles, point)
            
            # Track nearest obstacle
            if distance < min_distance
                min_distance = distance
                nearest_idx = length(obstacles)
            end
        end
    end
    
    # Get nearest obstacle
    nearest_obstacle = nearest_idx > 0 ? 
                      obstacles[nearest_idx] : 
                      SVector{3, Float64}(Inf, Inf, Inf)
    
    return obstacles, nearest_obstacle, min_distance
end

end # module
