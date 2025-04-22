module State

export State, extract_state, get_distance_to_target, get_angle_to_target, detect_obstacles

using LinearAlgebra
using StaticArrays

"""
State structure representing the drone's state relative to environment
"""
struct State
    # Position information
    position::SVector{3, Float64}
    orientation::SVector{3, Float64}  # Roll, pitch, yaw in radians
    
    # Target information
    target_position::SVector{3, Float64}
    distance_to_target::Float64
    angle_to_target::Float64
    
    # Obstacle information
    nearest_obstacle::SVector{3, Float64}
    distance_to_obstacle::Float64
    obstacles::Vector{SVector{3, Float64}}
    
    # Uncertainty estimates
    position_uncertainty::Float64
    obstacle_uncertainty::Float64
end

"""
    extract_state(point_cloud, drone_position, target_position)

Extract the latent state from sensor data including:
- Distance and angle to target
- Nearest obstacles and their coordinates
- Uncertainty estimates

Returns a State struct.
"""
function extract_state(point_cloud, drone_position::AbstractVector, target_position::AbstractVector)
    # Default values
    default_pos = SVector{3, Float64}(0.0, 0.0, 0.0)
    default_orientation = SVector{3, Float64}(0.0, 0.0, 0.0)
    
    # Use provided positions or defaults
    position = length(drone_position) == 3 ? SVector{3, Float64}(drone_position) : default_pos
    target = length(target_position) == 3 ? SVector{3, Float64}(target_position) : default_pos
    
    # Calculate distance and angle to target
    distance_to_target = norm(target - position)
    
    # Calculate angle in the horizontal plane (yaw angle)
    direction = target - position
    angle_to_target = atan(direction[2], direction[1])  # atan2(y, x)
    
    # Detect obstacles from point cloud
    obstacles, nearest_obstacle, distance_to_obstacle = detect_obstacles(point_cloud, position)
    
    # Simple uncertainty estimates (could be more sophisticated)
    position_uncertainty = 0.1  # Fixed value for now
    obstacle_uncertainty = isnothing(point_cloud) ? 1.0 : 
        min(1.0, 10.0 / (isempty(obstacles) ? 1 : length(obstacles)))
    
    return State(
        position,
        default_orientation,
        target,
        distance_to_target,
        angle_to_target,
        nearest_obstacle,
        distance_to_obstacle,
        obstacles,
        position_uncertainty,
        obstacle_uncertainty
    )
end

"""
    get_distance_to_target(state::State)

Get the distance to the target from the current state.
"""
function get_distance_to_target(state::State)
    return state.distance_to_target
end

"""
    get_angle_to_target(state::State)

Get the angle to the target from the current state.
"""
function get_angle_to_target(state::State)
    return state.angle_to_target
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
