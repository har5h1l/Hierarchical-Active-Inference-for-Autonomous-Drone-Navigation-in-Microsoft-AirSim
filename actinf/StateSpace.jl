module StateSpace

export DroneState, DroneObservation, create_state_from_observation

using StaticArrays
using LinearAlgebra

"""
    DroneState

Represents the current state of the drone in relation to target and environment.
"""
struct DroneState
    distance::Float64       # Distance to target
    azimuth::Float64        # Horizontal angle to target (radians)
    elevation::Float64      # Vertical angle to target (radians)
    suitability::Float64    # Environmental suitability measure (0-1)
    obstacle_density::Float64 # Density of obstacles around the drone (optional)
end

"""
    DroneState constructor with default values
"""
function DroneState(; 
    distance = 0.0,
    azimuth = 0.0,
    elevation = 0.0, 
    suitability = 1.0,
    obstacle_density = 0.0
)
    return DroneState(distance, azimuth, elevation, suitability, obstacle_density)
end

"""
    DroneObservation

Contains raw sensor data and derived observations.
"""
struct DroneObservation
    drone_position::SVector{3, Float64}
    drone_orientation::SVector{4, Float64}  # Quaternion [w, x, y, z]
    target_position::SVector{3, Float64}
    nearest_obstacle_distances::Vector{Float64}
    voxel_grid::Vector{SVector{3, Float64}}
    obstacle_density::Float64
end

"""
    DroneObservation constructor with default values
"""
function DroneObservation(;
    drone_position = SVector{3, Float64}(0.0, 0.0, 0.0),
    drone_orientation = SVector{4, Float64}(1.0, 0.0, 0.0, 0.0),
    target_position = SVector{3, Float64}(10.0, 0.0, -3.0),
    nearest_obstacle_distances = [100.0, 100.0],
    voxel_grid = Vector{SVector{3, Float64}}(),
    obstacle_density = 0.0
)
    return DroneObservation(
        drone_position,
        drone_orientation,
        target_position,
        nearest_obstacle_distances,
        voxel_grid,
        obstacle_density
    )
end

"""
    create_state_from_observation(observation::DroneObservation)::DroneState

Convert a DroneObservation to a DroneState by calculating relevant state variables.
"""
function create_state_from_observation(observation::DroneObservation)::DroneState
    # Vector from drone to target
    to_target = observation.target_position - observation.drone_position
    
    # Calculate distance to target
    distance_to_target = norm(to_target)
    
    # Calculate azimuth (yaw) and elevation angles
    azimuth = atan(to_target[2], to_target[1])  # atan2(y, x)
    elevation = atan(to_target[3], sqrt(to_target[1]^2 + to_target[2]^2))
    
    # Calculate suitability based on obstacle distance and density
    # Higher when obstacles are far away, lower when close or dense
    obstacle_distance = isempty(observation.nearest_obstacle_distances) ? 100.0 : 
                       minimum(observation.nearest_obstacle_distances)
    safety_factor = exp(-1.0 / max(obstacle_distance, 0.1))
    density_factor = exp(-observation.obstacle_density * 5.0)  # Scale by constant
    suitability = safety_factor * density_factor
    
    return DroneState(
        distance = distance_to_target,
        azimuth = azimuth,
        elevation = elevation,
        suitability = suitability,
        obstacle_density = observation.obstacle_density
    )
end

end # module
