module StateSpace

using StaticArrays
using LinearAlgebra

export DroneState, DroneObservation, create_state_from_observation, global_to_egocentric, egocentric_to_global, calculate_suitability

"""
    DroneState

Represents the current state of the drone in relation to target and environment.
"""
struct DroneState
    distance::Float64       # Distance to target
    azimuth::Float64        # Horizontal angle to target (radians)
    elevation::Float64      # Vertical angle to target (radians)
    suitability::Float64    # Environmental suitability measure (0-1)
end

"""
    DroneState constructor with default values
"""
function DroneState(; 
    distance = 0.0,
    azimuth = 0.0,
    elevation = 0.0, 
    suitability = 1.0
)
    return DroneState(distance, azimuth, elevation, suitability)
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
    quaternion_to_matrix(q::SVector{4, Float64})::Matrix{Float64}

Convert a quaternion to a rotation matrix.
AirSim uses the NED coordinate system and right-handed rotation.
"""
function quaternion_to_matrix(q::SVector{4, Float64})::Matrix{Float64}
    # Normalize quaternion first to ensure proper rotation
    qnorm = norm(q)
    if qnorm ≈ 0
        return Matrix{Float64}(I, 3, 3)
    end
    q = q / qnorm
    
    w, x, y, z = q
    
    # Corrected rotation matrix for NED coordinate system
    return [
        w^2 + x^2 - y^2 - z^2    2(x*y - w*z)           2(x*z + w*y);
        2(x*y + w*z)             w^2 - x^2 + y^2 - z^2   2(y*z - w*x);
        2(x*z - w*y)             2(y*z + w*x)            w^2 - x^2 - y^2 + z^2
    ]
end

"""
    global_to_egocentric(global_point::SVector{3, Float64}, drone_pos::SVector{3, Float64}, drone_orientation::SVector{4, Float64})::SVector{3, Float64}

Convert a point from global coordinates to drone's egocentric frame.
"""
function global_to_egocentric(global_point::SVector{3, Float64}, drone_pos::SVector{3, Float64}, drone_orientation::SVector{4, Float64})::SVector{3, Float64}
    # Translate to drone's position
    translated = global_point - drone_pos
    
    # Get rotation matrix from quaternion (inverse rotation)
    R = quaternion_to_matrix(drone_orientation)'
    
    # Apply rotation to get point in drone's frame
    return SVector{3, Float64}(R * translated)
end

"""
    egocentric_to_global(ego_point::SVector{3, Float64}, drone_pos::SVector{3, Float64}, drone_orientation::SVector{4, Float64})::SVector{3, Float64}

Convert a point from drone's egocentric frame to global coordinates.
"""
function egocentric_to_global(ego_point::SVector{3, Float64}, drone_pos::SVector{3, Float64}, drone_orientation::SVector{4, Float64})::SVector{3, Float64}
    # Get rotation matrix from quaternion
    R = quaternion_to_matrix(drone_orientation)
    
    # Apply rotation and translation
    return SVector{3, Float64}(R * ego_point + drone_pos)
end

"""
    calculate_suitability(obstacle_distance::Float64, obstacle_density::Float64; 
                         obstacle_weight::Float64=0.7, density_weight::Float64=0.3,
                         cutoff_distance::Float64=2.5, steepness_distance::Float64=3.0,
                         cutoff_density::Float64=0.2, steepness_density::Float64=10.0)::Float64

Calculate environmental suitability based on obstacle distance and density.
Higher values (closer to 1.0) indicate safer navigation conditions.

Parameters:
- obstacle_distance: Distance to nearest obstacle (meters)
- obstacle_density: Density of obstacles in local region (0-1)
- obstacle_weight: Weight for obstacle distance factor (default: 0.7)
- density_weight: Weight for density factor (default: 0.3)
- cutoff_distance: Distance below which suitability rapidly decreases (default: 2.5)
- steepness_distance: Controls how quickly suitability transitions (default: 3.0)
- cutoff_density: Density above which suitability rapidly decreases (default: 0.2)
- steepness_density: Controls how quickly density suitability transitions (default: 10.0)

Returns:
- Suitability score (0-1)
"""
function calculate_suitability(obstacle_distance::Float64, obstacle_density::Float64; 
                             obstacle_weight::Float64=0.7, density_weight::Float64=0.3,
                             cutoff_distance::Float64=2.5, steepness_distance::Float64=3.0,
                             cutoff_density::Float64=0.2, steepness_density::Float64=10.0)::Float64
    # Safety factor using sigmoid-like function for more predictable transition
    # 1 / (1 + e^(-steepness * (distance - cutoff)))
    # When distance equals cutoff, value is 0.5
    # As distance increases, approaches 1.0
    # As distance decreases, approaches 0.0
    safety_factor = 1.0 / (1.0 + exp(-steepness_distance * (obstacle_distance - cutoff_distance)))
    
    # Density factor similarly uses sigmoid-like function, but inverted
    # 1 / (1 + e^(steepness * (density - cutoff)))
    # When density equals cutoff, value is 0.5
    # As density decreases, approaches 1.0
    # As density increases, approaches 0.0
    density_factor = 1.0 / (1.0 + exp(steepness_density * (obstacle_density - cutoff_density)))
    
    # Combine factors with relative weights - both should be high for good suitability
    # Ensure weights sum to 1.0 for proper scaling
    weight_sum = obstacle_weight + density_weight
    normalized_obstacle_weight = obstacle_weight / weight_sum
    normalized_density_weight = density_weight / weight_sum
    
    suitability = normalized_obstacle_weight * safety_factor + normalized_density_weight * density_factor
    
    return clamp(suitability, 0.0, 1.0)  # Ensure result is between 0 and 1
end

"""
    create_state_from_observation(observation::DroneObservation)::DroneState

Convert a DroneObservation to a DroneState by calculating relevant state variables.
Uses egocentric coordinates for state representation.
"""
function create_state_from_observation(observation::DroneObservation)::DroneState
    # Convert target position to drone's egocentric frame
    target_ego = global_to_egocentric(
        observation.target_position,
        observation.drone_position,
        observation.drone_orientation
    )
    
    # Calculate distance to target (remains the same in either frame)
    distance_to_target = norm(target_ego)
    
    # Calculate azimuth (yaw) in drone's frame
    # Using atan2 to get proper quadrant, considering NED coordinate system
    # Positive azimuth means target is to the right
    azimuth = atan(target_ego[2], target_ego[1])  # y/x for direction in horizontal plane
    
    # Calculate elevation in NED frame
    # Negative sign because NED has positive Z pointing down
    elevation = -atan(target_ego[3], sqrt(target_ego[1]^2 + target_ego[2]^2))
    
    # Get obstacle distance from observation
    obstacle_distance = isempty(observation.nearest_obstacle_distances) ? 100.0 : 
                       minimum(observation.nearest_obstacle_distances)
    
    # Calculate suitability using the standardized function
    # Giving higher weight to obstacle distance than density
    suitability = calculate_suitability(obstacle_distance, observation.obstacle_density)
    
    return DroneState(
        distance = distance_to_target,
        azimuth = azimuth,
        elevation = elevation,
        suitability = suitability
    )
end

end # module StateSpace
