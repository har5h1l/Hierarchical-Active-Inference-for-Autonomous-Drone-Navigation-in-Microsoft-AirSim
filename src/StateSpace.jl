module StateSpace

export DroneState, DroneObservation, create_state_from_observation, discretize_state, calculate_suitability

using LinearAlgebra
using StaticArrays

"""
DroneState represents the discretized state space of the drone.

State space components:
1. distance_to_target: Discretized by 1m increments
2. azimuth: Discretized by 15° increments (horizontal angle to target)
3. elevation: Discretized by 15° increments (vertical angle to target)
4. suitability: Range 0-1, discretized by 0.1 increments (safety/obstacle measure)
"""
struct DroneState
    # Discretized state variables
    distance_to_target::Int  # Discretized by 1m
    azimuth::Int             # Discretized by 15° (24 possible values)
    elevation::Int           # Discretized by 15° (12 possible values)
    suitability::Int         # Discretized by 0.1 (11 possible values)
end

"""
DroneObservation represents the raw sensor observations.

Observation space:
1. obs_distance: Raw distance from target (meters)
2. obs_azimuth: Raw azimuth angle to target (radians)
3. obs_elevation: Raw elevation angle to target (radians)
4. obs_nearest_obstacle: Distance to nearest obstacle (meters)
"""
struct DroneObservation
    obs_distance::Float64
    obs_azimuth::Float64
    obs_elevation::Float64
    obs_nearest_obstacle::Float64
end

"""
    discretize_state(distance::Float64, azimuth::Float64, 
                    elevation::Float64, suitability::Float64)

Discretize continuous values according to our state space definition.
Returns tuple of discretized (integer) values.
"""
function discretize_state(distance::Float64, azimuth::Float64, 
                         elevation::Float64, suitability::Float64)
    # Discretize distance by 1m increments
    distance_idx = max(1, Int(floor(distance)))
    
    # Discretize azimuth by 15° increments (24 possible values)
    # Convert from radians to 0-23 index (15° = π/12 radians)
    azimuth_normalized = mod(azimuth, 2π)  # Normalize to [0, 2π)
    azimuth_idx = Int(floor(azimuth_normalized / (π/12))) + 1
    
    # Discretize elevation by 15° increments (12 possible values)
    # Convert from radians to 0-11 index
    elevation_normalized = clamp(elevation, -π/2, π/2)  # Clamp to [-π/2, π/2]
    elevation_idx = Int(floor((elevation_normalized + π/2) / (π/12))) + 1
    
    # Discretize suitability by 0.1 increments (0 to 1.0)
    suitability_idx = Int(floor(suitability * 10)) + 1
    
    return distance_idx, azimuth_idx, elevation_idx, suitability_idx
end

"""
    calculate_suitability(obstacle_distance::Float64; beta::Float64=3.0)

Calculate suitability measure (0-1) based on distance to nearest obstacle.
Uses exponential transformation with parameter beta as specified.
"""
function calculate_suitability(obstacle_distance::Float64; beta::Float64=3.0)
    # Higher penalty as you get closer to obstacle
    return min(1.0, exp(-beta / (obstacle_distance + 1e-3)))
end

"""
    create_state_from_observation(observation::DroneObservation)

Create a DroneState from raw observations.
"""
function create_state_from_observation(observation::DroneObservation)
    # Calculate suitability from nearest obstacle distance
    suitability = calculate_suitability(observation.obs_nearest_obstacle)
    
    # Discretize the state variables
    disc_distance, disc_azimuth, disc_elevation, disc_suitability = 
        discretize_state(
            observation.obs_distance, 
            observation.obs_azimuth,
            observation.obs_elevation,
            suitability
        )
    
    return DroneState(
        disc_distance,
        disc_azimuth,
        disc_elevation,
        disc_suitability
    )
end

"""
    get_nearest_obstacle(position::AbstractVector, point_cloud::Vector)

Find the nearest obstacle point and its distance.
"""
function get_nearest_obstacle(position::AbstractVector, point_cloud::Vector)
    if isempty(point_cloud)
        return SVector{3, Float64}(Inf, Inf, Inf), Inf
    end
    
    min_distance = Inf
    nearest_point = SVector{3, Float64}(Inf, Inf, Inf)
    pos = SVector{3, Float64}(position)
    
    for point in point_cloud
        dist = norm(pos - point)
        if dist > 0.5 && dist < min_distance  # Avoid points too close (drone itself)
            min_distance = dist
            nearest_point = point
        end
    end
    
    return nearest_point, min_distance
end

end # module
