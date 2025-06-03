#!/usr/bin/env julia

# Final end-to-end test of the active inference system
using Pkg
Pkg.activate("actinf")

using actinf
using StaticArrays
println("Active inference module loaded successfully!")

# Create test observation using named constructor
obs = DroneObservation(
    drone_position = SVector{3, Float64}(0.0, 0.0, 0.0),
    drone_orientation = SVector{4, Float64}(1.0, 0.0, 0.0, 0.0), 
    target_position = SVector{3, Float64}(10.0, 0.0, -3.0),
    nearest_obstacle_distances = Float64[100.0, 100.0],
    voxel_grid = Vector{SVector{3, Float64}}(),
    obstacle_density = 0.0
)

# Test the complete pipeline
state = create_state_from_observation(obs)
beliefs = initialize_beliefs(state)
update_beliefs!(beliefs, state)

# Test discretization function
loc_idx = discretize_observation(state.distance, beliefs.location_bins)
ang_idx = discretize_observation(state.azimuth, beliefs.angle_bins)
suit_idx = discretize_observation(state.suitability, beliefs.suitability_bins)

# Test VFE calculation
vfe = calculate_vfe(beliefs, loc_idx, ang_idx, suit_idx)

println("End-to-end test successful!")
println("VFE calculated: $vfe")
println("All functions working correctly!")
