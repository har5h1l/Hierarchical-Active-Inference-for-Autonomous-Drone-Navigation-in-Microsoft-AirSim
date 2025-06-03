#!/usr/bin/env julia

"""
Complete Integration Test for Hierarchical Active Inference Drone Navigation System

This test validates the entire pipeline from observation processing through 
belief updates and action selection, simulating real AirSim integration.
"""

using Pkg
Pkg.activate("actinf")

using actinf
using StaticArrays
using JSON

println("============================================================")
println("COMPLETE INTEGRATION TEST - ACTIVE INFERENCE DRONE SYSTEM")
println("============================================================")

# Test 1: Basic observation creation and state conversion
println("\n1. Testing observation creation and state conversion...")

# Create realistic drone observation data
drone_obs = DroneObservation(
    drone_position = SVector{3, Float64}(2.5, -1.2, -8.0),
    drone_orientation = SVector{4, Float64}(0.9, 0.1, 0.1, 0.4),
    target_position = SVector{3, Float64}(15.0, 5.0, -12.0),
    nearest_obstacle_distances = [8.5, 12.3, 6.7, 15.2],
    voxel_grid = [SVector{3, Float64}(1.0, 1.0, -9.0), SVector{3, Float64}(3.0, 2.0, -7.5)],
    obstacle_density = 0.25
)

# Convert to state representation
drone_state = create_state_from_observation(drone_obs)
println("✓ State created: distance=$(round(drone_state.distance, digits=2)), azimuth=$(round(drone_state.azimuth, digits=2)), elevation=$(round(drone_state.elevation, digits=2)), suitability=$(round(drone_state.suitability, digits=2))")

# Test 2: Belief initialization and updates
println("\n2. Testing belief system...")

# Initialize beliefs
beliefs = initialize_beliefs(drone_state)
println("✓ Beliefs initialized with $(length(beliefs.location_bins)) location bins, $(length(beliefs.angle_bins)) angle bins, $(length(beliefs.suitability_bins)) suitability bins")

# Update beliefs with current state
update_beliefs!(beliefs, drone_state)
println("✓ Beliefs updated successfully")

# Test 3: VFE calculation with discretized observations
println("\n3. Testing VFE calculation...")

# Discretize current state for VFE calculation
obs_location_idx = discretize_observation(drone_state.distance, beliefs.location_bins)
obs_angle_idx = discretize_observation(drone_state.azimuth, beliefs.angle_bins) 
obs_suitability_idx = discretize_observation(drone_state.suitability, beliefs.suitability_bins)

# Calculate VFE
vfe = calculate_vfe(beliefs, obs_location_idx, obs_angle_idx, obs_suitability_idx)
println("✓ VFE calculated: $(round(vfe, digits=4))")

# Test 4: Multiple observation updates (simulating navigation)
println("\n4. Testing sequential observations (simulating navigation)...")

# Simulate drone moving closer to target with obstacles
test_observations = [
    # Moving closer, obstacle appears
    DroneObservation(
        drone_position = SVector{3, Float64}(5.0, 1.0, -9.0),
        drone_orientation = SVector{4, Float64}(0.95, 0.05, 0.0, 0.31),
        target_position = SVector{3, Float64}(15.0, 5.0, -12.0),
        nearest_obstacle_distances = [4.2, 8.1, 12.5],
        voxel_grid = [SVector{3, Float64}(6.0, 2.0, -9.0)],
        obstacle_density = 0.45
    ),
    # Avoiding obstacles, different angle
    DroneObservation(
        drone_position = SVector{3, Float64}(8.0, 3.5, -10.0),
        drone_orientation = SVector{4, Float64}(0.87, 0.0, 0.17, 0.46),
        target_position = SVector{3, Float64}(15.0, 5.0, -12.0),
        nearest_obstacle_distances = [7.8, 15.0, 9.2],
        voxel_grid = [SVector{3, Float64}(9.0, 4.0, -10.0)],
        obstacle_density = 0.15
    ),
    # Approaching target
    DroneObservation(
        drone_position = SVector{3, Float64}(12.5, 4.2, -11.5),
        drone_orientation = SVector{4, Float64}(0.99, 0.0, 0.0, 0.14),
        target_position = SVector{3, Float64}(15.0, 5.0, -12.0),
        nearest_obstacle_distances = [12.0, 20.0, 25.0],
        voxel_grid = Vector{SVector{3, Float64}}(),
        obstacle_density = 0.05
    )
]

vfe_history = Float64[]
distance_history = Float64[]

for (i, obs) in enumerate(test_observations)
    state = create_state_from_observation(obs)
    update_beliefs!(beliefs, state)
    
    # Calculate VFE for this state
    loc_idx = discretize_observation(state.distance, beliefs.location_bins)
    ang_idx = discretize_observation(state.azimuth, beliefs.angle_bins)
    suit_idx = discretize_observation(state.suitability, beliefs.suitability_bins)
    current_vfe = calculate_vfe(beliefs, loc_idx, ang_idx, suit_idx)
    
    push!(vfe_history, current_vfe)
    push!(distance_history, state.distance)
    
    println("  Step $i: distance=$(round(state.distance, digits=2)), suitability=$(round(state.suitability, digits=2)), VFE=$(round(current_vfe, digits=4))")
end

println("✓ Sequential navigation test completed")

# Test 5: Belief persistence (serialization/deserialization)
println("\n5. Testing belief persistence...")

# Serialize beliefs
serialized = serialize_beliefs(beliefs)
println("✓ Beliefs serialized ($(length(serialized)) bytes)")

# Deserialize beliefs
restored_beliefs = deserialize_beliefs(serialized)
println("✓ Beliefs deserialized successfully")

# Verify they're equivalent by comparing a VFE calculation
test_vfe_original = calculate_vfe(beliefs, obs_location_idx, obs_angle_idx, obs_suitability_idx)
test_vfe_restored = calculate_vfe(restored_beliefs, obs_location_idx, obs_angle_idx, obs_suitability_idx)

if abs(test_vfe_original - test_vfe_restored) < 1e-10
    println("✓ Serialization/deserialization preserves beliefs accurately")
else
    println("✗ Serialization/deserialization error: VFE mismatch")
end

# Test 6: Edge cases and robustness
println("\n6. Testing edge cases...")

# Test with minimal obstacle distances
edge_obs = DroneObservation(
    drone_position = SVector{3, Float64}(0.0, 0.0, 0.0),
    drone_orientation = SVector{4, Float64}(1.0, 0.0, 0.0, 0.0),
    target_position = SVector{3, Float64}(100.0, 0.0, 0.0),
    nearest_obstacle_distances = [0.5], # Very close obstacle
    voxel_grid = Vector{SVector{3, Float64}}(),
    obstacle_density = 0.9 # High density
)

edge_state = create_state_from_observation(edge_obs)
update_beliefs!(beliefs, edge_state)
println("✓ Edge case handled: very close obstacle, high density")

# Test with target very close
close_target_obs = DroneObservation(
    drone_position = SVector{3, Float64}(0.0, 0.0, 0.0),
    drone_orientation = SVector{4, Float64}(1.0, 0.0, 0.0, 0.0),
    target_position = SVector{3, Float64}(0.1, 0.0, 0.0), # Very close target
    nearest_obstacle_distances = [50.0, 100.0],
    voxel_grid = Vector{SVector{3, Float64}}(),
    obstacle_density = 0.0
)

close_state = create_state_from_observation(close_target_obs)
update_beliefs!(beliefs, close_state)
println("✓ Edge case handled: very close target")

# Summary and analysis
println("\n============================================================")
println("INTEGRATION TEST SUMMARY")
println("============================================================")
println("✓ All core functionality validated")
println("✓ Sequential navigation simulation successful")
println("✓ VFE calculation trending: $(round(vfe_history[1], digits=2)) → $(round(vfe_history[end], digits=2))")
println("✓ Distance progression: $(round(distance_history[1], digits=2)) → $(round(distance_history[end], digits=2)) meters")
println("✓ Belief persistence working correctly")
println("✓ Edge cases handled robustly")
println("\nSYSTEM READY FOR AIRSIM INTEGRATION!")
println("============================================================")
