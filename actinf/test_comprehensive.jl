#!/usr/bin/env julia

using StaticArrays

# Comprehensive test script for the hierarchical active inference system
println("="^60)
println("HIERARCHICAL ACTIVE INFERENCE - COMPREHENSIVE TEST")
println("="^60)

# Test 1: Main module loading
println("\n1. Testing main actinf module loading...")
try
    include("src/actinf.jl")
    println("✓ actinf.jl syntax is valid and loads successfully")
    
    # Import the module
    using .actinf
    println("✓ actinf module imported successfully")    # Test that all exports are available
    println("✓ Checking exported functions and types...")
    
    # Test DroneObservation creation first (using proper constructor)
    obs = DroneObservation(
        drone_position=SVector{3, Float64}(0.0, 0.0, 0.0),
        target_position=SVector{3, Float64}(10.0, 5.0, -3.0),
        obstacle_density=0.1
    )
    println("✓ DroneObservation created successfully")
    
    # Test state creation
    state = create_state_from_observation(obs)
    println("✓ DroneState created from observation successfully")
    
    # Test DroneBeliefs creation with state
    beliefs = initialize_beliefs(state)
    println("✓ DroneBeliefs initialized successfully")
      # Test belief updating (need to pass state, not observation)
    update_beliefs!(beliefs, state)
    println("✓ Belief updating completed successfully")
      # Test VFE calculation (convert continuous state to discrete observation indices)
    obs_location_idx = discretize_observation(state.distance, beliefs.location_bins)
    obs_angle_idx = discretize_observation(state.azimuth, beliefs.angle_bins)
    obs_suitability_idx = discretize_observation(state.suitability, beliefs.suitability_bins)
    vfe = calculate_vfe(beliefs, obs_location_idx, obs_angle_idx, obs_suitability_idx)
    println("✓ VFE calculation completed: ", vfe)
    
    # Test serialization
    serialized = serialize_beliefs(beliefs)
    deserialized = deserialize_beliefs(serialized)
    println("✓ Belief serialization/deserialization successful")
    
    println("✓ ALL CORE FUNCTIONALITY TESTS PASSED!")
    
catch e
    println("✗ Error in main module test: ", e)
    println("\nDetailed error information:")
    if isa(e, LoadError) || isa(e, UndefVarError) || isa(e, MethodError)
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
    rethrow()
end

println("\n" * "="^60)
println("TESTING COMPLETED SUCCESSFULLY!")
println("="^60)
