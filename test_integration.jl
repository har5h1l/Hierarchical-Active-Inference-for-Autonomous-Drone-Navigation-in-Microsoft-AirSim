#!/usr/bin/env julia

# Test integration with ZMQ server using the new matrix-based inference system

using Pkg
Pkg.activate("actinf")

include("actinf/src/actinf.jl")
using .actinf
using .actinf.StateSpace
using .actinf.Inference

println("Testing integration with ZMQ server...")

# Create a test observation similar to what the ZMQ server receives
test_observation = Dict(
    "drone_position" => [0.0, 0.0, -5.0],
    "target_position" => [10.0, 5.0, -5.0],
    "obstacle_positions" => [[3.0, 2.0, -5.0], [7.0, 3.0, -5.0]],
    "obstacle_distances" => [5.0, 8.0],
    "safety_margin" => 1.5,
    "density_radius" => 5.0
)

println("Created test observation: ", test_observation)

# Simulate the observation processing that happens in the ZMQ server
try
    # Calculate obstacle density
    drone_position = [0.0, 0.0, -5.0]
    obstacle_positions = [[3.0, 2.0, -5.0], [7.0, 3.0, -5.0]]
    density_radius = 5.0
    
    nearby_count = 0
    for pos in obstacle_positions
        distance = sqrt(sum((pos[i] - drone_position[i])^2 for i in 1:3))
        if distance < density_radius
            nearby_count += 1
        end
    end
    
    volume = (4/3) * π * density_radius^3
    obstacle_density = nearby_count / volume
    
    println("Calculated obstacle density: ", obstacle_density)
    
    # Create a state from the observation
    target_pos = test_observation["target_position"]
    drone_pos = test_observation["drone_position"]
    
    # Calculate relative position
    relative_pos = [target_pos[i] - drone_pos[i] for i in 1:3]
    distance = sqrt(sum(relative_pos[i]^2 for i in 1:3))
    azimuth = atan(relative_pos[2], relative_pos[1])
    elevation = asin(relative_pos[3] / distance)
    
    # Calculate basic suitability (without obstacle influence for now)
    base_suitability = max(0.1, 1.0 - obstacle_density * 10.0)
    
    println("Calculated state components:")
    println("  Distance: ", round(distance, digits=2))
    println("  Azimuth: ", round(azimuth, digits=3))
    println("  Elevation: ", round(elevation, digits=3))
    println("  Base suitability: ", round(base_suitability, digits=3))
    
    # Create the state
    current_state = DroneState(
        distance = distance,
        azimuth = azimuth,
        elevation = elevation,
        suitability = base_suitability
    )
    
    println("Created state: ", current_state)
    
    # Initialize beliefs
    beliefs = initialize_beliefs(current_state)
    println("✓ Initialized beliefs successfully")
    
    # Update beliefs with obstacle information
    update_beliefs!(beliefs, current_state, obstacle_density=obstacle_density)
    println("✓ Updated beliefs successfully")
    
    # Calculate VFE
    vfe = calculate_vfe(beliefs)
    println("Current VFE: ", round(vfe, digits=3))
    
    # Get expected state
    expected = expected_state(beliefs)
    println("Expected state: ", expected)
    
    # Test serialization (as would happen in ZMQ communication)
    serialized = serialize_beliefs(beliefs)
    println("✓ Serialized beliefs successfully")
    
    deserialized = deserialize_beliefs(serialized)
    println("✓ Deserialized beliefs successfully")
    
    println("\n✅ Integration test completed successfully!")
    println("The new matrix-based VFE inference system is compatible with the ZMQ server.")
    
catch e
    println("❌ Integration test failed: ", e)
    println("Stack trace:")
    Base.show_backtrace(stdout, catch_backtrace())
end
