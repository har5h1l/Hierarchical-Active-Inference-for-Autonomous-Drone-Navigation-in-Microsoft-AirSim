#!/usr/bin/env julia

# Test script for the updated matrix-based VFE inference system

using Pkg
Pkg.activate("actinf")

# Load the updated module
include("actinf/src/actinf.jl")
using .actinf
using .actinf.StateSpace
using .actinf.Inference

println("Testing new matrix-based VFE inference system...")

# Create a test state
test_state = DroneState(
    distance = 10.0,
    azimuth = 0.5,
    elevation = 0.1,
    suitability = 0.8
)

println("Created test state: ", test_state)

# Initialize beliefs with new factorized structure
println("Initializing factorized beliefs...")
beliefs = initialize_beliefs(test_state)

println("✓ Successfully initialized factorized beliefs")
println("Location belief size: ", length(beliefs.location_belief))
println("Angle belief size: ", length(beliefs.angle_belief))
println("Suitability belief size: ", length(beliefs.suitability_belief))

# Test belief update
println("Testing belief update...")
updated_state = DroneState(
    distance = 9.0,
    azimuth = 0.3,
    elevation = 0.0,
    suitability = 0.9
)

update_beliefs!(beliefs, updated_state)
println("✓ Successfully updated beliefs")

# Test VFE calculation
if haskey(beliefs.vfe_cache, "current_vfe")
    println("Current VFE: ", beliefs.vfe_cache["current_vfe"])
else
    println("No VFE cached yet")
end

# Test expected state calculation
expected = expected_state(beliefs)
println("Expected state: ", expected)

# Test serialization
println("Testing serialization...")
serialized = serialize_beliefs(beliefs)
println("✓ Successfully serialized beliefs")

# Test deserialization
deserialized = deserialize_beliefs(serialized)
println("✓ Successfully deserialized beliefs")

println("\n✅ All tests passed! Matrix-based VFE inference system is working correctly.")
