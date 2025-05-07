#!/usr/bin/env julia

# Simplified precompilation script for Active Inference components
# This ensures all components are properly compiled before navigation begins

println("\n=== Starting Active Inference Precompilation ===")
start_time = time()

# Import packages we need for the precompilation process
import Pkg
try
    using Dates
    # Use simple ASCII output for status messages to avoid encoding issues
    println("✓ Basic packages loaded")
catch e
    println("Adding basic packages...")
    Pkg.add("Dates")
    using Dates
end

# Ensure we can write files with proper encoding
function safe_write(filepath, content)
    try
        open(filepath, "w") do f
            write(f, content)
        end
        return true
    catch e
        println("Error writing to file $filepath: $e")
        return false
    end
end

# Try to add JSON separately to handle potential encoding issues
try
    using JSON
    println("✓ JSON package loaded")
catch e
    println("Adding JSON package...")
    Pkg.add("JSON")
    using JSON
end

# Set up status files to communicate with Python
const PRECOMP_SUCCESS_FLAG = joinpath(@__DIR__, ".precompilation_success")
const PRECOMP_STATUS_FILE = joinpath(@__DIR__, ".precompilation_status.json")

# Function to update status file with proper encoding handling
function update_status(status, message)
    try
        status_data = Dict(
            "status" => status,
            "message" => message,
            "timestamp" => string(Dates.now())
        )
        
        # Write JSON with ASCII-only output
        status_json = JSON.json(status_data)
        safe_write(PRECOMP_STATUS_FILE, status_json)
        println(message)
    catch e
        println("Warning: Could not write status file: $e")
    end
end

# Clean up any previous status files
for file in [PRECOMP_SUCCESS_FLAG, PRECOMP_STATUS_FILE]
    isfile(file) && rm(file, force=true)
end

# Show current project environment information
println("Current directory: $(pwd())")
println("Current project: $(Base.active_project())")

# Activate the project environment
try
    # Activate the project in the current directory
    Pkg.activate(@__DIR__)
    println("✓ Project activated: $(Base.active_project())")
catch e
    update_status("error", "Failed to activate project: $e")
    println("❌ Project activation failed: $e")
    exit(1)
end

# Add required dependencies with better error handling
println("\nInstalling required packages...")
required_packages = ["JSON", "ZMQ", "StaticArrays", "LinearAlgebra"]

for pkg in required_packages
    if pkg != "LinearAlgebra"  # LinearAlgebra is a standard library
        try
            Pkg.add(pkg)
            println("✓ Added $pkg package")
        catch e
            println("⚠️ Error adding $pkg (may already be installed)")
        end
    end
end

# Make sure actinf package is developed properly
try
    println("\nDeveloping actinf package...")
    Pkg.develop(path=joinpath(@__DIR__, "actinf"))
    println("✓ actinf package developed")
catch e
    println("⚠️ Could not develop actinf package: $e")
    update_status("warning", "Could not develop actinf package: $e")
end

# Instantiate the project to ensure all dependencies are satisfied
println("\nInstantiating project...")
Pkg.instantiate()
println("✓ Project instantiated")

# Precompile everything
println("\nPrecompiling packages...")
Pkg.precompile()
println("✓ Packages precompiled")

# Load and verify key packages with better error handling
println("\nVerifying package loading...")
all_packages_loaded = true

for pkg in required_packages
    print("  Testing $pkg... ")
    try
        @eval using $(Symbol(pkg))
        println("✓")
    catch e
        println("❌")
        all_packages_loaded = false
    end
end

if !all_packages_loaded
    update_status("warning", "Some core packages couldn't be loaded")
    println("⚠️ Some packages failed to load but will continue")
end

# Now test loading the actinf package with better error handling
println("\nTesting actinf package loading...")
actinf_loaded = false

try
    # First add actinf explicitly to the environment if it's not already there
    try
        Pkg.develop(path=joinpath(@__DIR__, "actinf"))
    catch
        # Already developed, ignore error
    end
    
    @eval using actinf
    
    # Only try to load submodules if main package loaded successfully
    @eval using actinf.StateSpace
    @eval using actinf.Inference
    @eval using actinf.Planning
    
    println("✓ actinf package and modules loaded successfully")
    actinf_loaded = true
    
    # Create simple test case to ensure all key functions work
    println("\nVerifying core functionality...")
    
    using StaticArrays
    
    # Create sample drone state
    sample_drone_position = SVector{3, Float64}(0.0, 0.0, 0.0)
    sample_target_position = SVector{3, Float64}(10.0, 0.0, -3.0)
    sample_orientation = SVector{4, Float64}(1.0, 0.0, 0.0, 0.0)
    
    # Test StateSpace
    observation = actinf.StateSpace.DroneObservation(
        drone_position = sample_drone_position,
        drone_orientation = sample_orientation,
        target_position = sample_target_position,
        nearest_obstacle_distances = Float64[],
        voxel_grid = Vector{SVector{3, Float64}}(),
        obstacle_density = 0.0
    )
    state = actinf.StateSpace.create_state_from_observation(observation)
    println("  StateSpace module test: ✓")
    
    # Test Inference module
    beliefs = actinf.Inference.initialize_beliefs(state)
    actinf.Inference.update_beliefs!(beliefs, state)
    println("  Inference module test: ✓")
    
    # Test Planning module
    planner = actinf.Planning.ActionPlanner(
        preference_model = actinf.Planning.PreferenceModel()
    )
    println("  Planning module test: ✓")
    
    # Test ZMQ functionality without starting server
    println("\nTesting ZMQ functionality...")
    zmq_context = ZMQ.Context()
    zmq_socket = ZMQ.Socket(zmq_context, ZMQ.REP)
    ZMQ.close(zmq_socket)
    ZMQ.close(zmq_context)
    println("  ZMQ functionality: ✓")
    
    # Mark as successful
    touch(PRECOMP_SUCCESS_FLAG)
    update_status("success", "✓ All components successfully precompiled")
catch e
    println("❌ Failed to load or test actinf package: $e")
    update_status("error", "Failed to load actinf package: $e")
    exit(1)
end

# Report completion status
elapsed = round(time() - start_time, digits=1)
println("\n=== Active Inference Precompilation Completed in $elapsed seconds ===")
println("The system is now ready for efficient navigation")

# Always exit cleanly with success to avoid alarming Python
exit(0)