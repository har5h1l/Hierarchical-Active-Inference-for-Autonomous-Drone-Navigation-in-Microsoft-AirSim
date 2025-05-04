#!/usr/bin/env julia

# Precompilation script for Active Inference components
# This script ensures all components are precompiled before any navigation operations start

println("\n=== Starting Active Inference Precompilation ===")
start_time = time()

# Set environment variable to indicate precompilation has been done
ENV["JULIA_ACTINF_PRECOMPILED"] = "true"

# Create a status file to indicate precompilation status for Python to check
const PRECOMPILE_SUCCESS_FLAG = joinpath(@__DIR__, ".precompilation_success")
const PRECOMPILE_STATUS_FILE = joinpath(@__DIR__, ".precompilation_status.json")

# Function to update status file
function update_status(status, message)
    try
        open(PRECOMPILE_STATUS_FILE, "w") do f
            JSON.print(f, Dict(
                "status" => status,
                "message" => message,
                "timestamp" => string(Dates.now())
            ))
        end
        println(message)
    catch e
        println("Warning: Could not write status file: $e")
    end
end

# Cleanup any previous status files at start
for file in [PRECOMPILE_SUCCESS_FLAG, PRECOMPILE_STATUS_FILE]
    if isfile(file)
        try
            rm(file)
        catch
            println("Warning: Could not remove previous status file: $file")
        end
    end
end

try
    # Activate the project - essential for package environment
    import Pkg
    import Dates
    import JSON
    
    # Only activate if not already active
    if Base.active_project() != abspath(joinpath(@__DIR__, "Project.toml"))
        Pkg.activate(@__DIR__)
        println("✅ Project activated")
    else
        println("✅ Project already active")
    end
    
    # Develop the actinf package in place
    try
        Pkg.develop(path=joinpath(@__DIR__, "actinf"))
        println("✅ actinf package developed")
    catch e
        println("⚠️ Could not develop actinf package: $e")
        println("Will continue with existing package")
    end
    
    # Instantiate to ensure all packages are installed
    println("Ensuring all dependencies are installed...")
    Pkg.instantiate()
    println("✅ Package dependencies instantiated")
    
    # Status for verification
    println("\nVerifying package status:")
    Pkg.status()
catch e
    update_status("error", "❌ Error activating or setting up project: $e")
    exit(1)
end

# Import packages to precompile them
println("\nPrecompiling core packages...")
using_packages = ["JSON", "LinearAlgebra", "StaticArrays", "ZMQ"]

function precompile_package(pkg_name)
    print("   $pkg_name... ")
    try
        @eval using $(Symbol(pkg_name))
        println("✅")
        return true
    catch e
        println("❌ ($e)")
        return false
    end
end

all_packages_loaded = true
for pkg in using_packages
    all_packages_loaded &= precompile_package(pkg)
end

# Only continue if core packages loaded successfully
if !all_packages_loaded
    update_status("error", "❌ Failed to load some core packages. Precompilation cannot complete.")
    exit(1)
end

# Precompile actinf package
println("\nPrecompiling actinf package...")
try
    @eval using actinf
    println("✅ actinf package loaded")
    
    # Precompile key functions by importing and executing
    @eval using actinf.StateSpace
    @eval using actinf.Inference
    @eval using actinf.Planning
    
    println("✅ Successfully imported actinf modules")
    
    # Create sample data to exercise functions
    println("\nExecuting core functions to ensure precompilation...")
    
    # Sample data for testing
    using StaticArrays
    
    # Create sample drone state
    sample_drone_position = SVector{3, Float64}(0.0, 0.0, 0.0)
    sample_target_position = SVector{3, Float64}(10.0, 0.0, -3.0)
    sample_orientation = SVector{4, Float64}(1.0, 0.0, 0.0, 0.0)
    obstacle_distances = [100.0, 100.0]
    
    # Exercise the StateSpace module
    println("   Testing StateSpace module... ")
    observation = actinf.StateSpace.DroneObservation(
        drone_position = sample_drone_position,
        drone_orientation = sample_orientation,
        target_position = sample_target_position,
        nearest_obstacle_distances = obstacle_distances,
        voxel_grid = Vector{SVector{3, Float64}}(),
        obstacle_density = 0.0
    )
    state = actinf.StateSpace.create_state_from_observation(observation)
    println("✅ StateSpace functions precompiled")
    
    # Exercise the Inference module
    println("   Testing Inference module... ")
    beliefs = actinf.Inference.initialize_beliefs(state)
    actinf.Inference.update_beliefs!(beliefs, state)
    expected = actinf.Inference.expected_state(beliefs)
    println("✅ Inference functions precompiled")
    
    # Exercise the Planning module
    println("   Testing Planning module... ")
    planner = actinf.Planning.ActionPlanner(
        actinf.Planning.PreferenceModel(
            distance_preference = 0.0,
            path_preference = 0.9
        )
    )
    action = actinf.Planning.select_action(planner, beliefs)
    println("✅ Planning functions precompiled")
    
    # Test ZMQ socket creation but don't actually start server
    println("   Testing ZMQ functionality...")
    zmq_context = ZMQ.Context()
    zmq_socket = ZMQ.Socket(zmq_context, ZMQ.REP)
    ZMQ.close(zmq_socket)
    ZMQ.close(zmq_context)
    println("✅ ZMQ functionality verified")
    
    # Create the success flag file for Python to detect
    touch(PRECOMPILE_SUCCESS_FLAG)
    
    # Update status file with success
    update_status("success", "✅ Core functions precompiled successfully")
    
    # Don't precompile ZMQ server as that would start the server
    println("\n✅ Core functions precompiled successfully")
catch e
    println("\n❌ Error precompiling actinf package: $e")
    
    # Update status file with error
    update_status("error", "❌ Error precompiling actinf package: $e")
    
    # Try to include files directly if package loading fails
    println("\nAttempting to directly include actinf module files...")
    try
        include(joinpath(@__DIR__, "actinf", "src", "StateSpace.jl"))
        include(joinpath(@__DIR__, "actinf", "src", "Inference.jl"))
        include(joinpath(@__DIR__, "actinf", "src", "Planning.jl"))
        println("✅ Successfully included actinf module files")
        
        # Create partial success flag
        update_status("partial", "⚠️ Partial precompilation completed via direct inclusion")
    catch e2
        println("❌ Failed to include actinf module files: $e2")
        println("Precompilation incomplete. Some functions may be slower on first use.")
        update_status("failed", "❌ Failed to include actinf module files: $e2")
        exit(1)
    end
end

# Report completion
elapsed = round(time() - start_time, digits=1)
println("\n=== Active Inference Precompilation Completed in $elapsed seconds ===")
println("The system should now run with minimal compilation delays.")

# Final success status
update_status("complete", "✅ Precompilation completed in $elapsed seconds")
exit(0)