#!/usr/bin/env julia

println("\n=== Starting Active Inference Package Rebuild ===")

# Capture start time for performance reporting
start_time = time()

function run_step(step_name, func)
    println("\n▶ $step_name...")
    step_start = time()
    try
        func()
        step_duration = round(time() - step_start, digits=2)
        println("✓ $step_name completed in $step_duration seconds")
        return true
    catch e
        println("✕ $step_name failed: $e")
        return false
    end
end

# Activate main project with error handling
if !run_step("Activating main project", () -> begin
    import Pkg
    Pkg.activate(@__DIR__)
end)
    println("Cannot continue without activating project")
    exit(1)
end

# Remove existing actinf development with error recovery
run_step("Removing existing actinf development", () -> begin
    try
        Pkg.rm("actinf")
    catch e
        println("Note: Could not remove actinf package - it may not be currently added: $e")
        # This is not a fatal error - continue
    end
end)

# Add actinf package in development mode with retries
actinf_path = joinpath(@__DIR__, "actinf")
if !run_step("Adding actinf package in development mode", () -> begin
    max_retries = 3
    for i in 1:max_retries
        try
            Pkg.develop(path=actinf_path)
            break
        catch e
            if i == max_retries
                println("Failed after $max_retries attempts")
                rethrow(e)
            else
                println("Attempt $i failed, retrying...")
                sleep(1)  # Short delay before retry
            end
        end
    end
end)
    println("Cannot continue without actinf package")
    exit(1)
end

# Install dependencies with progress feedback
if !run_step("Installing dependencies", () -> begin
    Pkg.instantiate()
end)
    println("Warning: Some dependencies may not be installed correctly")
end

# Verify installation by importing actinf
if !run_step("Verifying installation", () -> begin
    # Use Base.require instead of 'using' for better error messages
    Base.require(Main, :actinf)
    
    # Verify submodules are also accessible
    Base.require(Main, Symbol("actinf.StateSpace"))
    Base.require(Main, Symbol("actinf.Inference"))
    Base.require(Main, Symbol("actinf.Planning"))
    
    println("  Successfully loaded all actinf modules")
end)
    println("\nWARNING: Package verification failed!")
    println("Some components may not work correctly.")
    println("You may need to restart Julia before continuing.")
else
    # Run precompilation if verification succeeds
    if !run_step("Running precompilation", () -> begin
        include(joinpath(@__DIR__, "precompile.jl"))
    end)
        println("\nWARNING: Precompilation failed!")
        println("This may cause delays during the first execution of functions.")
    end
end

# Report total duration
total_duration = round(time() - start_time, digits=2)
println("\n=== Active Inference Package Rebuild Complete ===")
println("Total time: $total_duration seconds")
println("The system is ready for navigation operations")