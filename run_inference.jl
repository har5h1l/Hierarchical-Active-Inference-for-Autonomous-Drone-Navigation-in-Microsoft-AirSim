#!/usr/bin/env julia

# Only activate if environment is not already active
if Base.active_project() != abspath(joinpath(@__DIR__, "Project.toml"))
    import Pkg
    Pkg.activate(@__DIR__)
    # Note: No need to develop path here as it should be already set up during precompilation
end

# Start the ZMQ server if not already running
function ensure_zmq_server_running()
    # Check if the server is already running
    using ZMQ
    
    # First, try to connect to the server
    println("Checking if ZMQ server is already running...")
    
    # Platform-specific detection logic
    if Sys.iswindows()
        # Windows: Use wmic to check for Julia processes
        try
            cmd = `wmic process where "name='julia.exe'" get commandline`
            output = read(cmd, String)
            if occursin("zmq_server.jl", output)
                println("⚠️ Julia process found running ZMQ server (wmic)")
                return true
            else
                println("No ZMQ server process found running (wmic)")
            end
        catch e
            println("Error checking for ZMQ server process: $e")
        end
    else
        # Unix-like: Use ps
        try
            cmd = `ps -ef | grep "[z]mq_server.jl"`
            output = read(cmd, String)
            if length(output) > 0
                println("⚠️ Julia process found running ZMQ server (ps)")
                return true
            else
                println("No ZMQ server process found running (ps)")
            end
        catch e
            println("Error checking for ZMQ server process: $e")
        end
    end
    
    # Direct socket test (try to connect and ping)
    try
        ctx = Context()
        socket = Socket(ctx, REQ)
        socket.linger = 0
        try
            socket.rcvtimeo = 1000  # 1 second timeout
            socket.sndtimeo = 1000  # 1 second timeout
        catch
            # Older ZMQ version - use setsockopt
            ZMQ.setsockopt(socket, ZMQ.RCVTIMEO, 1000)
            ZMQ.setsockopt(socket, ZMQ.SNDTIMEO, 1000)
        end
        
        socket.connect("tcp://localhost:5555")
        
        # Try to ping
        try
            socket.send("ping")
            response = String(socket.recv())
            if response == "pong"
                println("✅ Connected to existing ZMQ server (ping successful)")
                socket.close()
                ctx.close()
                return true
            else
                println("⚠️ Connected but unexpected response: $response")
            end
        catch e
            println("Could not ping server: $e")
        end
        
        socket.close()
        ctx.close()
    catch e
        println("Socket connection test failed: $e")
    end
    
    # If we got here, no server was found - start one
    println("No running ZMQ server detected, starting a new one...")
    
    # Build the path to the server script
    zmq_server_path = joinpath(@__DIR__, "actinf", "zmq_server.jl")
    
    # Check if the server script exists
    if !isfile(zmq_server_path)
        println("❌ ZMQ server script not found at: $zmq_server_path")
        # Try an alternative path
        zmq_server_path = joinpath(@__DIR__, "actinf", "src", "zmq_server.jl")
        if !isfile(zmq_server_path)
            error("ZMQ server script not found at any expected location")
        end
    end
    
    # Start the server as a background process
    if Sys.iswindows()
        # Windows: Use start command to launch in background
        run(`cmd /c start /B julia $zmq_server_path`, wait=false)
    else
        # Unix-like: Use nohup
        run(`nohup julia $zmq_server_path &`, wait=false)
    end
    
    # Give the server time to start
    println("ZMQ server starting, waiting 5 seconds for initialization...")
    sleep(5)
    
    # Verify the server started successfully
    try
        ctx = Context()
        socket = Socket(ctx, REQ)
        socket.linger = 0
        try
            socket.rcvtimeo = 5000  # 5 second timeout
            socket.sndtimeo = 5000  # 5 second timeout
        catch
            # Older ZMQ version - use setsockopt
            ZMQ.setsockopt(socket, ZMQ.RCVTIMEO, 5000)
            ZMQ.setsockopt(socket, ZMQ.SNDTIMEO, 5000)
        end
        
        socket.connect("tcp://localhost:5555")
        
        # Try to ping
        try
            socket.send("ping")
            response = String(socket.recv())
            if response == "pong"
                println("✅ ZMQ server started and responding to pings")
                socket.close()
                ctx.close()
                return true
            else
                println("⚠️ ZMQ server responded but with unexpected response: $response")
            end
        catch e
            println("⚠️ Could not ping new server: $e")
        end
        
        socket.close()
        ctx.close()
    catch e
        println("⚠️ Could not connect to new server: $e")
    end
    
    println("⚠️ Could not verify ZMQ server is running, but will continue anyway")
    return false
end

# Import packages - already precompiled by precompile.jl
using JSON
using LinearAlgebra
using StaticArrays
using actinf
using ZMQ

# Constants and parameters with more robust path handling for cross-platform compatibility
const INTERFACE_DIR = joinpath(@__DIR__, "interface")
const OBS_INPUT_PATH = joinpath(INTERFACE_DIR, "obs_input.json")
const INFERRED_STATE_PATH = joinpath(INTERFACE_DIR, "inferred_state.json")

# Make sure interface directory exists and is writable
try
    # Create the directory if it doesn't exist
    if !isdir(INTERFACE_DIR)
        mkpath(INTERFACE_DIR)
        println("Created interface directory: $INTERFACE_DIR")
    end
    
    # Test write access by touching the file - safer approach
    temp_file = joinpath(INTERFACE_DIR, "test_write_access.tmp")
    touch(temp_file)
    if isfile(temp_file)
        rm(temp_file)  # Clean up the test file if successful
    end
catch e
    println("Warning: Issue with interface directory: $e")
    # Don't error out, let's try to continue and create the directory in the write section
end

# Ensure ZMQ server is running before continuing
println("=== Checking ZMQ Server ===")
zmq_running = ensure_zmq_server_running()
println("ZMQ server status: $(zmq_running ? "Running" : "Uncertain")")
println()

function main()
    # Print startup message
    println("\nStarting drone navigation inference...")
    
    # Check if input file exists
    if !isfile(OBS_INPUT_PATH)
        error("Input file $OBS_INPUT_PATH not found.")
    end

    # Read observation data from JSON
    println("\nReading sensor data...")
    obs_data = JSON.parsefile(OBS_INPUT_PATH)

    # Extract and print initial positions
    drone_position = SVector{3, Float64}(get(obs_data, "drone_position", [0.0, 0.0, 0.0])...)
    target_location = SVector{3, Float64}(get(obs_data, "target_location", [10.0, 0.0, -3.0])...)
    
    println("\nInitial Positions (Global Coordinates):")
    println("Drone:  [$(round(drone_position[1], digits=2)), $(round(drone_position[2], digits=2)), $(round(drone_position[3], digits=2))]")
    println("Target: [$(round(target_location[1], digits=2)), $(round(target_location[2], digits=2)), $(round(target_location[3], digits=2))]")
    
    # Extract remaining data
    drone_orientation = SVector{4, Float64}(get(obs_data, "drone_orientation", [1.0, 0.0, 0.0, 0.0])...)
    nearest_obstacle_distances = get(obs_data, "nearest_obstacle_distances", [100.0, 100.0])
    obstacle_density = get(obs_data, "obstacle_density", 0.0)
    
    # Parse voxel grid more efficiently
    voxel_grid = [SVector{3, Float64}(point...) for point in get(obs_data, "voxel_grid", Vector{Vector{Float64}}()) if length(point) == 3]

    # Create DroneObservation object
    observation = DroneObservation(
        drone_position = drone_position,
        drone_orientation = drone_orientation,
        target_position = target_location,
        nearest_obstacle_distances = nearest_obstacle_distances,
        voxel_grid = voxel_grid,
        obstacle_density = obstacle_density
    )

    # Create current state from observation
    current_state = create_state_from_observation(observation)
    
    # Initialize or update beliefs
    println("\nUpdating belief state...")
    beliefs = try
        if isfile(INFERRED_STATE_PATH)
            println("Found existing inferred state file: $INFERRED_STATE_PATH")
            prev_beliefs_json = JSON.parsefile(INFERRED_STATE_PATH)
            prev_beliefs = haskey(prev_beliefs_json, "beliefs") ? 
                          deserialize_beliefs(prev_beliefs_json["beliefs"]) :
                          initialize_beliefs(current_state, voxel_grid=voxel_grid)
            update_beliefs!(prev_beliefs, current_state; voxel_grid=voxel_grid, obstacle_density=obstacle_density)
        else
            println("No existing inferred state file, initializing new beliefs")
            initialize_beliefs(current_state, voxel_grid=voxel_grid, obstacle_density=obstacle_density)
        end
    catch e
        println("Error loading previous beliefs: $e")
        println("Initializing new beliefs")
        # If there's any error reading/parsing previous beliefs, initialize new ones
        initialize_beliefs(current_state, voxel_grid=voxel_grid, obstacle_density=obstacle_density)
    end
    
    # Get expected state from beliefs
    expected_drone_state = expected_state(beliefs)
    
    # Print navigation information
    println("\nNavigation State:")
    println("Current distance to target: $(round(current_state.distance, digits=2)) meters")
    println("Current azimuth to target: $(round(rad2deg(current_state.azimuth), digits=2))°")
    println("Current elevation to target: $(round(rad2deg(current_state.elevation), digits=2))°")
    println("Path suitability: $(round(current_state.suitability, digits=2))")
    println("Obstacle density: $(round(obstacle_density, digits=2))")
    
    # Print target and drone positions in global coordinates
    println("\nGlobal Coordinates:")
    println("Target position: [$(round(target_location[1], digits=2)), $(round(target_location[2], digits=2)), $(round(target_location[3], digits=2))]")
    println("Drone position: [$(round(drone_position[1], digits=2)), $(round(drone_position[2], digits=2)), $(round(drone_position[3], digits=2))]")
    println("Distance to target: $(round(current_state.distance, digits=2))")
    println("Azimuth to target: $(round(rad2deg(current_state.azimuth), digits=2))°")
    println("Elevation to target: $(round(rad2deg(current_state.elevation), digits=2))°")
    println("Path suitability: $(round(current_state.suitability, digits=2))\n")
    
    # Serialize the current state and beliefs to JSON
    output_data = Dict(
        "state" => Dict(
            "distance" => current_state.distance,
            "azimuth" => current_state.azimuth,
            "elevation" => current_state.elevation,
            "suitability" => current_state.suitability
        ),
        "expected_state" => Dict(
            "distance" => expected_drone_state.distance,
            "azimuth" => expected_drone_state.azimuth,
            "elevation" => expected_drone_state.elevation,
            "suitability" => expected_drone_state.suitability
        ),
        "beliefs" => serialize_beliefs(beliefs),
        "drone_position" => [drone_position[1], drone_position[2], drone_position[3]],
        "target_position" => [target_location[1], target_location[2], target_location[3]],
        "nearest_obstacle_distances" => nearest_obstacle_distances,
        "obstacle_density" => obstacle_density,
        "voxel_count" => length(voxel_grid)
    )
    
    # Write to file with robust error handling
    try
        # Make sure the directory exists again right before writing
        if !isdir(INTERFACE_DIR)
            mkpath(INTERFACE_DIR)
            println("Created interface directory before writing: $INTERFACE_DIR")
        end
        
        println("Writing inferred state to: $INFERRED_STATE_PATH")
        open(INFERRED_STATE_PATH, "w") do f
            JSON.print(f, output_data)
        end
        
        # Verify the file was written
        if isfile(INFERRED_STATE_PATH)
            println("Successfully wrote inferred state file")
        else
            println("Warning: File writing seemed to succeed but file doesn't exist")
        end
    catch e
        println("Error during first write attempt: $e")
        try
            # Try a different approach as fallback
            mkpath(dirname(INFERRED_STATE_PATH))
            println("Retrying write with alternative approach...")
            
            # Write directly without using a function
            file = open(INFERRED_STATE_PATH, "w")
            JSON.print(file, output_data)
            close(file)
            
            println("Alternative write approach completed")
        catch e2
            println("Fatal error writing to file: $e2")
            # Try one last approach with a different filename
            fallback_path = joinpath(dirname(@__FILE__), "inferred_state_fallback.json")
            println("Trying last resort to $fallback_path")
            open(fallback_path, "w") do f
                JSON.print(f, output_data)
            end
        end
    end
end

# Run the main function
main()