#!/usr/bin/env julia

println("Starting drone navigation planning script...")

# Only activate if environment is not already active
if Base.active_project() != abspath(joinpath(@__DIR__, "Project.toml"))
    println("Activating project environment...")
    import Pkg
    Pkg.activate(@__DIR__)
    println("Project activated")
else
    println("Project environment already active")
end

# Import StaticArrays early in the script to ensure SVector is available
import StaticArrays
import StaticArrays: SVector

# Load required packages with explicit error handling
println("Loading required packages...")
function load_package(pkgname)
    try
        println("Loading $pkgname...")
        @eval import $pkgname
        println("Successfully loaded $pkgname")
        return true
    catch e
        println("Error loading $pkgname: $e")
        return false
    end
end

# Load standard libraries first
load_package(:JSON)
load_package(:LinearAlgebra)
load_package(:StaticArrays)
load_package(:ZMQ)

# Make StaticArrays available in main scope
using StaticArrays  # This explicitly brings SVector into scope
using StaticArrays: SVector  # Explicitly import SVector type

# For StaticArrays version compatibility, alias SVector if not defined
if !isdefined(Main, :SVector)
    const SVector = StaticArrays.SVector
    println("Defined SVector alias for compatibility")
end

# ZMQ.jl API compatibility layer
# This ensures the script works with both newer and older versions of ZMQ.jl
module ZMQCompat
    # Export all the functions we'll use as compatibility wrappers
    export setup_zmq_api, connect, send, recv, close, set_linger, set_rcvhwm, set_sndhwm, set_tcp_keepalive
    
    # Global variables to store which API version we're using
    using_new_api = false
    zmq_module = nothing
    
    # Setup function to determine API version and configure wrappers
    function setup_zmq_api(zmq_mod)
        global zmq_module = zmq_mod
        global using_new_api = false
        
        # Test if we're using the newer API by trying to call set_linger directly
        try
            # Create temporary socket to test API
            ctx = zmq_mod.Context()
            sock = zmq_mod.Socket(ctx, zmq_mod.REQ)
            
            # Try to use newer direct method
            zmq_mod.set_linger(sock, 0)
            global using_new_api = true
            println("✓ Using newer ZMQ.jl API with direct socket methods")
            
            # Clean up test socket
            zmq_mod.close(sock)
            zmq_mod.close(ctx)
        catch
            # Fall back to older API
            println("✓ Using older ZMQ.jl API with ZMQ.setsockopt")
        end
        
        return using_new_api
    end
    
    # Compatibility wrappers for ZMQ functions
    function connect(socket, address)
        if using_new_api
            return zmq_module.connect(socket, address)
        else
            return zmq_module.connect(socket, address)  # Same in both APIs
        end
    end
    
    function send(socket, message)
        if using_new_api
            return zmq_module.send(socket, message)
        else
            return zmq_module.send(socket, message)  # Same in both APIs
        end
    end
    
    function recv(socket)
        if using_new_api
            return zmq_module.recv(socket)
        else
            return zmq_module.recv(socket)  # Same in both APIs
        end
    end
    
    function close(obj)
        if using_new_api
            return zmq_module.close(obj)
        else
            # Determine if obj is a socket or context
            if typeof(obj) <: zmq_module.Socket
                return zmq_module.close(obj)  # Same in both APIs
            else
                # For context, use term in older API
                return zmq_module.term(obj)
            end
        end
    end
    
    function set_linger(socket, value)
        if using_new_api
            return zmq_module.set_linger(socket, value)
        else
            return zmq_module.setsockopt(socket, zmq_module.LINGER, value)
        end
    end
    
    function set_rcvhwm(socket, value)
        if using_new_api
            return zmq_module.set_rcvhwm(socket, value)
        else
            return zmq_module.setsockopt(socket, zmq_module.RCVHWM, value)
        end
    end
    
    function set_sndhwm(socket, value)
        if using_new_api
            return zmq_module.set_sndhwm(socket, value)
        else
            return zmq_module.setsockopt(socket, zmq_module.SNDHWM, value)
        end
    end
    
    function set_tcp_keepalive(socket, value)
        try
            if using_new_api
                return zmq_module.set_tcp_keepalive(socket, value)
            else
                return zmq_module.setsockopt(socket, zmq_module.TCP_KEEPALIVE, value)
            end
        catch
            println("TCP_KEEPALIVE not supported in this ZMQ version")
        end
    end
end

# Load actinf with special handling - using include if necessary
println("Loading actinf package...")
try
    # Try direct loading first
    @eval using actinf
    println("Successfully loaded actinf package")
catch e
    println("Could not load actinf directly: $e")
    
    # Try to include the module files manually
    try
        println("Attempting to manually include actinf module files...")
        
        # Include StateSpace.jl directly
        include(joinpath(@__DIR__, "actinf", "src", "StateSpace.jl"))
        include(joinpath(@__DIR__, "actinf", "src", "Inference.jl"))
        include(joinpath(@__DIR__, "actinf", "src", "Planning.jl"))
        include(joinpath(@__DIR__, "actinf", "src", "actinf.jl"))
        
        # Import the manually included module
        using .actinf
        println("Successfully loaded actinf via manual inclusion")
    catch e2
        println("Failed to manually include actinf: $e2")
        error("Cannot continue without actinf package")
    end
end

# Import specific components from actinf to make them available in Main scope
using actinf.StateSpace: DroneObservation, create_state_from_observation
using actinf.Inference: DroneBeliefs, initialize_beliefs, update_beliefs!, expected_state
using actinf.Planning: ActionPlanner, PreferenceModel, select_action, SUITABILITY_THRESHOLD

# Constants and parameters with more robust path handling for cross-platform compatibility
const INTERFACE_DIR = joinpath(@__DIR__, "interface")
const OBS_INPUT_PATH = joinpath(INTERFACE_DIR, "obs_input.json")
const PLANS_OUTPUT_PATH = joinpath(INTERFACE_DIR, "plans.json")
const SERVER_RUNNING_FLAG = joinpath(@__DIR__, ".zmq_server_running")
const DEFAULT_ZMQ_PORT = 5555
const ZMQ_SOCKET_ADDRESS = "tcp://localhost:$DEFAULT_ZMQ_PORT"
const POLICY_LENGTH = 3  # Number of steps in the policy

# Make sure interface directory exists
if !isdir(INTERFACE_DIR)
    mkpath(INTERFACE_DIR)
    println("Created interface directory: $INTERFACE_DIR")
end

# Set up ZMQ compatibility layer
using .ZMQCompat
ZMQCompat.setup_zmq_api(ZMQ)

function check_zmq_server_running()
    if isfile(SERVER_RUNNING_FLAG)
        return true
    end
    
    # Try to check via ping
    try
        # Create context and socket
        context = ZMQ.Context()
        socket = ZMQ.Socket(context, ZMQ.REQ)
        
        # Configure socket
        ZMQCompat.set_linger(socket, 500)
        
        # Try to connect with timeout
        println("Checking ZMQ server with ping...")
        ZMQCompat.connect(socket, ZMQ_SOCKET_ADDRESS)
        
        # Send ping and wait for response with timeout
        ZMQCompat.send(socket, "ping")
        
        # Try to receive with a timeout mechanism
        response = ""
        for i in 1:5  # Try 5 times with a short sleep
            if ZMQ.has_more(socket)
                response = String(ZMQCompat.recv(socket))
                break
            end
            sleep(0.1)
        end
        
        # Close socket and context
        ZMQCompat.close(socket)
        ZMQCompat.close(context)
        
        # Check if we got a response
        if response == "pong"
            println("ZMQ server responded to ping")
            return true
        end
    catch e
        println("Error checking ZMQ server: $e")
    end
    
    return false
end

function request_waypoint_from_zmq_server(obs_data)
    # Create context and socket
    context = ZMQ.Context()
    socket = ZMQ.Socket(context, ZMQ.REQ)
    
    # Configure socket options
    ZMQCompat.set_linger(socket, 500)
    ZMQCompat.set_rcvhwm(socket, 1000)
    ZMQCompat.set_sndhwm(socket, 1000)
    ZMQCompat.set_tcp_keepalive(socket, 1)
    
    try
        # Connect to server
        println("Connecting to ZMQ server at $ZMQ_SOCKET_ADDRESS...")
        ZMQCompat.connect(socket, ZMQ_SOCKET_ADDRESS)
        
        # Convert observation data to JSON and send
        json_data = JSON.json(obs_data)
        println("Sending observation data ($(length(json_data)) bytes)...")
        ZMQCompat.send(socket, json_data)
        
        # Wait for response with timeout handling
        println("Waiting for response...")
        response_received = false
        local json_response
        
        # Try multiple times with timeout
        for attempt in 1:5
            if ZMQ.has_more(socket) || attempt > 1
                response = String(ZMQCompat.recv(socket))
                json_response = JSON.parse(response)
                response_received = true
                break
            end
            println("Waiting for response (attempt $attempt)...")
            sleep(0.5)
        end
        
        if response_received
            println("Received response from ZMQ server")
            return json_response
        else
            println("No response received from ZMQ server within timeout")
            return Dict("error" => "No response from server within timeout")
        end
    catch e
        println("Error in ZMQ communication: $e")
        return Dict("error" => "ZMQ communication error: $e")
    finally
        # Always close socket and context
        try
            ZMQCompat.close(socket)
            ZMQCompat.close(context)
        catch e
            println("Error closing ZMQ resources: $e")
        end
    end
end

function main()
    # Print startup message
    println("\nStarting drone navigation planning...")
    
    # Check if ZMQ server is running, otherwise fallback to local computation
    use_zmq = check_zmq_server_running()
    if use_zmq
        println("ZMQ server is running - will use for planning calculations")
    else
        println("ZMQ server not detected - will use local computation")
    end
    
    # Check if input file exists
    if !isfile(OBS_INPUT_PATH)
        error("Input file $OBS_INPUT_PATH not found.")
    end

    # Read observation data from JSON with robust error handling
    println("\nReading sensor data...")
    
    local obs_data
    try
        obs_data = open(OBS_INPUT_PATH, "r") do file
            JSON.parse(file)
        end
        println("Successfully parsed observation data")
    catch e
        println("Error parsing JSON from $OBS_INPUT_PATH: $e")
        # Create fallback minimal data
        obs_data = Dict(
            "drone_position" => [0.0, 0.0, 0.0],
            "target_location" => [10.0, 0.0, -3.0],
            "drone_orientation" => [1.0, 0.0, 0.0, 0.0],
            "nearest_obstacle_distances" => [100.0, 100.0],
            "obstacle_density" => 0.0,
            "voxel_grid" => []
        )
        println("Using fallback observation data")
    end

    # If ZMQ server is running, use it for planning
    if use_zmq
        println("Using ZMQ server for waypoint planning...")
        response = request_waypoint_from_zmq_server(obs_data)
        
        if haskey(response, "error")
            println("Error from ZMQ server: $(response["error"])")
            println("Falling back to local computation")
            use_zmq = false
        else
            # Write the response to the plans file
            open(PLANS_OUTPUT_PATH, "w") do f
                JSON.print(f, response)
            end
            
            # Print results
            if haskey(response, "next_waypoint")
                waypoint = response["next_waypoint"]
                println("\nNext waypoint: [$(round(waypoint[1], digits=2)), $(round(waypoint[2], digits=2)), $(round(waypoint[3], digits=2))]")
                
                if haskey(response, "distance_to_target")
                    println("Distance to target: $(round(response["distance_to_target"], digits=2)) meters")
                end
                
                if haskey(response, "suitability")
                    println("Path suitability: $(round(response["suitability"], digits=2))")
                end
                
                println("\nPlanning completed successfully (via ZMQ server)")
                return
            else
                println("Invalid response from ZMQ server, falling back to local computation")
                use_zmq = false
            end
        end
    end
    
    # If we get here, we need to do local computation
    if !use_zmq
        println("\nPerforming local planning computation...")
        
        # Extract and print initial positions with better error checking
        drone_position = try
            SVector{3, Float64}(get(obs_data, "drone_position", [0.0, 0.0, 0.0])...)
        catch e
            println("Error parsing drone_position, using default: $e")
            SVector{3, Float64}(0.0, 0.0, 0.0)
        end
        
        target_location = try
            target_key = haskey(obs_data, "target_position") ? "target_position" : "target_location"
            SVector{3, Float64}(get(obs_data, target_key, [10.0, 0.0, -3.0])...)
        catch e
            println("Error parsing target position, using default: $e")
            SVector{3, Float64}(10.0, 0.0, -3.0)
        end

        # Extract remaining data with robust error handling
        drone_orientation = try
            SVector{4, Float64}(get(obs_data, "drone_orientation", [1.0, 0.0, 0.0, 0.0])...)
        catch e
            println("Error parsing drone_orientation, using default: $e")
            SVector{4, Float64}(1.0, 0.0, 0.0, 0.0)
        end
        
        nearest_obstacle_distances = try
            Float64.(get(obs_data, "obstacle_distances", get(obs_data, "nearest_obstacle_distances", [100.0, 100.0])))
        catch e
            println("Error parsing nearest_obstacle_distances, using default: $e")
            Float64[100.0, 100.0]
        end
        
        obstacle_density = try
            Float64(get(obs_data, "obstacle_density", 0.0))
        catch e
            println("Error parsing obstacle_density, using default: $e")
            0.0
        end
        
        # Handle obstacle positions with better key detection and error handling
        voxel_grid = Vector{SVector{3, Float64}}()
        try
            grid_data = Vector{Vector{Float64}}()
            
            # Try different possible keys for obstacle data
            if haskey(obs_data, "voxel_grid") && !isempty(obs_data["voxel_grid"])
                grid_data = obs_data["voxel_grid"]
                println("Using voxel_grid data with $(length(grid_data)) points")
            elseif haskey(obs_data, "obstacle_positions") && !isempty(obs_data["obstacle_positions"])
                grid_data = obs_data["obstacle_positions"]
                println("Using obstacle_positions data with $(length(grid_data)) points")
            end
            
            # Convert points to SVector format with validation
            for point in grid_data
                if length(point) >= 3
                    push!(voxel_grid, SVector{3, Float64}(point[1], point[2], point[3]))
                end
            end
            
            println("Processed $(length(voxel_grid)) valid obstacle points")
        catch e
            println("Error processing obstacle positions: $e")
            voxel_grid = Vector{SVector{3, Float64}}()
        end
        
        println("\nInitial Positions (Global Coordinates):")
        println("Drone:  [$(round(drone_position[1], digits=2)), $(round(drone_position[2], digits=2)), $(round(drone_position[3], digits=2))]")
        println("Target: [$(round(target_location[1], digits=2)), $(round(target_location[2], digits=2)), $(round(target_location[3], digits=2))]")

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
        
        # Initialize beliefs
        println("\nInitializing belief state...")
        beliefs = initialize_beliefs(current_state, voxel_grid=voxel_grid, obstacle_density=obstacle_density)
        
        # Update beliefs with current state
        update_beliefs!(beliefs, current_state, voxel_grid=voxel_grid, obstacle_density=obstacle_density)
        
        # Extract planning parameters with defaults
        max_step_size = get(obs_data, "max_step_size", 0.5)
        num_angles = get(obs_data, "num_angles", 8)
        num_step_sizes = get(obs_data, "num_step_sizes", 3)
        pragmatic_weight = get(obs_data, "pragmatic_weight", 1.0)
        epistemic_weight = get(obs_data, "epistemic_weight", 0.2)
        risk_weight = get(obs_data, "risk_weight", 2.0)
        safety_distance = get(obs_data, "safety_distance", 1.5)
        density_weight = get(obs_data, "density_weight", 1.0)
        obstacle_weight = get(obs_data, "obstacle_weight", 0.7)
        
        # Create planner
        planner = ActionPlanner(
            max_step_size = max_step_size,
            num_angles = num_angles,
            num_step_sizes = num_step_sizes,
            pragmatic_weight = pragmatic_weight,
            epistemic_weight = epistemic_weight,
            risk_weight = risk_weight,
            safety_distance = safety_distance,
            density_weight = density_weight,
            preference_model = PreferenceModel()
        )
        
        # Find min obstacle distance for safety calculations
        obstacle_distance = isempty(nearest_obstacle_distances) ? 100.0 : minimum(nearest_obstacle_distances)
        
        # Select actions using the planner
        println("\nSelecting actions...")
        selected_actions = select_action(
            current_state, 
            beliefs, 
            planner, 
            drone_position, 
            target_location, 
            obstacle_distance=obstacle_distance,
            obstacle_density=obstacle_density,
            num_policies=POLICY_LENGTH,
            obstacle_weight=obstacle_weight
        )
        
        # Extract best action and create waypoint
        if !isempty(selected_actions)
            best_action, best_efe = selected_actions[1]
            next_waypoint = drone_position + best_action
            
            println("\nBest action: [$(round(best_action[1], digits=2)), $(round(best_action[2], digits=2)), $(round(best_action[3], digits=2))]")
            println("Expected Free Energy: $(round(best_efe, digits=2))")
            println("Next waypoint: [$(round(next_waypoint[1], digits=2)), $(round(next_waypoint[2], digits=2)), $(round(next_waypoint[3], digits=2))]")
            println("Distance to target: $(round(current_state.distance, digits=2)) meters")
            println("Path suitability: $(round(current_state.suitability, digits=2))")
            
            # Create response data structure
            response = Dict(
                "next_waypoint" => [next_waypoint[1], next_waypoint[2], next_waypoint[3]],
                "action" => [best_action[1], best_action[2], best_action[3]],
                "efe" => best_efe,
                "suitability" => current_state.suitability,
                "distance_to_target" => current_state.distance,
                "angle_to_target" => [current_state.azimuth, current_state.elevation]
            )
            
            # Add policy information if available (more than one action)
            if length(selected_actions) > 1
                policy = []
                for i in 2:min(length(selected_actions), POLICY_LENGTH+1)
                    action, efe = selected_actions[i]
                    push!(policy, Dict(
                        "action" => [action[1], action[2], action[3]],
                        "efe" => efe
                    ))
                end
                response["policy"] = policy
            end
            
            # Write the response to the plans file
            open(PLANS_OUTPUT_PATH, "w") do f
                JSON.print(f, response)
            end
            
            println("\nPlanning completed successfully (local computation)")
        else
            println("\nWarning: No actions selected! Staying in place.")
            next_waypoint = drone_position
            
            # Create minimal response
            response = Dict(
                "next_waypoint" => [drone_position[1], drone_position[2], drone_position[3]],
                "error" => "No valid actions found",
                "suitability" => current_state.suitability,
                "distance_to_target" => current_state.distance
            )
            
            # Write the response to the plans file
            open(PLANS_OUTPUT_PATH, "w") do f
                JSON.print(f, response)
            end
            
            println("Planning failed - no valid actions found")
        end
    end
end

# Run the main function
try
    main()
    println("\nPlanning script completed successfully")
catch e
    println("\n❌ Error in planning script main function: $e")
    bt = backtrace()
    println("Stack trace:")
    for (i, frame) in enumerate(bt)
        if i <= 10 # Only print first 10 frames to keep it readable
            println("  $i: $frame")
        end
    end
    
    # Try to write error to plans file
    try
        error_response = Dict(
            "error" => "Planning script error: $e",
            "next_waypoint" => nothing
        )
        open(PLANS_OUTPUT_PATH, "w") do f
            JSON.print(f, error_response)
        end
    catch
        println("Could not write error to plans file")
    end
    
    exit(1)
end