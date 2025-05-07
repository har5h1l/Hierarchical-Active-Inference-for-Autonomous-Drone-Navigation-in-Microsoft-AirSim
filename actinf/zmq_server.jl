#!/usr/bin/env julia

# ZeroMQ-based server for drone navigation with active inference
# Simplified and optimized for reliable connections

println("\n=== Starting Active Inference ZMQ Server ===")

# Import basic packages immediately
import Pkg

# Ensure we're working in the correct project environment
println("Current directory: $(pwd())")
println("Activating project...")
Pkg.activate(dirname(@__DIR__))  # Activate main project directory
println("Project activated: $(Base.active_project())")

# Load essential packages
println("\nLoading required packages...")
try
    using JSON, Dates
catch e
    println("Adding JSON package...")
    Pkg.add("JSON")
    using JSON, Dates
end

# Set up status files for communication with Python
const SERVER_RUNNING_FLAG = joinpath(dirname(@__DIR__), ".zmq_server_running")
const SERVER_STATUS_FILE = joinpath(dirname(@__DIR__), ".zmq_server_status.json")

# Update status file function
function update_status(status, message, port=5555)
    try
        # Create or remove running flag
        if status == "running"
            touch(SERVER_RUNNING_FLAG)
        elseif isfile(SERVER_RUNNING_FLAG)
            rm(SERVER_RUNNING_FLAG)
        end
        
        # Update status file with detailed information
        open(SERVER_STATUS_FILE, "w") do f
            JSON.print(f, Dict(
                "status" => status,
                "message" => message,
                "port" => port,
                "timestamp" => string(Dates.now())
            ))
        end
        println(message)
    catch e
        println("Warning: Could not update status file: $e")
    end
end

# Clean up any existing status files at startup
try
    isfile(SERVER_RUNNING_FLAG) && rm(SERVER_RUNNING_FLAG)
    isfile(SERVER_STATUS_FILE) && rm(SERVER_STATUS_FILE)
catch e
    println("Warning: Could not clean up existing status files: $e")
end

# Load ZMQ and other required packages
for package in ["ZMQ", "StaticArrays", "LinearAlgebra"]
    try
        println("Loading $package package...")
        @eval using $(Symbol(package))
        println("✅ $package loaded successfully")
    catch e
        println("⚠️ Failed to load $package, attempting to add it...")
        try
            Pkg.add(package)
            @eval using $(Symbol(package))
            println("✅ $package added and loaded successfully")
        catch e2
            update_status("error", "Failed to load required package $package: $e2")
            println("❌ Failed to add $package: $e2")
            exit(1)
        end
    end
end

# Load actinf package
println("\nLoading actinf package...")
try
    # Ensure actinf package is developed properly
    Pkg.develop(path=joinpath(@__DIR__))
    println("✅ actinf package developed")
    
    # Load the package and its modules
    @eval using actinf
    @eval using actinf.StateSpace
    @eval using actinf.Inference
    @eval using actinf.Planning
    println("✅ actinf package and modules loaded")
catch e
    update_status("error", "Failed to load actinf package: $e")
    println("❌ Failed to load actinf package: $e")
    exit(1)
end

# Define server constants
const DEFAULT_PORT = 5555
const MAX_PORT_RETRIES = 5
const SOCKET_TIMEOUT = 10000  # milliseconds
const POLICY_LENGTH = 3

# Handle clean shutdown
import Base.exit
exit_handlers = []

function register_exit_handler(f)
    global exit_handlers
    push!(exit_handlers, f)
end

# Override exit to run handlers
orig_exit = exit
function exit(code::Int=0)
    for handler in exit_handlers
        try
            handler()
        catch e
            println("Error in exit handler: $e")
        end
    end
    orig_exit(code)
end

# Signal handler for cleaner shutdowns
function handle_interrupt(sig)
    println("\nReceived interrupt signal. Shutting down server cleanly...")
    update_status("stopped", "Server shutdown due to interrupt signal")
    exit(0)
end

# Register signal handler for Ctrl+C
try
    Base.sigatomic_begin()
    Base.signal_hook(Base.SIGINT, handle_interrupt)
    Base.sigatomic_end()
catch e
    println("Warning: Could not register signal handler: $e")
end

# Process observation function
function process_observation(observation_data::Dict)
    try
        # Extract data from observation with proper error handling
        drone_position = try
            SVector{3, Float64}(get(observation_data, "drone_position", [0.0, 0.0, 0.0])...)
        catch e
            println("Error parsing drone_position, using default: $e")
            SVector{3, Float64}(0.0, 0.0, 0.0)
        end
        
        drone_orientation = try
            if haskey(observation_data, "drone_orientation") && length(observation_data["drone_orientation"]) >= 4
                SVector{4, Float64}(observation_data["drone_orientation"]...)
            else
                SVector{4, Float64}(1.0, 0.0, 0.0, 0.0)  # Default identity quaternion
            end
        catch e
            println("Error parsing drone_orientation, using default: $e")
            SVector{4, Float64}(1.0, 0.0, 0.0, 0.0)
        end
        
        target_position = try
            SVector{3, Float64}(get(observation_data, "target_position", [10.0, 0.0, -3.0])...)
        catch e
            println("Error parsing target_position, using default: $e")
            SVector{3, Float64}(10.0, 0.0, -3.0)
        end
        
        # Get optional parameters
        safety_margin = get(observation_data, "safety_margin", 1.5)
        density_radius = get(observation_data, "density_radius", 5.0)
        policy_length = get(observation_data, "policy_length", POLICY_LENGTH)
        
        # Handle obstacle data
        obstacle_positions = Vector{SVector{3, Float64}}()
        obstacle_distances = Float64[]
        
        if haskey(observation_data, "obstacle_positions")
            for pos in observation_data["obstacle_positions"]
                if length(pos) >= 3
                    push!(obstacle_positions, SVector{3, Float64}(pos...))
                end
            end
        end
        
        if haskey(observation_data, "obstacle_distances")
            obstacle_distances = Float64.(observation_data["obstacle_distances"])
        end
        
        # Calculate obstacle density
        obstacle_density = 0.0
        if !isempty(obstacle_positions)
            # Count obstacles within density_radius of drone position
            nearby_count = count(p -> norm(p - drone_position) < density_radius, obstacle_positions)
            volume = (4/3) * π * density_radius^3
            obstacle_density = nearby_count / volume
        end
        
        # Print key information for debugging
        println("Drone position: $(round.(drone_position; digits=2))")
        println("Target position: $(round.(target_position; digits=2))")
        println("Obstacles: $(length(obstacle_positions)) positions, $(length(obstacle_distances)) distances")
        println("Obstacle density: $(round(obstacle_density; digits=4))")
        
        # Create observation object
        observation = DroneObservation(
            drone_position = drone_position,
            drone_orientation = drone_orientation,
            target_position = target_position,
            nearest_obstacle_distances = obstacle_distances,
            voxel_grid = obstacle_positions,
            obstacle_density = obstacle_density
        )
        
        # Convert observation to state
        current_state = create_state_from_observation(observation)
        
        # Initialize and update beliefs
        beliefs = initialize_beliefs(current_state)
        update_beliefs!(beliefs, current_state)
        
        # Create planner with proper parameters
        planner = ActionPlanner(
            preference_model = PreferenceModel(
                distance_weight = get(observation_data, "distance_weight", 2.0),
                angle_weight = get(observation_data, "angle_weight", 0.5),
                suitability_weight = get(observation_data, "suitability_weight", 1.0),
                suitability_threshold = get(observation_data, "safety_margin", 1.5) / 5.0 # Convert safety margin to appropriate threshold
            )
        )
        
        # Select best action
        selected_action = select_action(
            current_state,
            beliefs,
            planner,
            drone_position,
            target_position,
            obstacle_distance=isempty(obstacle_distances) ? 10.0 : minimum(obstacle_distances),
            obstacle_density=obstacle_density
        )
        
        # Extract the best action (first in the returned list)
        if !isempty(selected_action) && length(selected_action[1]) >= 1
            best_action = selected_action[1][1]  # First action from the first policy
            
            # Calculate waypoint from action
            next_waypoint = drone_position + best_action
            println("Selected action: $(round.(best_action; digits=2))")
            println("Next waypoint: $(round.(next_waypoint; digits=2))")
            
            # Create response with waypoint and other information
            response = Dict(
                "next_waypoint" => [next_waypoint[1], next_waypoint[2], next_waypoint[3]],
                "action" => [best_action[1], best_action[2], best_action[3]],
                "distance_to_target" => current_state.distance,
                "suitability" => current_state.suitability
            )
            
            # Add policy information
            policy = []
            for i in 1:min(policy_length, length(selected_action))
                action_tuple = selected_action[i]
                action_vec = action_tuple[1]
                waypoint = drone_position + action_vec
                
                push!(policy, Dict(
                    "action" => [action_vec[1], action_vec[2], action_vec[3]],
                    "waypoint" => [waypoint[1], waypoint[2], waypoint[3]],
                    "score" => action_tuple[2]
                ))
            end
            response["policy"] = policy
            
            return response
        else
            # If no valid action was selected, stay in place
            println("No valid action selected, staying in place")
            return Dict(
                "next_waypoint" => [drone_position[1], drone_position[2], drone_position[3]],
                "action" => [0.0, 0.0, 0.0],
                "error" => true,
                "message" => "Could not compute valid action",
                "policy" => []
            )
        end
    catch e
        println("Error processing observation: $e")
        return Dict("error" => true, "message" => "Error processing observation: $e")
    end
end

# Start ZMQ server
println("\nInitializing ZMQ server...")
try
    # Create ZMQ context and socket
    context = ZMQ.Context()
    socket = ZMQ.Socket(context, ZMQ.REP)
    
    # Configure socket options
    socket.linger = 1000     # Wait up to 1000ms on close
    
    # Try to set additional options if supported
    try
        socket.rcvtimeo = SOCKET_TIMEOUT
        socket.sndtimeo = SOCKET_TIMEOUT
        socket.rcvhwm = 10   # Receive high water mark
        socket.sndhwm = 10   # Send high water mark
    catch e
        println("Note: Some socket options not available: $e")
    end
    
    # Register cleanup handler
    register_exit_handler() do
        println("Cleaning up ZMQ resources...")
        try
            close(socket)
            close(context)
            update_status("stopped", "Server shutdown cleanly")
        catch e
            println("Error during cleanup: $e")
        end
    end
    
    # Try to bind to a port
    bound_port = DEFAULT_PORT
    success = false
    
    for port_attempt in 1:MAX_PORT_RETRIES
        current_port = DEFAULT_PORT + port_attempt - 1
        bind_address = "tcp://*:$current_port"
        
        try
            println("Attempting to bind to $bind_address...")
            ZMQ.bind(socket, bind_address)
            bound_port = current_port
            success = true
            println("✅ Successfully bound to port $bound_port")
            break
        catch e
            println("Failed to bind to port $current_port: $e")
            if port_attempt == MAX_PORT_RETRIES
                println("❌ Failed to bind to any port after $MAX_PORT_RETRIES attempts")
                update_status("error", "Failed to bind to any port")
                exit(1)
            end
        end
    end
    
    # Update status with actual bound port
    update_status("running", "Server started and listening", bound_port)
    
    # Server main loop
    println("\n=== Active Inference ZMQ Server Ready on Port $bound_port ===")
    println("Waiting for requests...")
    
    while true
        try
            # Wait for a message
            request = String(ZMQ.recv(socket))
            
            # Handle special commands
            if request == "ping"
                println("Received ping, responding with pong")
                ZMQ.send(socket, "pong")
                continue
            elseif request == "status"
                println("Received status request")
                ZMQ.send(socket, "running")
                continue
            elseif request == "shutdown"
                println("Received shutdown request")
                ZMQ.send(socket, "shutting_down")
                update_status("stopped", "Server shutdown by client request")
                exit(0)
            end
            
            # Process normal requests
            println("\nReceived request ($(length(request)) bytes)")
            
            try
                # Parse JSON request
                observation_data = JSON.parse(request)
                
                # Process observation
                response = process_observation(observation_data)
                
                # Send response
                response_json = JSON.json(response)
                println("Sending response ($(length(response_json)) bytes)")
                ZMQ.send(socket, response_json)
            catch e
                println("Error processing request: $e")
                error_response = Dict("error" => true, "message" => "Error: $e")
                ZMQ.send(socket, JSON.json(error_response))
            end
        catch e
            if isa(e, InterruptException)
                println("Server interrupted. Shutting down...")
                update_status("stopped", "Server interrupted")
                break
            else
                println("Socket error: $e")
                # Continue and try to receive again
            end
        end
    end
    
catch e
    update_status("error", "Fatal error: $e")
    println("❌ Fatal error: $e")
    exit(1)
end

# Final cleanup
update_status("stopped", "Server shutdown")
exit(0)