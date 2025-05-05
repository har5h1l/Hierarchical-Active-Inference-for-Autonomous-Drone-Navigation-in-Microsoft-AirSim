#!/usr/bin/env julia

# ZeroMQ-based server for drone state inference and planning
# Replaces file-based communication with a direct ZMQ REP socket

println("Starting Active Inference ZMQ Server...")

# Import Pkg early for path handling and package loading
import Pkg

# Show current project and environment
println("Current directory: $(pwd())")
println("Project file: $(Base.active_project())")

# Ensure we're in the right project environment
Pkg.activate(dirname(@__DIR__))  # Activate parent directory project
println("Project activated: $(Base.active_project())")

# Make sure JSON is available for status file updates
try
    using JSON, Dates
catch e
    println("Adding JSON package...")
    Pkg.add("JSON")
    using JSON, Dates
end

# Create a status file to indicate server status for Python to check
const SERVER_RUNNING_FLAG = joinpath(dirname(@__DIR__), ".zmq_server_running")
const SERVER_STATUS_FILE = joinpath(dirname(@__DIR__), ".zmq_server_status.json")

# Function to update status file
function update_server_status(status, message, port=5555)
    try
        # Remove old flag file if it exists
        if isfile(SERVER_RUNNING_FLAG)
            rm(SERVER_RUNNING_FLAG)
        end
        
        # Create new flag file if server is running
        if status == "running"
            touch(SERVER_RUNNING_FLAG)
        end
        
        # Update JSON status file with more details
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
        println("Warning: Could not write server status file: $e")
    end
end

# Clean up any existing status files at startup
try
    if isfile(SERVER_RUNNING_FLAG)
        rm(SERVER_RUNNING_FLAG)
    end
    if isfile(SERVER_STATUS_FILE)
        rm(SERVER_STATUS_FILE)
    end
catch e
    println("Warning: Could not clean up existing status files: $e")
end

# Import all necessary packages with error handling
println("Loading required packages...")

# Helper function to load a package with clear error messaging
function load_package(pkg_name)
    try
        println("Loading package: $pkg_name")
        try
            # First try to use the package if it's already loaded
            @eval using $pkg_name
            println("✓ Successfully loaded $pkg_name")
            return true
        catch e
            # If using fails, try adding and then using
            println("⚠️ Could not load $pkg_name directly, trying to add it...")
            try
                Pkg.add(String(pkg_name))
                @eval using $pkg_name
                println("✓ Successfully added and loaded $pkg_name")
                return true
            catch e2
                println("❌ Failed to add and load $pkg_name: $e2")
                return false
            end
        end
    catch e
        println("❌ Critical error with $pkg_name: $e")
        return false
    end
end

# Load the essential packages
global packages_loaded = true
packages_loaded &= load_package(:ZMQ)
packages_loaded &= load_package(:JSON)
packages_loaded &= load_package(:LinearAlgebra)
packages_loaded &= load_package(:StaticArrays)

if !packages_loaded
    update_server_status("error", "Failed to load required packages")
    println("❌ Some required packages failed to load. Cannot continue.")
    exit(1)
end

# Initialize actinf package - critical path for drone navigation
println("\nInitializing actinf package...")

# First, develop the package to ensure it's available
try
    Pkg.develop(path=joinpath(@__DIR__))
    println("✓ actinf package developed")
catch e
    println("⚠️ Could not develop actinf package: $e")
end

# Try to load the actinf package
actinf_loaded = false
try
    @eval using actinf
    @eval using actinf.StateSpace
    @eval using actinf.Inference
    @eval using actinf.Planning
    println("✓ actinf package and modules loaded successfully")
    actinf_loaded = true
catch e
    println("❌ Failed to load actinf package: $e")
    
    # Try to include module files directly
    try
        println("Attempting to load modules manually...")
        include(joinpath(@__DIR__, "src", "actinf.jl"))
        include(joinpath(@__DIR__, "src", "StateSpace.jl"))
        include(joinpath(@__DIR__, "src", "Inference.jl")) 
        include(joinpath(@__DIR__, "src", "Planning.jl"))
        
        println("✓ actinf modules loaded manually")
        
        # Import modules into global namespace
        @eval using .StateSpace
        @eval using .Inference
        @eval using .Planning
        
        actinf_loaded = true
    catch e2
        println("❌ Failed to load actinf modules manually: $e2")
        update_server_status("error", "Failed to load actinf modules: $e2")
        exit(1)
    end
end

if !actinf_loaded
    update_server_status("error", "Failed to load actinf package")
    println("❌ Could not load required actinf package. Cannot continue.")
    exit(1)
end

# Define constants with more robust defaults
const DEFAULT_PORT = 5555
const SOCKET_ADDRESS = "tcp://*:$DEFAULT_PORT"
const POLICY_LENGTH = 3  # Number of steps in the policy
const SOCKET_TIMEOUT = 10000  # 10 seconds timeout
const MAX_RETRIES = 3

# Setup signal handling for clean shutdown
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

# Handle interrupts gracefully
function handle_interrupt(sig)
    println("\nReceived interrupt signal. Shutting down server cleanly...")
    update_server_status("stopped", "Server shutdown due to interrupt signal")
    exit(0)
end

# Register signal handlers for SIGINT (Ctrl+C)
try
    Base.sigatomic_begin()
    Base.signal_hook(@static Sys.iswindows() ? Base.SIGINT : Base.SIGINT, handle_interrupt)
    Base.sigatomic_end()
catch e
    println("Warning: Could not set up signal handlers: $e")
end

# Function to process observation data and return a response
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
            target_key = haskey(observation_data, "target_position") ? "target_position" : "target_location"
            SVector{3, Float64}(get(observation_data, target_key, [10.0, 0.0, -3.0])...)
        catch e
            println("Error parsing target position, using default: $e")
            SVector{3, Float64}(10.0, 0.0, -3.0)
        end
        
        # Get optional parameters with proper defaults
        waypoint_count = get(observation_data, "waypoint_count", 75)
        safety_margin = get(observation_data, "safety_margin", 1.5)
        policy_length = get(observation_data, "policy_length", 3)
        density_radius = get(observation_data, "density_radius", 5.0)
        
        # Get obstacle data with better error handling
        obstacle_positions = Vector{SVector{3, Float64}}()
        obstacle_distances = Float64[]
        
        try
            # Handle obstacle positions
            if haskey(observation_data, "obstacle_positions") && isa(observation_data["obstacle_positions"], Array)
                for pos in observation_data["obstacle_positions"]
                    if length(pos) >= 3
                        push!(obstacle_positions, SVector{3, Float64}(pos[1], pos[2], pos[3]))
                    end
                end
            end
            
            # Handle obstacle distances
            if haskey(observation_data, "obstacle_distances") && isa(observation_data["obstacle_distances"], Array)
                obstacle_distances = Float64.(observation_data["obstacle_distances"])
            end
        catch e
            println("Error processing obstacle data: $e")
        end
        
        # Calculate obstacle density if possible
        obstacle_density = 0.0
        if !isempty(obstacle_positions)
            try
                # Count obstacles within density_radius of drone position
                nearby_count = count(p -> norm(p - drone_position) < density_radius, obstacle_positions)
                volume = (4/3) * π * density_radius^3
                obstacle_density = nearby_count / volume
            catch e
                println("Error calculating obstacle density: $e")
            end
        end
        
        # Print key data for debugging
        println("Drone position: $(round.(drone_position; digits=2))")
        println("Target position: $(round.(target_position; digits=2))")
        println("Obstacle data: $(length(obstacle_positions)) positions, $(length(obstacle_distances)) distances")
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
        
        # Create state from observation
        current_state = create_state_from_observation(observation)
        
        # Initialize beliefs with obstacle data
        beliefs = initialize_beliefs(current_state, 
                                  voxel_grid=obstacle_positions, 
                                  obstacle_density=obstacle_density)
        
        # Update beliefs with current state
        update_beliefs!(beliefs, current_state, 
                      voxel_grid=obstacle_positions, 
                      obstacle_density=obstacle_density)
        
        # Create planner with appropriate parameters
        planner = ActionPlanner(
            PreferenceModel(
                distance_preference = get(observation_data, "distance_preference", 0.1),
                path_preference = get(observation_data, "path_preference", 0.9),
                safety_margin = safety_margin
            )
        )
        
        # Find min obstacle distance for safety calculations
        obstacle_distance = isempty(obstacle_distances) ? 100.0 : minimum(obstacle_distances)
        
        # Select action - adapt the function call based on your actual Planning module API
        # This is a key part that needs to match your Planning.select_action signature
        selected_action = select_action(planner, beliefs)
        
        # Calculate next waypoint
        next_waypoint = drone_position + selected_action
        
        println("Selected action: $(round.(selected_action; digits=2))")
        println("Next waypoint: $(round.(next_waypoint; digits=2))")
        
        # Prepare response with waypoint
        response = Dict(
            "next_waypoint" => [next_waypoint[1], next_waypoint[2], next_waypoint[3]],
            "action" => [selected_action[1], selected_action[2], selected_action[3]],
            "suitability" => current_state.suitability,
            "distance_to_target" => current_state.distance
        )
        
        # Add policy placeholder - you can expand this based on your actual implementation
        policy = []
        for _ in 1:policy_length
            push!(policy, Dict(
                "action" => [selected_action[1], selected_action[2], selected_action[3]],
                "waypoint" => [next_waypoint[1], next_waypoint[2], next_waypoint[3]]
            ))
        end
        response["policy"] = policy
        
        return response
    catch e
        println("Error in process_observation: $e")
        return Dict(
            "error" => true,
            "message" => "Error processing observation: $e"
        )
    end
end

# Setup ZMQ server infrastructure
try
    # Create ZMQ context and socket
    context = ZMQ.Context()
    socket = ZMQ.Socket(context, ZMQ.REP)
    
    # Configure socket options
    socket.linger = 500     # Wait up to 500ms to send pending messages on close
    
    # Try to set high water marks and TCP keepalive if available in this ZMQ version
    try
        socket.rcvhwm = 1000    # High water mark for incoming messages
        socket.sndhwm = 1000    # High water mark for outgoing messages
        socket.tcp_keepalive = 1 # Enable TCP keepalive
    catch e
        println("Note: Some socket options not available in this ZMQ version: $e")
    end
    
    # Register cleanup handler
    register_exit_handler() do
        println("Cleaning up ZMQ resources...")
        try
            close(socket)
            close(context)
            update_server_status("stopped", "Server shutdown cleanly")
        catch e
            println("Error during cleanup: $e")
        end
    end
    
    # Try to bind the socket - handle port conflicts
    bound_port = DEFAULT_PORT
    bound = false
    
    # First try the default port
    try
        println("Binding ZMQ socket to $SOCKET_ADDRESS...")
        ZMQ.bind(socket, SOCKET_ADDRESS)
        println("✓ Socket bound successfully to $SOCKET_ADDRESS")
        bound = true
    catch e
        println("❌ Failed to bind to $SOCKET_ADDRESS: $e")
    end
    
    # If default port fails, try alternative ports
    if !bound
        alternative_ports = [5556, 5557, 5558, 5559, 5560]
        
        for port in alternative_ports
            alt_address = "tcp://*:$port"
            try
                println("Trying alternative address: $alt_address")
                ZMQ.bind(socket, alt_address)
                println("✓ Socket bound successfully to $alt_address")
                bound_port = port
                bound = true
                break
            catch e2
                println("❌ Failed to bind to $alt_address: $e2")
            end
        end
    end
    
    if !bound
        println("❌ Failed to bind to any port. Server cannot start.")
        update_server_status("error", "Failed to bind to any port")
        exit(1)
    end
    
    # Update status with the actual bound port
    update_server_status("running", "Server started and listening", bound_port)
    
    # Server main loop
    println("\n=== Active Inference ZMQ Server Ready on Port $bound_port ===")
    println("Waiting for requests...")
    
    while true
        # Wait for requests
        try
            # Receive request from client
            request = String(ZMQ.recv(socket))
            
            # Handle special commands
            if request == "ping"
                println("Received ping request, sending pong")
                ZMQ.send(socket, "pong")
                continue
            elseif request == "status"
                println("Received status request")
                ZMQ.send(socket, "running")
                continue
            elseif request == "shutdown"
                println("Received shutdown request")
                ZMQ.send(socket, "shutting_down")
                update_server_status("stopped", "Server shutdown by client request")
                exit(0)
            end
            
            # Try to parse the request as JSON
            try
                # Parse JSON request
                println("Received request of length $(length(request)) bytes")
                observation_data = JSON.parse(request)
                
                # Process observation using our dedicated function
                response = process_observation(observation_data)
                
                # Send response
                response_json = JSON.json(response)
                println("Sending response: $(length(response_json)) bytes")
                ZMQ.send(socket, response_json)
                
            catch e
                # Handle errors in request processing
                println("Error processing request: $e")
                
                # Send error response
                error_response = Dict(
                    "error" => true,
                    "message" => "Error processing request: $e"
                )
                ZMQ.send(socket, JSON.json(error_response))
            end
            
        catch e
            # Handle socket errors
            if isa(e, InterruptException)
                println("Server interrupted. Shutting down...")
                update_server_status("stopped", "Server interrupted")
                break
            else
                println("Socket error: $e")
                # Continue and try to receive the next message
            end
        end
    end
    
catch e
    # Handle any unexpected errors
    update_server_status("error", "Unexpected error: $e")
    println("❌ Fatal error: $e")
    exit(1)
end

# Final cleanup
update_server_status("stopped", "Server shutdown")
exit(0)