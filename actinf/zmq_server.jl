#!/usr/bin/env julia

# ZeroMQ-based server for drone state inference and planning
# Replaces file-based communication with a direct ZMQ REP socket

println("Starting Active Inference ZMQ Server...")

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

# Activate the project environment first to ensure packages are available
println("Activating project environment...")
try
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    println("Project activated")
    
    # Import JSON for status file creation
    import JSON
    import Dates
catch e
    println("❌ Error activating project: $e")
    update_server_status("error", "Failed to activate project: $e")
    exit(1)
end

# ZMQ.jl API compatibility layer
# This ensures the script works with both newer and older versions of ZMQ.jl
module ZMQCompat
    # Export all the functions we'll use as compatibility wrappers
    export setup_zmq_api, bind, send, recv, close, set_linger, set_rcvhwm, set_sndhwm, set_tcp_keepalive
    
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
            sock = zmq_mod.Socket(ctx, zmq_mod.REP)
            
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
    function bind(socket, address)
        if using_new_api
            return zmq_module.bind(socket, address)
        else
            return zmq_module.bind(socket, address)  # Same in both APIs
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

# Explicit package loading with error handling
function load_package(pkg_name)
    try
        println("Loading package: $pkg_name")
        @eval import Pkg
        try
            # First try to use the package if it's already loaded
            @eval using $pkg_name
            println("✓ Successfully loaded $pkg_name")
            return true
        catch e
            # If using fails, try adding and then using
            println("⚠️ Could not load $pkg_name directly, trying to add it...")
            try
                Pkg.add("$pkg_name")
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

# Check Julia version
println("Running on Julia version: $(VERSION)")

# Import all necessary packages with error handling
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

# Set up ZMQ compatibility layer
using .ZMQCompat
ZMQCompat.setup_zmq_api(ZMQ)

# Load actinf package (with special handling)
try
    println("Loading actinf package...")
    @eval using actinf
    println("✓ actinf package loaded")
    
    # Directly import needed components from modules
    @eval using actinf.StateSpace: DroneState, DroneObservation, create_state_from_observation
    @eval using actinf.Inference: DroneBeliefs, initialize_beliefs, update_beliefs!, expected_state, serialize_beliefs, deserialize_beliefs
    @eval using actinf.Planning: ActionPlanner, PreferenceModel, select_action, SUITABILITY_THRESHOLD
    println("✓ Module components imported")
catch e
    println("❌ Failed to load actinf package: $e")
    println("Trying to include manually...")
    
    try
        # Include files directly
        include(joinpath(@__DIR__, "src", "StateSpace.jl"))
        include(joinpath(@__DIR__, "src", "Inference.jl"))
        include(joinpath(@__DIR__, "src", "Planning.jl"))
        include(joinpath(@__DIR__, "src", "actinf.jl"))
        
        # Import modules
        using .StateSpace
        using .Inference
        using .Planning
        println("✓ actinf modules loaded manually")
    catch e2
        println("❌ Failed to manually include actinf modules: $e2")
        update_server_status("error", "Failed to load actinf modules: $e2")
        global packages_loaded = false
        exit(1)
    end
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
    trap = Base.signal_hook
    Base.signal_hook(@static Sys.iswindows() ? Base.SIGINT : Base.SIGINT, handle_interrupt)
    Base.sigatomic_end()
catch e
    println("Warning: Could not set up signal handlers: $e")
end

# Function to process observation data and return a response
function process_observation(observation_data::Dict)
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
    obstacle_weight = get(observation_data, "obstacle_weight", 0.7)
    density_weight = get(observation_data, "density_weight", 1.0)
    
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
    if !isempty(obstacle_positions) && haskey(observation_data, "density_radius")
        density_radius = get(observation_data, "density_radius", 5.0)
        try
            # Count obstacles within density_radius of drone position
            nearby_count = count(p -> norm(p - drone_position) < density_radius, obstacle_positions)
            volume = (4/3) * π * density_radius^3
            obstacle_density = nearby_count / volume
        catch e
            println("Error calculating obstacle density: $e")
        end
    end
    
    # Create observation object for state creation
    observation = DroneObservation(
        drone_position = drone_position,
        drone_orientation = drone_orientation,
        target_position = target_position,
        nearest_obstacle_distances = obstacle_distances,
        voxel_grid = obstacle_positions,
        obstacle_density = obstacle_density
    )
    
    # Print key data for debugging
    println("Drone position: $(round.(drone_position, digits=2))")
    println("Target position: $(round.(target_position, digits=2))")
    println("Obstacle density: $(round(obstacle_density, digits=4))")
    println("$(length(obstacle_positions)) obstacle positions, $(length(obstacle_distances)) distances")
    
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
    
    # Create planner with appropriate parameters matching the Planning module's API
    planner = ActionPlanner(
        max_step_size = get(observation_data, "max_step_size", 0.5),
        num_angles = get(observation_data, "num_angles", 8),
        num_step_sizes = get(observation_data, "num_step_sizes", 3),
        pragmatic_weight = get(observation_data, "pragmatic_weight", 1.0),
        epistemic_weight = get(observation_data, "epistemic_weight", 0.2),
        risk_weight = get(observation_data, "risk_weight", 2.0),
        safety_distance = get(observation_data, "safety_distance", 1.5),
        density_weight = density_weight,
        preference_model = PreferenceModel()
    )
    
    # Find min obstacle distance for safety calculations
    obstacle_distance = isempty(obstacle_distances) ? 100.0 : minimum(obstacle_distances)
    
    # Select the best actions using select_action matching the Planning module's API
    selected_actions = select_action(
        current_state, 
        beliefs, 
        planner,
        drone_position, 
        target_position,
        obstacle_distance=obstacle_distance,
        obstacle_density=obstacle_density,
        num_policies=get(observation_data, "policy_length", 3),
        obstacle_weight=obstacle_weight
    )
    
    # Extract best action (first in the list of selected actions)
    if !isempty(selected_actions)
        best_action, best_efe = selected_actions[1]
        next_waypoint = drone_position + best_action
        
        println("Best action: $(round.(best_action, digits=2)), EFE: $(round(best_efe, digits=2))")
        println("Next waypoint: $(round.(next_waypoint, digits=2))")
        
        # Prepare response with waypoint and additional data
        response = Dict(
            "next_waypoint" => [next_waypoint[1], next_waypoint[2], next_waypoint[3]],
            "suitability" => current_state.suitability,
            "distance_to_target" => current_state.distance,
            "angle_to_target" => [current_state.azimuth, current_state.elevation],
            "action" => [best_action[1], best_action[2], best_action[3]],
            "efe" => best_efe
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
        
        return response
    else
        println("Warning: No actions selected, returning stay-in-place action")
        return Dict(
            "next_waypoint" => [drone_position[1], drone_position[2], drone_position[3]],
            "suitability" => current_state.suitability,
            "distance_to_target" => current_state.distance,
            "error" => "No valid actions found"
        )
    end
end

# Setup ZMQ server infrastructure
try
    # Create ZMQ context
    context = ZMQ.Context()
    socket = ZMQ.Socket(context, ZMQ.REP)
    
    # Configure socket options
    ZMQCompat.set_linger(socket, 500)     # Wait up to 500ms to send pending messages on close
    ZMQCompat.set_rcvhwm(socket, 1000)    # High water mark for incoming messages
    ZMQCompat.set_sndhwm(socket, 1000)    # High water mark for outgoing messages
    ZMQCompat.set_tcp_keepalive(socket, 1) # Enable TCP keepalive
    
    # Register cleanup handler
    register_exit_handler() do
        println("Cleaning up ZMQ resources...")
        try
            ZMQCompat.close(socket)
            ZMQCompat.close(context)
            update_server_status("stopped", "Server shutdown cleanly")
        catch e
            println("Error during cleanup: $e")
        end
    end
    
    # Try to bind the socket - handle port conflicts
    try
        println("Binding ZMQ socket to $SOCKET_ADDRESS...")
        ZMQCompat.bind(socket, SOCKET_ADDRESS)
        println("✓ Socket bound successfully to $SOCKET_ADDRESS")
        update_server_status("running", "Server started and listening", DEFAULT_PORT)
    catch e
        println("❌ Failed to bind to $SOCKET_ADDRESS: $e")
        
        # Try alternative ports
        alternative_ports = [5556, 5557, 5558, 5559, 5560]
        bound = false
        
        for port in alternative_ports
            alt_address = "tcp://*:$port"
            try
                println("Trying alternative address: $alt_address")
                ZMQCompat.bind(socket, alt_address)
                println("✓ Socket bound successfully to $alt_address")
                update_server_status("running", "Server started on alternative port", port)
                bound = true
                break
            catch e2
                println("❌ Failed to bind to $alt_address: $e2")
            end
        end
        
        if !bound
            println("❌ Failed to bind to any port. Server cannot start.")
            update_server_status("error", "Failed to bind to any port")
            exit(1)
        end
    end
    
    # Server main loop
    println("\n=== Active Inference ZMQ Server Ready ===")
    println("Waiting for requests...")
    
    while true
        # Wait for requests
        try
            # Receive request from client
            request = String(ZMQCompat.recv(socket))
            
            # Handle special commands
            if request == "ping"
                println("Received ping request, sending pong")
                ZMQCompat.send(socket, "pong")
                continue
            elseif request == "status"
                println("Received status request")
                ZMQCompat.send(socket, "running")
                continue
            elseif request == "shutdown"
                println("Received shutdown request")
                ZMQCompat.send(socket, "shutting_down")
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
                ZMQCompat.send(socket, JSON.json(response))
                
            catch e
                # Handle errors in request processing
                println("Error processing request: $e")
                
                # Send error response
                error_response = Dict(
                    "error" => true,
                    "message" => "Error processing request: $e"
                )
                ZMQCompat.send(socket, JSON.json(error_response))
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