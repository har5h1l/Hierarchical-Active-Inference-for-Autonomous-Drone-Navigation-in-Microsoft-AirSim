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
    using JSON, Dates, Statistics
catch e
    println("Adding JSON and Statistics packages...")
    Pkg.add("JSON")
    Pkg.add("Statistics")
    using JSON, Dates, Statistics
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

# Register signal handler for Ctrl+C using a modern approach
try
    # Modern way to handle signals in Julia
    if isdefined(Base, :handler_hooks)
        # Julia 1.7+ approach
        Base.handler_hooks[] = Base.handler_hooks[] ∪ [handle_interrupt]
    else
        # Fallback for compatibility with various Julia versions
        Base.sigatomic_begin()
        sig_handled = ccall(:jl_exit_on_sigint, Int32, (Int32,), 0) == 0
        if sig_handled
            # Direct call without using signal_hook
            Base.SIGINT_handler[] = handle_interrupt
        end
        Base.sigatomic_end()
    end
    println("✓ Signal handler registered successfully")
catch e
    println("Warning: Could not register signal handler: $e")
    # Continue execution even if we can't register the handler
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
        density_radius = get(observation_data, "density_radius", 4.0)
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
        
        # Calculate obstacle density with improved clustering-aware calculation
        obstacle_density = 0.0
        if !isempty(obstacle_positions)
            # The obstacle positions from Python are now already clustered, but we can apply
            # additional distance thresholding to ensure no over-counting of nearby obstacles
            
            # Step 1: Apply distance-based deduplication as additional safety measure
            min_obstacle_separation = 1.0  # Minimum distance to consider obstacles as separate
            filtered_positions = Vector{SVector{3, Float64}}()
            
            for pos in obstacle_positions
                # Check if this position is too close to any already filtered position
                is_too_close = false
                for filtered_pos in filtered_positions
                    if norm(pos - filtered_pos) < min_obstacle_separation
                        is_too_close = true
                        break
                    end
                end
                
                if !is_too_close
                    push!(filtered_positions, pos)
                end
            end
            
            # Step 2: Calculate density using filtered positions and adaptive radius
            if !isempty(filtered_positions)
                # Use adaptive density radius based on obstacle distribution
                adaptive_radius = density_radius
                
                # If we have many nearby obstacles, increase the radius to get better density estimate
                nearby_count = count(p -> norm(p - drone_position) < density_radius, filtered_positions)
                if nearby_count > 8
                    adaptive_radius = density_radius * 1.2  # Expand radius slightly for dense areas
                elseif nearby_count < 3
                    adaptive_radius = density_radius * 0.8  # Contract radius for sparse areas
                end
                
                # Count obstacles within adaptive radius
                final_count = count(p -> norm(p - drone_position) < adaptive_radius, filtered_positions)
                volume = (4/3) * π * adaptive_radius^3
                obstacle_density = final_count / volume
                
                println("Obstacle density calculation: $(length(obstacle_positions)) raw -> $(length(filtered_positions)) filtered -> $final_count within $(round(adaptive_radius, digits=2))m")
            end
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
          # Select best action - and get detailed metrics
        selected_action, details = select_action(
            current_state,
            beliefs,
            planner,
            drone_position,
            target_position,
            obstacle_distance=isempty(obstacle_distances) ? 10.0 : minimum(obstacle_distances),
            obstacle_density=obstacle_density,
            obstacle_weight=get(observation_data, "obstacle_distance_weight", 0.8),
            suitability_threshold=get(observation_data, "suitability_threshold", 0.75),
            return_details=true  # Get additional details for metrics
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
              # Compute EFE components for current state before building policy
            # We need this for delta_vfe calculation and to store in the best action
            current_efe_tuple = calculate_efe(
                current_state,
                beliefs,
                best_action,
                planner.preference_model,
                pragmatic_weight=planner.pragmatic_weight,
                epistemic_weight=planner.epistemic_weight,
                obstacle_density=obstacle_density,
                obstacle_distance=isempty(obstacle_distances) ? 10.0 : minimum(obstacle_distances)
            )
            
            # Extract the VFE and EFE components
            vfe = -current_efe_tuple[2]  # Negative pragmatic value gives VFE
            efe = current_efe_tuple[1]   # Total EFE
            efe_pragmatic = current_efe_tuple[2]  # Pragmatic component
            efe_epistemic = current_efe_tuple[3]  # Epistemic component
            
            # Add policy information with enhanced metrics
            policy = []
            
            # First, add the best action with all metrics
            best_action_vec = selected_action[1][1]
            best_action_score = selected_action[1][2]
            best_waypoint = drone_position + best_action_vec
            
            # Calculate heading angle of the action
            action_heading_angle_rad = atan(best_action_vec[2], best_action_vec[1])  # atan2(y, x)
            action_heading_angle_deg = action_heading_angle_rad * 180 / π
              # Calculate standard deviation of suitability for retained waypoints
            suitability_values = []
            if haskey(details, "sorted_indices") && haskey(details, "filtered_suitabilities")
                sorted_indices = details["sorted_indices"]
                filtered_suitabilities = details["filtered_suitabilities"]
                
                for i in 1:min(policy_length, length(selected_action))
                    if i <= length(sorted_indices)
                        action_idx = sorted_indices[i]
                        if action_idx <= length(filtered_suitabilities)
                            push!(suitability_values, filtered_suitabilities[action_idx])
                        end
                    end
                end
            end
            
            suitability_std = length(suitability_values) > 1 ? 
                              std(suitability_values) : 0.0
            
            # Create the policy entry with all metrics
            push!(policy, Dict(
                "action" => [best_action_vec[1], best_action_vec[2], best_action_vec[3]],
                "waypoint" => [best_waypoint[1], best_waypoint[2], best_waypoint[3]],
                "score" => best_action_score,
                "vfe" => vfe,
                "efe" => efe,
                "efe_vs_vfe_gap" => efe - vfe,
                "efe_pragmatic" => efe_pragmatic,
                "efe_epistemic" => efe_epistemic,
                "suitability_std" => suitability_std,
                "action_heading_angle_rad" => action_heading_angle_rad,
                "action_heading_angle_deg" => action_heading_angle_deg,
                "replanning_triggered_reason" => get(observation_data, "replanning_reason", "none")
            ))
            
            # Add the rest of the policy actions with basic information
            for i in 2:min(policy_length, length(selected_action))
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