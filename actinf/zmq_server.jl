#!/usr/bin/env julia

# ZeroMQ-based server for drone state inference and planning
# Replaces file-based communication with a direct ZMQ REP socket

println("Starting Active Inference ZMQ Server...")

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

# Ensure we can write to the status file before proceeding
status_file_path = nothing
try
    # Create status file path - handle both Windows and UNIX paths
    if Sys.iswindows()
        # Use absolute path with proper Windows separators
        parent_dir = dirname(dirname(abspath(@__FILE__)))
        status_file_path = joinpath(parent_dir, "zmq_server_running.status")
    else
        status_file_path = joinpath(dirname(dirname(@__FILE__)), "zmq_server_running.status")
    end
    
    # Test that we can write to this location
    println("Testing if we can write status file to: $status_file_path")
    touch(status_file_path)
    rm(status_file_path)
    println("✓ Status file location is writable")
catch e
    # Try another location if the first one fails
    println("❌ Could not write to status file location: $e")
    try
        if Sys.iswindows()
            # Try temp directory on Windows
            status_file_path = joinpath(tempdir(), "zmq_server_running.status")
        else
            status_file_path = joinpath("/tmp", "zmq_server_running.status")
        end
        println("Trying alternative status file location: $status_file_path")
        touch(status_file_path)
        rm(status_file_path)
        println("✓ Alternative status file location is writable")
    catch e2
        println("❌ Could not write to alternative status file location: $e2")
        println("Will continue without status file. Client may not detect server properly.")
        status_file_path = nothing
    end
end

# Activate project environment
println("Setting up Julia environment...")
if Base.active_project() != abspath(joinpath(dirname(@__DIR__), "Project.toml"))
    try
        import Pkg
        Pkg.activate(dirname(@__DIR__))
        println("✓ Project activated")
        try
            Pkg.develop(path=@__DIR__)  # Develop the actinf package
            println("✓ Actinf package developed")
        catch e
            println("⚠️ Could not develop actinf package: $e")
        end
    catch e
        println("❌ Error activating project: $e")
    end
end

# Check Julia version
println("Running on Julia version: $(VERSION)")

# Import all necessary packages with error handling
packages_loaded = true
packages_loaded &= load_package(:ZMQ)
packages_loaded &= load_package(:JSON)
packages_loaded &= load_package(:LinearAlgebra)
packages_loaded &= load_package(:StaticArrays)

# Load actinf package (with special handling)
try
    println("Loading actinf package...")
    @eval using actinf
    println("✓ actinf package loaded")
except e
    println("❌ Failed to load actinf package: $e")
    packages_loaded = false
end

if !packages_loaded
    println("❌ Some required packages failed to load. Cannot continue.")
    exit(1)
end

# Define constants
const SOCKET_ADDRESS = "tcp://*:5555"
const POLICY_LENGTH = 3  # Number of steps in the policy

# Print server info
println("Server initialized with:")
println("- Address: $SOCKET_ADDRESS")
println("- Policy length: $POLICY_LENGTH")
if status_file_path !== nothing
    println("- Status file: $status_file_path")
else
    println("- Status file: DISABLED")
end

# Initialize beliefs as a global variable for persistence across requests
global current_beliefs = nothing

# Function to process an observation and return inferred state and planned action
function process_observation(observation_data::Dict)
    # Extract observation data
    local drone_position, drone_orientation, target_location
    local nearest_obstacle_distances, obstacle_density, voxel_grid
    
    try
        # Extract data with safety checks
        drone_position = SVector{3, Float64}(get(observation_data, "drone_position", [0.0, 0.0, 0.0])...)
        drone_orientation = SVector{4, Float64}(get(observation_data, "drone_orientation", [1.0, 0.0, 0.0, 0.0])...)
        target_position_key = haskey(observation_data, "target_position") ? "target_position" : "target_location"
        target_location = SVector{3, Float64}(get(observation_data, target_position_key, [10.0, 0.0, -3.0])...)
        nearest_obstacle_distances = get(observation_data, "nearest_obstacle_distances", [100.0, 100.0])
        obstacle_density = get(observation_data, "obstacle_density", 0.0)
        
        # Parse voxel grid more efficiently with size information for logging
        raw_voxel_grid = get(observation_data, "voxel_grid", Vector{Vector{Float64}}())
        voxel_count = length(raw_voxel_grid)
        
        # Only convert voxel grid if it's not too large (optimization)
        max_voxels = 2000
        if voxel_count > max_voxels
            println("⚠️ Large voxel grid detected ($(voxel_count) points). Sampling to $(max_voxels) points.")
            # Take a random sample to keep processing time reasonable
            sample_indices = rand(1:voxel_count, min(max_voxels, voxel_count))
            raw_voxel_grid = raw_voxel_grid[sample_indices]
        end
        
        voxel_grid = [SVector{3, Float64}(point...) for point in raw_voxel_grid if length(point) == 3]
        
        println("Received data:")
        println("- Drone position: $(round.(drone_position, digits=2))")
        println("- Target position: $(round.(target_location, digits=2))")
        println("- Obstacle count: $(length(voxel_grid)) voxels")
        println("- Obstacle density: $(round(obstacle_density, digits=3))")
    catch e
        println("Error parsing observation data: $e")
        return Dict(
            "error" => "Failed to parse observation data: $e",
            "action" => [0.0, 0.0, 0.0]
        )
    end
    
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
    
    # Update or initialize beliefs
    global current_beliefs
    try
        if current_beliefs === nothing
            println("Initializing new belief system")
            current_beliefs = initialize_beliefs(current_state, voxel_grid=voxel_grid, obstacle_density=obstacle_density)
        else
            println("Updating existing belief system")
            update_beliefs!(current_beliefs, current_state; voxel_grid=voxel_grid, obstacle_density=obstacle_density)
        end
    catch e
        println("Error updating beliefs: $e")
        println("Reinitializing beliefs")
        current_beliefs = initialize_beliefs(current_state, voxel_grid=voxel_grid, obstacle_density=obstacle_density)
    end
    
    # Get expected state
    expected_drone_state = expected_state(current_beliefs)
    
    # Initialize preference model and action planner
    preference_model = PreferenceModel(
        distance_weight = 1.0,
        distance_scaling = 0.1,
        angle_weight = 0.8,
        angle_sharpness = 5.0,
        suitability_weight = 1.5,
        suitability_threshold = 0.3,
        max_distance = 50.0
    )
    
    planner = ActionPlanner(
        max_step_size = 1.0,
        num_angles = 8,
        num_step_sizes = 3,
        pragmatic_weight = 1.0,
        epistemic_weight = 0.2,
        risk_weight = 2.0,
        safety_distance = 1.5,
        density_weight = 1.0,
        preference_model = preference_model
    )
    
    # Plan actions
    println("Planning optimal action...")
    obstacle_distance = isempty(nearest_obstacle_distances) ? 100.0 : minimum(nearest_obstacle_distances)
    actions_with_efe = select_action(
        current_state,
        current_beliefs,
        planner,
        drone_position,
        target_location,
        obstacle_distance=obstacle_distance,
        obstacle_density=obstacle_density,
        num_policies=POLICY_LENGTH
    )
    
    # Extract best action and policy
    best_action, best_efe = actions_with_efe[1]
    policy = [action for (action, _) in actions_with_efe[1:min(POLICY_LENGTH, length(actions_with_efe))]]
    
    # Convert SVector to regular arrays for JSON serialization
    best_action_array = [best_action[1], best_action[2], best_action[3]]
    policy_array = [[a[1], a[2], a[3]] for a in policy]
    
    println("Selected waypoint: $(round.(best_action_array, digits=2))")
    
    # Create response
    response = Dict(
        "expected_state" => Dict(
            "distance" => expected_drone_state.distance,
            "azimuth" => expected_drone_state.azimuth,
            "elevation" => expected_drone_state.elevation,
            "suitability" => expected_drone_state.suitability
        ),
        "action" => best_action_array,
        "policy" => policy_array,
        "efe" => best_efe
    )
    
    return response
end

# Main function to run the ZMQ server
function run_server()
    # Initialize ZMQ context with IO threads
    context = Context(2)  # 2 IO threads is sufficient
    
    # Initialize reply socket with robust error handling
    println("Initializing ZMQ REP socket...")
    socket = Socket(context, REP)
    
    # Set socket options - keeping them minimal for better performance
    ZMQ.setsockopt(socket, ZMQ.LINGER, 0)  # Don't wait when closing
    ZMQ.setsockopt(socket, ZMQ.RCVHWM, 10)  # High water mark
    ZMQ.setsockopt(socket, ZMQ.SNDHWM, 10)
    
    # Enable keep-alive to detect disconnected clients
    ZMQ.setsockopt(socket, ZMQ.TCP_KEEPALIVE, 1)
    
    # Create a status file to signal that the server is running
    if status_file_path !== nothing
        try
            open(status_file_path, "w") do f
                write(f, "running")
            end
            println("Created status file: $status_file_path")
        catch e
            println("Warning: Could not create status file: $e")
        end
    end
    
    try
        # Bind to socket address
        println("Binding to $SOCKET_ADDRESS")
        ZMQ.bind(socket, SOCKET_ADDRESS)
        println("ZMQ server is ready and listening for connections!")
        
        # Server loop
        while true
            try
                # Wait for a message
                println("\nWaiting for request...")
                msg = ZMQ.recv(socket)
                msg_size = length(msg) / 1024  # Size in KB
                println("Received request: $(round(msg_size, digits=1)) KB")
                
                # Check if it's a ping message
                if length(msg) < 100
                    ping_msg = String(msg)
                    if occursin("ping", ping_msg)
                        println("Ping received, sending pong")
                        ZMQ.send(socket, "pong")
                        continue
                    end
                end
                
                # Parse JSON message and process observation
                local response
                try
                    observation = JSON.parse(String(msg))
                    println("Processing observation...")
                    response = process_observation(observation)
                    println("Processing complete")
                catch e
                    println("Error processing request: $e")
                    response = Dict(
                        "error" => "Processing error: $e",
                        "action" => [0.0, 0.0, 0.0]
                    )
                end
                
                # Send response
                println("Sending response...")
                response_json = JSON.json(response)
                response_size = length(response_json) / 1024  # Size in KB
                println("Response size: $(round(response_size, digits=1)) KB")
                
                ZMQ.send(socket, response_json)
                println("Response sent!")
                
            catch e
                println("Error in request-response cycle: $e")
                
                # Always try to send a response to maintain REQ-REP pattern
                try
                    err_response = Dict(
                        "error" => "Server error: $(typeof(e))",
                        "action" => [0.0, 0.0, 0.0]
                    )
                    ZMQ.send(socket, JSON.json(err_response))
                    println("Error response sent")
                catch send_error
                    println("Failed to send error response: $send_error")
                    
                    # Try to recreate the socket
                    try
                        ZMQ.close(socket)
                        socket = Socket(context, REP)
                        ZMQ.setsockopt(socket, ZMQ.LINGER, 0)
                        ZMQ.setsockopt(socket, ZMQ.RCVHWM, 10)
                        ZMQ.setsockopt(socket, ZMQ.SNDHWM, 10)
                        ZMQ.setsockopt(socket, ZMQ.TCP_KEEPALIVE, 1)
                        ZMQ.bind(socket, SOCKET_ADDRESS)
                        println("Socket recreated and rebound")
                    catch rebind_error
                        println("Failed to recreate socket: $rebind_error")
                        rethrow()
                    end
                end
            end
        end
    catch e
        println("Fatal server error: $e")
    finally
        # Clean up
        println("Closing ZMQ socket...")
        ZMQ.close(socket)
        println("Terminating ZMQ context...")
        ZMQ.term(context)
        
        # Remove status file
        if status_file_path !== nothing
            try
                if isfile(status_file_path)
                    rm(status_file_path)
                    println("Removed status file")
                end
            catch e
                println("Warning: Could not remove status file: $e")
            end
        end
    end
end

# Run the server
println("Starting ZMQ server...")
run_server()