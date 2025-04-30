#!/usr/bin/env julia

# ZeroMQ-based server for drone state inference and planning
# Replaces file-based communication with a direct ZMQ REP socket

println("Starting Active Inference ZMQ Server...")

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

# Ensure we can write to the status file before proceeding
global status_file_path = nothing
try
    # Create status file path - handle both Windows and UNIX paths
    if Sys.iswindows()
        # Use absolute path with proper Windows separators
        parent_dir = dirname(dirname(abspath(@__FILE__)))
        global status_file_path = joinpath(parent_dir, "zmq_server_running.status")
    else
        global status_file_path = joinpath(dirname(dirname(@__FILE__)), "zmq_server_running.status")
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
            global status_file_path = joinpath(tempdir(), "zmq_server_running.status")
        else
            global status_file_path = joinpath("/tmp", "zmq_server_running.status")
        end
        println("Trying alternative status file location: $status_file_path")
        touch(status_file_path)
        rm(status_file_path)
        println("✓ Alternative status file location is writable")
    catch e2
        println("❌ Could not write to alternative status file location: $e2")
        println("Will continue without status file. Client may not detect server properly.")
        global status_file_path = nothing
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
global packages_loaded = true
packages_loaded &= load_package(:ZMQ)
packages_loaded &= load_package(:JSON)
packages_loaded &= load_package(:LinearAlgebra)
packages_loaded &= load_package(:StaticArrays)

# Set up ZMQ compatibility layer
using .ZMQCompat
ZMQCompat.setup_zmq_api(ZMQ)

# Load actinf package (with special handling)
try
    println("Loading actinf package...")
    @eval using actinf
    println("✓ actinf package loaded")
catch e
    println("❌ Failed to load actinf package: $e")
    global packages_loaded = false
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
    local obstacle_repulsion_weight, suitability_weights, target_preference_weight
    
    try
        # Extract data with safety checks
        drone_position = SVector{3, Float64}(get(observation_data, "drone_position", [0.0, 0.0, 0.0])...)
        drone_orientation = SVector{4, Float64}(get(observation_data, "drone_orientation", [1.0, 0.0, 0.0, 0.0])...)
        target_position_key = haskey(observation_data, "target_position") ? "target_position" : "target_location"
        target_location = SVector{3, Float64}(get(observation_data, target_position_key, [10.0, 0.0, -3.0])...)
        nearest_obstacle_distances = get(observation_data, "nearest_obstacle_distances", [100.0, 100.0])
        obstacle_density = get(observation_data, "obstacle_density", 0.0)
        
        # Extract new parameters for suitability and target preference 
        obstacle_repulsion_weight = get(observation_data, "obstacle_repulsion_weight", 0.0)
        target_preference_weight = get(observation_data, "target_preference_weight", 0.5)
        
        # Get suitability weights if provided
        suitability_weights = get(observation_data, "suitability_weights", Dict())
        
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
        println("- Obstacle repulsion weight: $(round(obstacle_repulsion_weight, digits=3))")
        println("- Target preference weight: $(round(target_preference_weight, digits=3))")
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
    
    # Extract suitability weights if provided in the observation data
    obstacle_weight = 0.7  # Default value
    density_weight = 0.3   # Default value
    suitability_threshold = 0.3 # Default value
    
    if !isempty(suitability_weights)
        # Use provided weights from Python code
        obstacle_weight = get(suitability_weights, "nearest_obstacle_weight", 0.4)
        density_weight = get(suitability_weights, "obstacle_density_weight", 0.6)
        suitability_threshold = get(suitability_weights, "min_suitability_threshold", 0.65)
        println("Using custom suitability weights:")
        println("- Obstacle weight: $(obstacle_weight)")
        println("- Density weight: $(density_weight)")
        println("- Min suitability threshold: $(suitability_threshold)")
    end
    
    # Initialize preference model with updated weights
    preference_model = PreferenceModel(
        distance_weight = 1.0,
        distance_scaling = 0.1,
        angle_weight = 0.8,
        angle_sharpness = 5.0,
        suitability_weight = 1.5,
        suitability_threshold = suitability_threshold,  # Use the updated threshold
        max_distance = 50.0
    )
    
    # Initialize planner with updated density weight
    planner = ActionPlanner(
        max_step_size = 1.0,
        num_angles = 8,
        num_step_sizes = 3,
        pragmatic_weight = target_preference_weight > 0.5 ? 1.5 : 1.0,  # Increased when target preference is high
        epistemic_weight = 0.2,
        risk_weight = 2.0 + obstacle_repulsion_weight * 3.0,  # Scale up risk weight based on obstacle repulsion
        safety_distance = 1.5,
        density_weight = density_weight / obstacle_weight * 2.0,  # Scale density weight relative to obstacle weight
        preference_model = preference_model
    )
    
    # Override the suitability threshold used in planning
    # This ensures that the filtering step in select_action uses the updated threshold
    global SUITABILITY_THRESHOLD = suitability_threshold
    
    # Plan actions
    println("Planning optimal action...")
    obstacle_distance = isempty(nearest_obstacle_distances) ? 100.0 : minimum(nearest_obstacle_distances)
    
    # Get nearest obstacle distance, maybe from an additional field if available
    if haskey(observation_data, "nearest_obstacle_dist")
        obstacle_distance = observation_data["nearest_obstacle_dist"]
    end
    
    actions_with_efe = select_action(
        current_state,
        current_beliefs,
        planner,
        drone_position,
        target_location,
        obstacle_distance=obstacle_distance,
        obstacle_density=obstacle_density,
        num_policies=POLICY_LENGTH,
        obstacle_weight=obstacle_weight
    )
    
    # Extract best action and policy
    best_action, best_efe = actions_with_efe[1]
    policy = [action for (action, _) in actions_with_efe[1:min(POLICY_LENGTH, length(actions_with_efe))]]
    
    # Convert SVector to regular arrays for JSON serialization
    best_action_array = [best_action[1], best_action[2], best_action[3]]
    policy_array = [[a[1], a[2], a[3]] for a in policy]
    
    println("Selected waypoint: $(round.(best_action_array, digits=2))")
    
    # Ensure best_action_array is valid (not NaN or Inf)
    for i in 1:length(best_action_array)
        if isnan(best_action_array[i]) || isinf(best_action_array[i])
            println("⚠️ Invalid waypoint component detected: $(best_action_array)")
            best_action_array[i] = 0.0
        end
    end
    
    # Create response
    response = Dict(
        "expected_state" => Dict(
            "distance" => expected_drone_state.distance,
            "azimuth" => expected_drone_state.azimuth,
            "elevation" => expected_drone_state.elevation,
            "suitability" => expected_drone_state.suitability
        ),
        "action" => best_action_array,
        "waypoint" => best_action_array,
        "policy" => policy_array,
        "efe" => best_efe
    )
    
    # Extra validation of the response to ensure JSON serialization will work
    try
        # Test serialization before returning
        response_json = JSON.json(response)
        # Verify deserialization works too
        test_parse = JSON.parse(response_json)
        # Verify action array is present and valid
        if !haskey(test_parse, "action") || !(test_parse["action"] isa Vector) || length(test_parse["action"]) != 3
            println("⚠️ JSON serialization issue detected. Fixing response...")
            # Provide fallback values
            response["action"] = [0.0, 0.0, 0.0]
            response["waypoint"] = [0.0, 0.0, 0.0]
        end
    catch e
        println("⚠️ Error in JSON response validation: $e")
        # Provide fallback values for action and waypoint
        response["action"] = [0.0, 0.0, 0.0]
        response["waypoint"] = [0.0, 0.0, 0.0]
    end
    
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
    # Using compatibility layer methods
    ZMQCompat.set_linger(socket, 0)  # Don't wait when closing
    ZMQCompat.set_rcvhwm(socket, 10)  # High water mark
    ZMQCompat.set_sndhwm(socket, 10)
    
    # Enable keep-alive to detect disconnected clients
    # Note: TCP_KEEPALIVE might not be available in all ZMQ.jl versions
    try
        ZMQCompat.set_tcp_keepalive(socket, 1)
        println("TCP keepalive enabled")
    catch e
        println("Warning: Could not set TCP keepalive: $e")
    end
    
    # Create a status file to signal that the server is running
    local status_created = false
    if status_file_path !== nothing
        try
            open(status_file_path, "w") do f
                write(f, "running")
            end
            println("Created status file: $status_file_path")
            status_created = true
        catch e
            println("Warning: Could not create status file: $e")
            # Try again with permissions details
            try
                println("Attempting to write to status file with explicit permissions...")
                touch(status_file_path)
                chmod(status_file_path, 0o666)  # Make readable/writable by everyone
                open(status_file_path, "w") do f
                    write(f, "running")
                end
                println("Successfully created status file on second attempt!")
                status_created = true
            catch e2
                println("Error on second attempt: $e2")
                println("Will continue without status file. Client may not detect server properly.")
            end
        end
    end
    
    # Print binding information
    println("Status file created: $status_created")
    
    try
        # Bind to socket address
        println("Binding to $SOCKET_ADDRESS")
        ZMQCompat.bind(socket, SOCKET_ADDRESS)  # Use compatibility layer
        println("ZMQ server is ready and listening for connections!")
        
        # Server loop
        while true
            try
                # Wait for a message
                println("\nWaiting for request...")
                msg = ZMQCompat.recv(socket)  # Use compatibility layer
                msg_size = length(msg) / 1024  # Size in KB
                println("Received request: $(round(msg_size, digits=1)) KB")
                
                # Check if it's a ping message
                if length(msg) < 100
                    ping_msg = String(msg)
                    if occursin("ping", ping_msg)
                        println("Ping received, sending pong")
                        ZMQCompat.send(socket, "pong")  # Use compatibility layer
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
                        "action" => [0.0, 0.0, 0.0],
                        "waypoint" => [0.0, 0.0, 0.0],
                        "policy" => [[0.0, 0.0, 0.0]]
                    )
                end
                
                # Send response
                println("Sending response...")
                response_json = JSON.json(response)
                response_size = length(response_json) / 1024  # Size in KB
                println("Response size: $(round(response_size, digits=1)) KB")
                
                ZMQCompat.send(socket, response_json)  # Use compatibility layer
                println("Response sent!")
                
            catch e
                println("Error in request-response cycle: $e")
                
                # Always try to send a response to maintain REQ-REP pattern
                try
                    err_response = Dict(
                        "error" => "Server error: $(typeof(e))",
                        "action" => [0.0, 0.0, 0.0],
                        "waypoint" => [0.0, 0.0, 0.0],
                        "policy" => [[0.0, 0.0, 0.0]]
                    )
                    ZMQCompat.send(socket, JSON.json(err_response))  # Use compatibility layer
                    println("Error response sent")
                catch send_error
                    println("Failed to send error response: $send_error")
                    
                    # Try to recreate the socket
                    try
                        ZMQCompat.close(socket)  # Use compatibility layer
                        socket = Socket(context, REP)
                        # Set options using compatibility layer
                        ZMQCompat.set_linger(socket, 0)
                        ZMQCompat.set_rcvhwm(socket, 10)
                        ZMQCompat.set_sndhwm(socket, 10)
                        try
                            ZMQCompat.set_tcp_keepalive(socket, 1)
                        catch
                            # Ignore if not available
                        end
                        ZMQCompat.bind(socket, SOCKET_ADDRESS)  # Use compatibility layer
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
    
        # Clean up
        println("Closing ZMQ socket...")
        ZMQCompat.close(socket)  # Use compatibility layer
        println("Terminating ZMQ context...")
        ZMQCompat.close(context)  # Use compatibility layer
        
        # Remove status file
        if status_file_path !== nothing && status_created
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