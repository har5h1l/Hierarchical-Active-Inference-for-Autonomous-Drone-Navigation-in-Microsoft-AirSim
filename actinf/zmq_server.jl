#!/usr/bin/env julia

# ZeroMQ-based server for drone state inference and planning
# Replaces file-based communication with a direct ZMQ REP socket

println("Starting Active Inference ZMQ Server...")

# Activate project environment
if Base.active_project() != abspath(joinpath(dirname(@__DIR__), "Project.toml"))
    import Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.develop(path=@__DIR__)  # Develop the actinf package
end

# Import all necessary packages
using ZMQ
using JSON
using LinearAlgebra
using StaticArrays
using actinf

# Define constants
const SOCKET_ADDRESS = "tcp://*:5555"
const POLICY_LENGTH = 3  # Number of steps in the policy
const RECV_TIMEOUT = 30000  # Socket receive timeout in ms (30 seconds)
const SEND_TIMEOUT = 5000   # Socket send timeout in ms (5 seconds)
const HIGH_WATER_MARK = 10  # Maximum number of messages to queue
const SOCKET_BUFFER_SIZE = 8 * 1024 * 1024  # 8MB buffer for large messages

# Print server info
println("Server initialized with:")
println("- Address: $SOCKET_ADDRESS")
println("- Policy length: $POLICY_LENGTH")
println("- Receive timeout: $(RECV_TIMEOUT/1000)s")
println("- Buffer size: $(SOCKET_BUFFER_SIZE/1024/1024)MB")

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
        target_location = SVector{3, Float64}(get(observation_data, "target_position", [10.0, 0.0, -3.0])...)
        nearest_obstacle_distances = get(observation_data, "nearest_obstacle_distances", [100.0, 100.0])
        obstacle_density = get(observation_data, "obstacle_density", 0.0)
        
        # Parse voxel grid more efficiently with size information for logging
        raw_voxel_grid = get(observation_data, "voxel_grid", Vector{Vector{Float64}}())
        voxel_count = length(raw_voxel_grid)
        
        # Only convert voxel grid if it's not too large (optimization)
        max_voxels = 5000  # Limit processing to reasonable number
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
    context = Context(3)  # Use 3 IO threads for better performance
    
    # Initialize reply socket with robust error handling
    println("Initializing ZMQ REP socket...")
    socket = Socket(context, REP)
    
    # Set socket options for better handling of large messages
    ZMQ.setsockopt(socket, ZMQ.RCVTIMEO, RECV_TIMEOUT)
    ZMQ.setsockopt(socket, ZMQ.SNDTIMEO, SEND_TIMEOUT)
    ZMQ.setsockopt(socket, ZMQ.RCVHWM, HIGH_WATER_MARK)
    ZMQ.setsockopt(socket, ZMQ.SNDHWM, HIGH_WATER_MARK)
    
    # Set buffer sizes for large messages
    ZMQ.setsockopt(socket, ZMQ.RCVBUF, SOCKET_BUFFER_SIZE)
    ZMQ.setsockopt(socket, ZMQ.SNDBUF, SOCKET_BUFFER_SIZE)
    
    # Enable keep-alive to detect disconnected clients
    ZMQ.setsockopt(socket, ZMQ.TCP_KEEPALIVE, 1)
    ZMQ.setsockopt(socket, ZMQ.TCP_KEEPALIVE_IDLE, 120)  # Start probing after 120s
    ZMQ.setsockopt(socket, ZMQ.TCP_KEEPALIVE_INTVL, 60)  # Probe every 60s
    
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
                
                # Check if it's a ping message (small message to test connection)
                if length(msg) < 100
                    ping_msg = String(msg)
                    if occursin("ping", ping_msg)
                        println("Ping received, sending pong")
                        ZMQ.send(socket, "pong")
                        continue
                    end
                end
                
                # Parse JSON message with timeout protection
                local observation, response
                
                # Run parsing and processing in a separate task with timeout
                task = @task begin
                    try
                        observation = JSON.parse(String(msg))
                        # Process observation
                        println("Processing observation...")
                        response = process_observation(observation)
                        println("Processing complete")
                        return response
                    catch e
                        println("Error in processing task: $e")
                        return Dict(
                            "error" => "Processing error: $e",
                            "action" => [0.0, 0.0, 0.0]
                        )
                    end
                end
                
                # Schedule the task
                schedule(task)
                
                # Wait for task with timeout
                timeout = 25  # seconds
                start_time = time()
                while !istaskdone(task) && time() - start_time < timeout
                    sleep(0.1)
                end
                
                if !istaskdone(task)
                    println("Processing timed out after $timeout seconds")
                    response = Dict(
                        "error" => "Processing timed out",
                        "action" => [0.0, 0.0, 0.0]
                    )
                else
                    response = fetch(task)
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
                
                # CRITICAL: Always send a response to maintain REQ-REP pattern
                try
                    err_response = Dict(
                        "error" => "Server error: $(typeof(e))",
                        "action" => [0.0, 0.0, 0.0]
                    )
                    ZMQ.send(socket, JSON.json(err_response))
                    println("Error response sent")
                catch send_error
                    println("Failed to send error response: $send_error")
                    
                    # Reset socket state if we can't send a response
                    # This is required to maintain the REQ-REP pattern
                    try
                        # Close and recreate socket
                        println("Recreating socket due to broken REQ-REP pattern...")
                        ZMQ.close(socket)
                        socket = Socket(context, REP)
                        
                        # Re-apply socket options
                        ZMQ.setsockopt(socket, ZMQ.RCVTIMEO, RECV_TIMEOUT)
                        ZMQ.setsockopt(socket, ZMQ.SNDTIMEO, SEND_TIMEOUT)
                        ZMQ.setsockopt(socket, ZMQ.RCVHWM, HIGH_WATER_MARK)
                        ZMQ.setsockopt(socket, ZMQ.SNDHWM, HIGH_WATER_MARK)
                        ZMQ.setsockopt(socket, ZMQ.RCVBUF, SOCKET_BUFFER_SIZE)
                        ZMQ.setsockopt(socket, ZMQ.SNDBUF, SOCKET_BUFFER_SIZE)
                        ZMQ.setsockopt(socket, ZMQ.TCP_KEEPALIVE, 1)
                        
                        # Rebind
                        ZMQ.bind(socket, SOCKET_ADDRESS)
                        println("Socket recreated and rebound")
                    catch rebind_error
                        println("Failed to recreate socket: $rebind_error")
                        rethrow()  # This will exit the server loop
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
    end
end

# Run the server
println("Starting ZMQ server...")
run_server()