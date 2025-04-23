#!/usr/bin/env julia

# Main entry point for AirSim + RxInfer.jl project
# This ties together all components for drone navigation using active inference

using Pkg
Pkg.activate(@__DIR__)  # Use the local project environment

# Import modules
include("src/Sensors.jl")
include("src/Fusion.jl")
include("src/State.jl")
include("src/Inference.jl")
include("src/Planning.jl")
include("src/InferenceModel.jl")  # New dedicated inference model

using .Sensors
using .Fusion
using .State
using .Inference
using .Planning
using .InferenceModel  # Import the new module
using PyCall
using LinearAlgebra

# Function to run the AirSim drone control Python script
function run_airsim_control(x, y, z; control_script="scripts/drone_control.py")
    script_path = joinpath(@__DIR__, control_script)
    command = `python $script_path --x $x --y $y --z $z`
    
    # Run the script in the background
    process = run(pipeline(command, stdout=devnull, stderr=devnull), wait=false)
    
    return process
end

# Main control loop
function main(; target_position=[10.0, 10.0, -5.0], 
              max_iterations=1000, 
              control_interval=0.2, 
              stop_distance=0.5)
    
    println("Starting AirSim + RxInfer Drone Navigation with Active Inference")
    println("Target position: $target_position")
    
    # Initialize planner with active inference parameters
    planner = Planning.ActionPlanner(
        pragmatic_weight = 1.0,    # Target-seeking behavior weight
        risk_weight = 2.0,         # Obstacle avoidance weight
        epistemic_weight = 0.2,    # Exploration weight
        safety_distance = 1.5,
        max_step_size = 0.5,
        lookahead_steps = 5,
        num_directions = 12,
        num_step_sizes = 3
    )
    
    # Initialize drone position
    current_position = [0.0, 0.0, 0.0]
    
    # Create data directories
    mkpath(joinpath(@__DIR__, "data", "camera"))
    mkpath(joinpath(@__DIR__, "data", "lidar"))
    
    # Start the AirSim control process
    control_process = run_airsim_control(target_position[1], target_position[2], target_position[3])
    
    # Wait for initial sensor data
    println("Waiting for initial sensor data...")
    while Sensors.get_latest_data_timestamp() === nothing
        sleep(0.5)
    end
    
    # Initial sensor reading
    rgb_img, depth_img, lidar_points, timestamp = Sensors.read_sensor_data()
    
    if isnothing(rgb_img) || isnothing(depth_img) || isnothing(lidar_points)
        error("Failed to get initial sensor data")
    end
    
    println("Sensor data received, starting navigation")
    
    # Process initial data to create point cloud
    point_cloud, _ = Fusion.fuse_sensor_data(rgb_img, depth_img, lidar_points)
    
    # Create initial state with voxel grid
    initial_state = State.extract_state(
        point_cloud, 
        current_position, 
        target_position,
        voxel_size = 0.5
    )
    
    # Initialize beliefs using the InferenceModel
    beliefs = nothing  # Will be initialized in the first loop iteration
    
    # Main control loop
    arrived = false
    iteration = 0
    
    try
        while !arrived && iteration < max_iterations
            iteration += 1
            
            # Read the latest sensor data
            rgb_img, depth_img, lidar_points, timestamp = Sensors.read_sensor_data()
            
            if isnothing(rgb_img) || isnothing(depth_img) || isnothing(lidar_points)
                println("Warning: Missing sensor data, skipping iteration")
                sleep(control_interval)
                continue
            end
            
            # Process data to create and voxelize point cloud
            point_cloud, colors = Fusion.fuse_sensor_data(rgb_img, depth_img, lidar_points)
            if !isnothing(point_cloud)
                voxelized_points, _ = Fusion.voxelize_point_cloud(point_cloud, colors)
            else
                voxelized_points = nothing
            end
            
            # Update state based on latest data
            current_state = State.extract_state(
                voxelized_points, 
                current_position, 
                target_position,
                voxel_size = 0.5
            )
            
            # Check if we're close enough to target to consider arrived
            distance_to_target = current_state.distance_to_target
            if distance_to_target < stop_distance
                arrived = true
                println("Target reached! Distance: $distance_to_target")
                break
            end
            
            println("Iteration $iteration: Distance to target: $distance_to_target")
            
            # Update beliefs using the new InferenceModel
            # First time initialize, then update
            if isnothing(beliefs)
                beliefs = InferenceModel.update_belief_states(current_state)
            else
                beliefs = InferenceModel.update_belief_states(current_state, beliefs)
            end
            
            # Plan next action using active inference
            # Generate potential actions
            potential_actions = Planning.generate_potential_actions(planner, current_state)
            
            # Find the action with the minimum expected free energy
            best_action = nothing
            best_efe = Inf
            
            for action in potential_actions
                # Calculate EFE for this action using the InferenceModel
                efe = InferenceModel.calculate_efe(
                    current_state, 
                    beliefs, 
                    action,
                    pragmatic_weight = planner.params.pragmatic_weight,
                    risk_weight = planner.params.risk_weight,
                    epistemic_weight = planner.params.epistemic_weight
                )
                
                if efe < best_efe
                    best_efe = efe
                    best_action = action
                end
            end
            
            # Use safe default if no good action found
            if isnothing(best_action)
                direction = current_state.direction_to_target
                best_action = direction * min(planner.params.max_step_size / 2, 0.1)
            end
            
            # Simulate execution of the action (update position)
            # In practice, this would be handled by the AirSim controller
            current_position = current_position + best_action
            
            println("  Action: $(round.(best_action, digits=2)), EFE: $(round(best_efe, digits=2))")
            
            # Sleep for control interval
            sleep(control_interval)
        end
        
        if arrived
            println("Successfully reached target!")
        else
            println("Failed to reach target in $max_iterations iterations")
        end
        
    catch e
        println("Error in main loop: $e")
    finally
        # Clean up: stop AirSim control process
        if isrunning(control_process)
            kill(control_process)
        end
    end
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
