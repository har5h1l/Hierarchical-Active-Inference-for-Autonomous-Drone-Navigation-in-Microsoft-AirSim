#!/usr/bin/env julia

# Main entry point for AirSim + RxInfer.jl project
# This ties together all components for drone navigation

using Pkg
Pkg.activate(@__DIR__)  # Use the local project environment

# Import modules
include("src/Sensors.jl")
include("src/Fusion.jl")
include("src/State.jl")
include("src/Inference.jl")
include("src/Planning.jl")

using .Sensors
using .Fusion
using .State
using .Inference
using .Planning
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
    
    println("Starting AirSim + RxInfer Drone Navigation")
    println("Target position: $target_position")
    
    # Initialize components
    planner = Planning.ActionPlanner(
        distance_weight = 1.0,
        obstacle_weight = 2.0,
        safety_distance = 1.5,
        max_step_size = 0.5,
        lookahead_steps = 5
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
    
    # Create initial state
    initial_state = State.extract_state(
        point_cloud, 
        current_position, 
        target_position
    )
    
    # Initialize beliefs
    beliefs = Inference.initialize_beliefs(initial_state)
    
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
            # In practice, we would get drone position from AirSim client
            # Here we just update it using the action taken in the previous iteration
            current_state = State.extract_state(
                voxelized_points, 
                current_position, 
                target_position
            )
            
            # Check if we're close enough to target to consider arrived
            distance_to_target = current_state.distance_to_target
            if distance_to_target < stop_distance
                arrived = true
                println("Target reached! Distance: $distance_to_target")
                break
            end
            
            println("Iteration $iteration: Distance to target: $distance_to_target")
            
            # Plan next action
            action = Planning.plan_next_action(planner, current_state, beliefs)
            
            # Update beliefs using the state and action
            beliefs = Inference.update_beliefs(beliefs, current_state, action)
            
            # Simulate execution of the action (update position)
            # In practice, this would be handled by the AirSim controller
            current_position = current_position + action
            
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
