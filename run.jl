#!/usr/bin/env julia

# Main entry point for AirSim + RxInfer.jl project with Active Inference
# Streamlined architecture focusing on belief updating and free energy minimization

using Pkg
Pkg.activate(@__DIR__)  # Use the local project environment

# Import modules - consolidated to remove redundancies
include("src/StateSpace.jl")     # Consolidated state representation
include("src/Inference.jl")      # Consolidated belief updating and inference
include("src/Planning.jl")       # Action planning with EFE
include("src/AirSimInterface.jl") # AirSim interface via Python

using .StateSpace
using .Inference
using .Planning
using .AirSimInterface
using StaticArrays
using LinearAlgebra

# Main Active Inference loop
function main(; target_position=[10.0, 10.0, -5.0], 
              max_iterations=1000, 
              control_interval=0.2, 
              stop_distance=0.5)
    
    println("Starting AirSim Drone Navigation with Active Inference")
    println("Target position: $target_position")
    
    # Convert target to SVector
    target = SVector{3, Float64}(target_position)
    
    # Initialize components
    
    # 1. Initialize AirSim interface
    println("Initializing AirSim...")
    airsim_client = AirSimInterface.initialize_airsim()
    
    # 2. Initialize planner with active inference parameters
    planner = Planning.ActionPlanner(
        max_step_size = 0.5,
        num_angles = 8,
        num_step_sizes = 3,
        pragmatic_weight = 1.0,    # Target-seeking behavior weight
        epistemic_weight = 0.2,    # Exploration weight
        risk_weight = 2.0,         # Obstacle avoidance weight
        safety_distance = 1.5
    )
    
    # 3. Initialize beliefs with uniform priors
    beliefs = Inference.initialize_beliefs()
    
    # Main active inference loop
    arrived = false
    iteration = 0
    
    try
        while !arrived && iteration < max_iterations
            iteration += 1
            println("\n==== Iteration $iteration ====")
            
            # 1. PERCEPTION: Get sensor data and create observation
            observation = AirSimInterface.get_sensor_data(airsim_client, target)
            
            # 2. STATE ESTIMATION: Create state from observation
            current_state = StateSpace.create_state_from_observation(observation)
            
            # 3. BELIEF UPDATING: Update beliefs using VFE minimization
            beliefs = Inference.update_beliefs_vfe(beliefs, observation)
            
            # 4. Check if we've reached the target
            distance_to_target = current_state.raw_distance
            println("Distance to target: $(round(distance_to_target, digits=2)) m")
            
            if distance_to_target < stop_distance
                arrived = true
                println("Target reached!")
                break
            end
            
            # 5. ACTION SELECTION: Plan next action by minimizing expected free energy
            action, efe = Planning.select_action(current_state, beliefs, planner)
            println("Selected action: $(round.(action, digits=2)), EFE: $(round(efe, digits=2))")
            
            # 6. ACTION EXECUTION: Send action to AirSim and execute
            new_position = AirSimInterface.send_action(airsim_client, action)
            println("New position: $(round.(new_position, digits=2))")
            
            # Sleep for control interval
            sleep(control_interval)
        end
        
        if arrived
            println("\nMission successful! Target reached in $iteration iterations.")
        else
            println("\nFailed to reach target in $max_iterations iterations.")
        end
        
    catch e
        println("Error in active inference loop: $e")
    end
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
