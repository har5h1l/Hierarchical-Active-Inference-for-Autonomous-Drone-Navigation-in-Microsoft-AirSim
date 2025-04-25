#!/usr/bin/env julia

# Add the current directory to the load path
push!(LOAD_PATH, @__DIR__)

# Import required modules
using JSON
using LinearAlgebra
using StaticArrays
import actinf.StateSpace
import actinf.Inference
import actinf.Planning

# Constants and parameters
const INTERFACE_DIR = "./interface"
const INFERRED_STATE_PATH = joinpath(INTERFACE_DIR, "inferred_state.json")
const ACTION_OUTPUT_PATH = joinpath(INTERFACE_DIR, "action_output.json")

# Hyperparameters
const MARGIN = 1.5  # Safety margin for waypoint generation (meters)
const WAYPOINT_SAMPLE_COUNT = 75  # Number of waypoints to consider
const POLICY_LENGTH = 3  # Length of policy (number of steps to look ahead)

# Make sure interface directory exists
mkpath(INTERFACE_DIR)

function main()
    println("Starting planning pipeline...")

    # Check if input file exists
    if !isfile(INFERRED_STATE_PATH)
        error("Input file $INFERRED_STATE_PATH not found.")
    end

    # Read inferred state and beliefs from JSON
    println("Reading inferred state from $INFERRED_STATE_PATH...")
    data = JSON.parsefile(INFERRED_STATE_PATH)
    
    # Extract state, beliefs, and positions
    state_dict = data["state"]
    current_state = StateSpace.DroneState(
        distance = state_dict["distance"],
        azimuth = state_dict["azimuth"],
        elevation = state_dict["elevation"],
        suitability = state_dict["suitability"],
        obstacle_density = get(state_dict, "obstacle_density", 0.0)
    )
    
    # Get beliefs including voxel grid
    beliefs = Inference.deserialize_beliefs(data["beliefs"])
    
    # Extract other data
    drone_position = get(data, "drone_position", [0.0, 0.0, 0.0])
    target_position = get(data, "target_position", [10.0, 0.0, -3.0])
    nearest_obstacle_distances = get(data, "nearest_obstacle_distances", [100.0, 100.0])
    obstacle_density = get(data, "obstacle_density", 0.0)
    
    # Convert to SVector
    drone_pos = SVector{3, Float64}(drone_position...)
    target_pos = SVector{3, Float64}(target_position...)
    
    # Get the nearest obstacle distances from the inferred state
    # This is used to determine the safe radius for waypoint generation
    min_obstacle_distance = minimum(nearest_obstacle_distances)
    
    # Create a preference model tuned for our environment
    preference_model = Planning.PreferenceModel(
        distance_weight = 1.2,
        distance_scaling = 0.1,
        angle_weight = 0.9,
        angle_sharpness = 4.0,
        suitability_weight = 2.0,  # Higher emphasis on safety
        suitability_threshold = 0.4,  # Minimum acceptable suitability
        max_distance = 50.0
    )
    
    # Create a planner with appropriate parameters
    planner = Planning.ActionPlanner(
        max_step_size = min(min_obstacle_distance / 2 + MARGIN, 2.0),  # Limit to 2m max
        num_angles = 12,  # More angles for finer control
        num_step_sizes = 3,
        pragmatic_weight = 1.0,
        epistemic_weight = 0.3,
        risk_weight = 2.0,
        safety_distance = MARGIN,
        density_weight = 1.5,  # Emphasize density consideration
        preference_model = preference_model
    )
    
    println("Selecting best action using voxel data ($(length(beliefs.voxel_grid)) voxels)...")
    # Generate and select the best actions
    actions_with_efe = Planning.select_action(
        current_state, 
        beliefs, 
        planner, 
        drone_pos, 
        target_pos,
        obstacle_distance = min_obstacle_distance,
        obstacle_density = obstacle_density,
        num_policies = POLICY_LENGTH
    )
    
    # Extract the top action and policy
    best_action, best_efe = actions_with_efe[1]
    policy = [action for (action, _) in actions_with_efe[1:min(POLICY_LENGTH, length(actions_with_efe))]]
    
    # Convert SVector to regular arrays for JSON serialization
    best_action_array = [best_action[1], best_action[2], best_action[3]]
    policy_array = [[a[1], a[2], a[3]] for a in policy]
    
    # Create action output data
    action_data = Dict(
        "next_waypoint" => best_action_array,
        "policy" => policy_array,
        "expected_free_energy" => best_efe
    )
    
    # Save to file
    println("Saving action to $ACTION_OUTPUT_PATH...")
    open(ACTION_OUTPUT_PATH, "w") do f
        JSON.print(f, action_data, 2)  # Pretty print with 2-space indent
    end
    
    println("Planning complete. Selected waypoint: $(best_action_array)")
end

# Run the main function
main()