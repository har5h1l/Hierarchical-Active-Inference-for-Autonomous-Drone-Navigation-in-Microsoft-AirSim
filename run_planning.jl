#!/usr/bin/env julia

# Activate the project environment if needed
if Base.active_project() != abspath(joinpath(@__DIR__, "Project.toml"))
    import Pkg
    Pkg.activate(@__DIR__)
    # No need for develop and instantiate as they're done during precompilation
end

# Import required modules - already precompiled
using JSON
using LinearAlgebra
using StaticArrays
using actinf

# Constants and parameters with more robust path handling for cross-platform compatibility
const INTERFACE_DIR = joinpath(@__DIR__, "interface")
const INFERRED_STATE_PATH = joinpath(INTERFACE_DIR, "inferred_state.json")
const ACTION_OUTPUT_PATH = joinpath(INTERFACE_DIR, "action_output.json")

# Hyperparameters
const MARGIN = 1.5  # Safety margin for waypoint generation (meters)
const WAYPOINT_SAMPLE_COUNT = 75  # Number of waypoints to consider
const POLICY_LENGTH = 3  # Length of policy (number of steps to look ahead)

# Make sure interface directory exists with better error handling
try
    if !isdir(INTERFACE_DIR)
        mkpath(INTERFACE_DIR)
        println("Created interface directory: $INTERFACE_DIR")
    end
catch e
    println("Warning: Issue with interface directory: $e")
    # Don't error out completely, we'll try again when writing
end

function main()
    println("Starting planning pipeline...")

    # Check if input file exists with better error reporting
    if !isfile(INFERRED_STATE_PATH)
        println("ERROR: Input file not found: $INFERRED_STATE_PATH")
        println("Current directory: $(pwd())")
        println("Directory contents: $(readdir())")
        if isdir(dirname(INFERRED_STATE_PATH))
            println("Interface directory exists, contents: $(readdir(dirname(INFERRED_STATE_PATH)))")
        else
            println("Interface directory does not exist!")
        end
        error("Input file $INFERRED_STATE_PATH not found.")
    end

    # Read inferred state and beliefs from JSON with better error handling
    println("Reading inferred state from $INFERRED_STATE_PATH...")
    local data
    try
        data = JSON.parsefile(INFERRED_STATE_PATH)
        println("Successfully parsed inferred state file")
    catch e
        println("Error parsing inferred state file: $e")
        # Create a minimal default state for emergency fallback
        data = Dict(
            "state" => Dict("distance" => 10.0, "azimuth" => 0.0, "elevation" => 0.0, "suitability" => 0.5),
            "drone_position" => [0.0, 0.0, 0.0],
            "target_position" => [10.0, 0.0, -3.0],
            "nearest_obstacle_distances" => [100.0, 100.0],
            "obstacle_density" => 0.0,
            "beliefs" => Dict()  # Empty beliefs will cause initialization of new ones
        )
        println("Using emergency fallback state")
    end
    
    # Extract state, beliefs, and positions with more robust error handling
    local current_state
    try
        state_dict = get(data, "state", Dict("distance" => 10.0, "azimuth" => 0.0, 
                                            "elevation" => 0.0, "suitability" => 0.5))
        current_state = DroneState(
            distance = get(state_dict, "distance", 10.0),
            azimuth = get(state_dict, "azimuth", 0.0),
            elevation = get(state_dict, "elevation", 0.0),
            suitability = get(state_dict, "suitability", 0.5)
        )
    catch e
        println("Error extracting state data: $e")
        current_state = DroneState()  # Use default values
    end
    
    # Get beliefs including voxel grid with error handling
    local beliefs
    try
        beliefs_data = get(data, "beliefs", Dict())
        if !isempty(beliefs_data)
            beliefs = deserialize_beliefs(beliefs_data)
            println("Successfully loaded beliefs from file")
        else
            println("No beliefs data found, initializing new beliefs")
            beliefs = initialize_beliefs(current_state)
        end
    catch e
        println("Error loading beliefs: $e")
        println("Initializing new beliefs")
        beliefs = initialize_beliefs(current_state)
    end
    
    # Extract other data safely
    drone_position = try
        SVector{3, Float64}(get(data, "drone_position", [0.0, 0.0, 0.0])...)
    catch
        SVector{3, Float64}(0.0, 0.0, 0.0)
    end
    
    target_position = try
        SVector{3, Float64}(get(data, "target_position", [10.0, 0.0, -3.0])...)
    catch
        SVector{3, Float64}(10.0, 0.0, -3.0)
    end
    
    nearest_obstacle_distances = get(data, "nearest_obstacle_distances", [100.0, 100.0])
    obstacle_density = get(data, "obstacle_density", 0.0)
    
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
        safety_distance = MARGIN,
        density_weight = 1.0,
        preference_model = preference_model
    )
    
    # Select best actions using the planner
    obstacle_distance = isempty(nearest_obstacle_distances) ? 100.0 : minimum(nearest_obstacle_distances)
    actions_with_efe = select_action(
        current_state,
        beliefs,
        planner,
        drone_position,
        target_position,
        obstacle_distance=obstacle_distance,
        obstacle_density=obstacle_density,
        num_policies=POLICY_LENGTH
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
    
    # Save to file with robust error handling
    try
        # Make sure interface directory exists right before writing
        if !isdir(INTERFACE_DIR)
            mkpath(INTERFACE_DIR)
            println("Created interface directory before writing: $INTERFACE_DIR")
        end
        
        println("Saving action to $ACTION_OUTPUT_PATH...")
        open(ACTION_OUTPUT_PATH, "w") do f
            JSON.print(f, action_data, 2)  # Pretty print with 2-space indent
        end
        
        # Verify the file was written
        if isfile(ACTION_OUTPUT_PATH)
            println("Successfully wrote action file")
        else
            println("Warning: File writing seemed to succeed but file doesn't exist")
        end
    catch e
        println("Error during first write attempt: $e")
        try
            # Try a different approach as fallback
            println("Retrying write with alternative approach...")
            file = open(ACTION_OUTPUT_PATH, "w")
            JSON.print(file, action_data, 2)
            close(file)
            println("Alternative write approach completed")
        catch e2
            println("Fatal error writing to file: $e2")
            # Try one last approach with a different filename
            fallback_path = joinpath(dirname(@__FILE__), "action_output_fallback.json")
            println("Trying last resort to $fallback_path")
            open(fallback_path, "w") do f
                JSON.print(f, action_data, 2)
            end
        end
    end
    
    println("Planning complete. Selected waypoint: $(best_action_array)")
end

# Run the main function
main()