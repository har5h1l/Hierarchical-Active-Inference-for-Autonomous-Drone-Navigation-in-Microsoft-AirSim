#!/usr/bin/env julia

println("Starting drone navigation planning script...")

# Check if we're running in a precompiled environment - fixed to avoid circular dependency
const IS_PRECOMPILED = haskey(ENV, "JULIA_ACTINF_PRECOMPILED")

# Only activate if environment is not already active
if Base.active_project() != abspath(joinpath(@__DIR__, "Project.toml"))
    println("Activating project environment...")
    import Pkg
    Pkg.activate(@__DIR__)
    println("Project activated")
else
    println("Project environment already active")
end

# Import required packages with explicit error handling
if !IS_PRECOMPILED
    println("Loading required packages...")
    function load_package(pkgname)
        try
            println("Loading $pkgname...")
            @eval import $pkgname
            println("Successfully loaded $pkgname")
            return true
        catch e
            println("Error loading $pkgname: $e")
            return false
        end
    end

    # Load standard libraries first
    load_package(:JSON)
    load_package(:LinearAlgebra)
    load_package(:StaticArrays)

    # Add direct imports for functions to make them available in Main scope
    using StaticArrays: SVector
    using LinearAlgebra: norm

    # Load actinf with special handling - using include if necessary
    println("Loading actinf package...")
    try
        # Try direct loading first
        @eval using actinf
        println("Successfully loaded actinf package")
    catch e
        println("Could not load actinf directly: $e")
        
        # Try to include the module files manually
        try
            println("Attempting to manually include actinf module files...")
            
            # Include module files directly
            include(joinpath(@__DIR__, "actinf", "src", "StateSpace.jl"))
            include(joinpath(@__DIR__, "actinf", "src", "Inference.jl"))
            include(joinpath(@__DIR__, "actinf", "src", "Planning.jl"))
            include(joinpath(@__DIR__, "actinf", "src", "actinf.jl"))
            
            # Import the manually included module
            using .actinf
            println("Successfully loaded actinf via manual inclusion")
        catch e2
            println("Failed to manually include actinf: $e2")
            error("Cannot continue without actinf package")
        end
    end
else
    # In precompiled mode, just ensure the necessary modules are available
    println("Using precompiled dependencies")
    using JSON, LinearAlgebra, StaticArrays
    using StaticArrays: SVector
    using actinf
end

# Import specific components from actinf to make them available in Main scope
using actinf.StateSpace: DroneState, DroneObservation, create_state_from_observation
using actinf.Inference: deserialize_beliefs, initialize_beliefs
using actinf.Planning: ActionPlanner, PreferenceModel, select_action

# Constants and parameters with more robust path handling for cross-platform compatibility
const INTERFACE_DIR = joinpath(@__DIR__, "interface")
const INFERRED_STATE_PATH = joinpath(INTERFACE_DIR, "inferred_state.json")
const ACTION_OUTPUT_PATH = joinpath(INTERFACE_DIR, "action_output.json")
const NEXT_WAYPOINT_PATH = joinpath(INTERFACE_DIR, "next_waypoint.json") # For backward compatibility
const OBS_INPUT_PATH = joinpath(INTERFACE_DIR, "obs_input.json")

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
    
    # Test write access by touching the file - safer approach
    temp_file = joinpath(INTERFACE_DIR, "test_write_access.tmp")
    touch(temp_file)
    if isfile(temp_file)
        rm(temp_file)  # Clean up the test file if successful
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
        data = open(INFERRED_STATE_PATH) do f
            JSON.parse(f)
        end
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
    
    # Get obstacle repulsion weight from the original observation file if available
    local obstacle_repulsion_weight = 0.0
    local direct_path_clear = true
    local direct_path_suitability = 1.0
    
    if isfile(OBS_INPUT_PATH)
        try
            obs_data = open(OBS_INPUT_PATH) do f
                JSON.parse(f)
            end
            obstacle_repulsion_weight = get(obs_data, "obstacle_repulsion_weight", 0.0)
            direct_path_clear = get(obs_data, "direct_path_clear", true)
            direct_path_suitability = get(obs_data, "direct_path_suitability", 1.0)
            
            println("Retrieved obstacle repulsion weight: $obstacle_repulsion_weight")
            println("Direct path clear: $direct_path_clear, suitability: $direct_path_suitability")
        catch e
            println("Warning: Could not read observation file: $e")
        end
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
    catch e
        println("Error parsing drone position: $e")
        SVector{3, Float64}(0.0, 0.0, 0.0)
    end
    
    target_position = try
        SVector{3, Float64}(get(data, "target_position", [10.0, 0.0, -3.0])...)
    catch e
        println("Error parsing target position: $e")
        SVector{3, Float64}(10.0, 0.0, -3.0)
    end
    
    nearest_obstacle_distances = try
        Float64.(get(data, "nearest_obstacle_distances", [100.0, 100.0]))
    catch e
        println("Error parsing obstacle distances: $e")
        Float64[100.0, 100.0]
    end
    
    obstacle_density = try
        Float64(get(data, "obstacle_density", 0.0))
    catch e
        println("Error parsing obstacle density: $e")
        0.0
    end
    
    # Calc distance to target
    current_to_target = target_position - drone_position
    current_to_target_dist = norm(current_to_target)
    
    # Print crucial information for debugging
    println("\nCurrent position: [$(round(drone_position[1], digits=2)), $(round(drone_position[2], digits=2)), $(round(drone_position[3], digits=2))]")
    println("Target position: [$(round(target_position[1], digits=2)), $(round(target_position[2], digits=2)), $(round(target_position[3], digits=2))]")
    println("Distance to target: $(round(current_to_target_dist, digits=2))")
    
    # Calculate target preference vs obstacle avoidance weights
    # More target preference when:
    # 1. Clear direct path to target
    # 2. Very close to target
    # 3. Low obstacle repulsion
    
    target_preference_weight = 0.6  # Base weight
    
    # Adjust based on distance to target
    if current_to_target_dist < 5.0
        target_preference_weight += 0.2  # Strong target preference when very close
    elseif current_to_target_dist < 10.0
        target_preference_weight += 0.1  # Moderate increase when somewhat close
    end
    
    # Adjust based on direct path clarity
    if direct_path_clear && direct_path_suitability > 0.7
        target_preference_weight += 0.1  # Increase when direct path is very clear
    end
    
    # Reduce for high obstacle repulsion
    if obstacle_repulsion_weight > 0.5
        target_preference_weight = max(0.4, target_preference_weight - 0.2)  # Reduce but maintain minimum
    elseif obstacle_repulsion_weight > 0.2
        target_preference_weight = max(0.4, target_preference_weight - 0.1)  # Smaller reduction
    end
    
    # Cap the weight
    target_preference_weight = clamp(target_preference_weight, 0.4, 0.8)
    obstacle_weight = 1.0 - target_preference_weight
    
    # Update suitability threshold based on target distance
    suitability_threshold = if current_to_target_dist < 5.0
        0.3  # Lower threshold when very close to target
    elseif current_to_target_dist < 10.0
        0.4  # Moderate threshold when somewhat close
    else
        0.5  # Standard threshold when far from target
    end
    
    # Calculate density weight based on distance and obstacle repulsion
    density_weight = 1.0
    if obstacle_repulsion_weight > 0.5
        density_weight = 1.5  # Higher density weight with high repulsion
    end
    
    # Initialize preference model and action planner
    preference_model = PreferenceModel(
        distance_weight = target_preference_weight,
        distance_scaling = 0.1,
        angle_weight = 0.8,
        angle_sharpness = 5.0,
        suitability_weight = obstacle_weight,
        suitability_threshold = suitability_threshold,
        max_distance = 50.0
    )
    
    # Risk weight is scaled up based on obstacle repulsion
    risk_weight_scale = 2.0 + (obstacle_repulsion_weight * 3.0)
    
    planner = ActionPlanner(
        max_step_size = 1.0,
        num_angles = 8,
        num_step_sizes = 3,
        pragmatic_weight = target_preference_weight > 0.5 ? 1.5 : 1.0,  # Increased when target preference is high
        epistemic_weight = 0.2,
        risk_weight = risk_weight_scale,  # Scale up risk weight based on obstacle repulsion
        safety_distance = MARGIN,
        density_weight = density_weight,
        preference_model = preference_model
    )
    
    # Override the suitability threshold used in planning
    # This ensures that the filtering step in select_action uses the updated threshold
    global actinf.Planning.SUITABILITY_THRESHOLD = suitability_threshold
    
    # Select best actions using the planner
    obstacle_distance = isempty(nearest_obstacle_distances) ? 100.0 : minimum(nearest_obstacle_distances)
    
    # Get nearest obstacle distance, maybe from an additional field if available
    if haskey(data, "nearest_obstacle_dist")
        obstacle_distance = data["nearest_obstacle_dist"]
    end
    
    println("Planning with parameters:")
    println("- Target preference weight: $(round(target_preference_weight, digits=2))")
    println("- Obstacle weight: $(round(obstacle_weight, digits=2))")
    println("- Suitability threshold: $(round(suitability_threshold, digits=2))")
    println("- Obstacle distance: $(round(obstacle_distance, digits=2))")
    println("- Obstacle density: $(round(obstacle_density, digits=2))")
    
    actions_with_efe = select_action(
        current_state,
        beliefs,
        planner,
        drone_position,
        target_position,
        obstacle_distance=obstacle_distance,
        obstacle_density=obstacle_density,
        num_policies=POLICY_LENGTH,
        obstacle_weight=obstacle_weight
    )
    
    # Extract the top action and policy
    best_action, best_efe = actions_with_efe[1]
    policy = [action for (action, _) in actions_with_efe[1:min(POLICY_LENGTH, length(actions_with_efe))]]
    
    # Convert SVector to regular arrays for JSON serialization
    best_action_array = [best_action[1], best_action[2], best_action[3]]
    policy_array = [[a[1], a[2], a[3]] for a in policy]
    
    # Calculate next waypoint based on current position and action
    next_waypoint = [
        drone_position[1] + best_action[1],
        drone_position[2] + best_action[2], 
        drone_position[3] + best_action[3]
    ]
    
    println("Selected next waypoint: [$(round(next_waypoint[1], digits=2)), $(round(next_waypoint[2], digits=2)), $(round(next_waypoint[3], digits=2))]")
    println("Action vector: [$(round(best_action[1], digits=2)), $(round(best_action[2], digits=2)), $(round(best_action[3], digits=2))]")
    
    # Create action output data
    action_data = Dict(
        "action" => best_action_array,
        "next_waypoint" => next_waypoint,
        "waypoint" => next_waypoint,  # Alternative key for compatibility
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
            JSON.print(f, action_data)
        end
        
        # Also save to next_waypoint.json for backward compatibility
        open(NEXT_WAYPOINT_PATH, "w") do f
            JSON.print(f, action_data)
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
            JSON.print(file, action_data)
            close(file)
            println("Alternative write approach completed")
        catch e2
            println("Fatal error writing to file: $e2")
            # Try one last approach with a different filename
            fallback_path = joinpath(dirname(@__FILE__), "action_output_fallback.json")
            println("Trying last resort to $fallback_path")
            open(fallback_path, "w") do f
                JSON.print(f, action_data)
            end
        end
    end
    
    println("Planning complete. Selected waypoint: $(next_waypoint)")
end

# Run the main function with better error handling
try
    main()
    println("\nPlanning script completed successfully")
catch e
    println("\n❌ Error in planning script main function: $e")
    bt = backtrace()
    println("Stack trace:")
    for (i, frame) in enumerate(bt)
        if i > 10  # Limit stack trace to first 10 frames
            println("...")
            break
        end
        try
            frame_info = Base.StackTraces.lookup(frame)
            println("  $(frame_info[1].func) at $(frame_info[1].file):$(frame_info[1].line)")
        catch
            println("  [Frame $i]")
        end
    end
    # Re-throw for proper exit code
    rethrow(e)
end