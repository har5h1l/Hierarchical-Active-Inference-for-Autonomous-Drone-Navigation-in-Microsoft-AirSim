#!/usr/bin/env julia

println("Starting drone navigation planning script...")

# Check if we've been precompiled already
const IS_PRECOMPILED = haskey(ENV, "JULIA_ACTINF_PRECOMPILED")

# Only activate if environment is not already active
if !IS_PRECOMPILED
    println("Not precompiled. Activating project environment...")
    import Pkg
    Pkg.activate(@__DIR__)
    println("Project activated")
else
    println("Using precompiled environment")
end

# Import packages without reloading if already loaded
function use_package(pkg_name, force_load=false)
    if !isdefined(Main, Symbol(pkg_name)) || force_load
        println("Loading $pkg_name...")
        @eval import $pkg_name
        println("Successfully loaded $pkg_name")
    else
        println("Using already loaded package: $pkg_name")
    end
end

# Core packages - only load if not already loaded
use_package(:JSON)
use_package(:LinearAlgebra)
use_package(:StaticArrays)

# Make StaticArrays available in main scope if needed
if !isdefined(Main, :SVector)
    @eval using StaticArrays: SVector
    println("Imported SVector into scope")
else
    println("SVector already in scope")
end

# Load actinf package if not already loaded
if !isdefined(Main, :actinf) 
    println("Loading actinf package...")
    try
        @eval using actinf
        println("Successfully loaded actinf package")
    catch e
        println("Error loading actinf package: $e")
        error("Cannot continue without actinf package")
    end
else
    println("Using already loaded actinf package")
end

# Import specific components from actinf to make them available in Main scope
using actinf.StateSpace: DroneState, DroneObservation, create_state_from_observation
using actinf.Inference: deserialize_beliefs, initialize_beliefs
using actinf.Planning: ActionPlanner, PreferenceModel, select_action

# Constants and parameters with more robust path handling for cross-platform compatibility
const INTERFACE_DIR = joinpath(@__DIR__, "interface")
const INFERRED_STATE_PATH = joinpath(INTERFACE_DIR, "inferred_state.json")
const NEXT_WAYPOINT_PATH = joinpath(INTERFACE_DIR, "next_waypoint.json") 
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
    
    # Test write access by touching the file
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
    
    # Calculate distance to target
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
        # When close to target, favor direct approach
        target_preference_weight = 0.8
    elseif current_to_target_dist > 20.0
        # When far from target, more exploration is reasonable
        target_preference_weight = 0.5
    end
    
    # Adjust based on obstacle factors
    if direct_path_clear && direct_path_suitability > 0.8
        # Clear path, strongly prefer direct route
        target_preference_weight += 0.15
    elseif !direct_path_clear || direct_path_suitability < 0.5
        # Blocked path, need more obstacle avoidance
        target_preference_weight -= 0.15
    end
    
    # Factor in explicit obstacle repulsion setting
    target_preference_weight -= obstacle_repulsion_weight * 0.05
    
    # Clamp to reasonable range
    target_preference_weight = max(0.3, min(0.9, target_preference_weight))
    
    println("Target preference weight: $(round(target_preference_weight, digits=2))")
    println("Obstacle avoidance weight: $(round(1.0 - target_preference_weight, digits=2))")
    
    # Create planner with preference model
    preference_model = PreferenceModel(
        distance_preference = target_preference_weight,
        path_preference = 1.0 - target_preference_weight  # Balance between these two primary preferences
    )
    
    planner = ActionPlanner(preference_model)
    
    # Select the next waypoint using active inference
    println("\nSelecting next waypoint...")
    next_waypoint = select_action(
        planner, 
        beliefs, 
        num_samples=WAYPOINT_SAMPLE_COUNT, 
        safety_margin=MARGIN,
        policy_length=POLICY_LENGTH,
        density_radius=get(data, "density_radius", 5.0)
    )
    
    # Calculate waypoint in global coordinates
    global_waypoint = try
        if length(next_waypoint) >= 3
            next_waypoint
        else
            # If we didn't get a valid waypoint, move slightly towards target
            normalized_to_target = current_to_target / norm(current_to_target)
            safe_step = normalized_to_target * 2.0  # 2-meter step
            drone_position + safe_step
        end
    catch e
        println("Error calculating global waypoint: $e")
        # Emergency fallback - small step towards target
        normalized_to_target = current_to_target / norm(current_to_target)
        safe_step = normalized_to_target * 1.0  # 1-meter step
        drone_position + safe_step
    end
    
    # Print the selected waypoint
    println("Selected next waypoint: [$(round(global_waypoint[1], digits=2)), $(round(global_waypoint[2], digits=2)), $(round(global_waypoint[3], digits=2))]")
    
    # Generate the policy for future steps (simplified for this implementation)
    policy = []
    for i in 1:min(POLICY_LENGTH, 3)  # Cap at 3 steps for simplicity
        # Create simplified future waypoints along same vector
        waypoint_vector = global_waypoint - drone_position
        norm_vector = waypoint_vector / norm(waypoint_vector)
        future_point = global_waypoint + (norm_vector * i * 2.0)  # Extrapolate in same direction
        push!(policy, Dict(
            "waypoint" => [future_point[1], future_point[2], future_point[3]],
            "step" => i
        ))
    end
    
    # Write waypoint and policy to file
    println("Writing waypoint to file...")
    output_dict = Dict(
        "next_waypoint" => [global_waypoint[1], global_waypoint[2], global_waypoint[3]],
        "policy" => policy,
        "current_position" => [drone_position[1], drone_position[2], drone_position[3]],
        "target_position" => [target_position[1], target_position[2], target_position[3]],
        "current_distance" => current_to_target_dist,
        "current_suitability" => current_state.suitability
    )
    
    try
        open(NEXT_WAYPOINT_PATH, "w") do f
            JSON.print(f, output_dict)
        end
        println("Successfully wrote waypoint file")
    catch e
        println("Error writing waypoint file: $e")
        # Try backup attempt with alternative approach
        try
            println("Attempting alternative write method...")
            # Make sure the directory exists
            if !isdir(dirname(NEXT_WAYPOINT_PATH))
                mkpath(dirname(NEXT_WAYPOINT_PATH))
            end
            
            # Write with manual file handling
            file = open(NEXT_WAYPOINT_PATH, "w")
            JSON.print(file, output_dict)
            close(file)
            println("Alternative write method succeeded")
        catch e2
            println("Fatal error: Could not write waypoint file: $e2")
            error("Failed to write waypoint file after multiple attempts")
        end
    end
    
    println("Planning step completed successfully")
    return global_waypoint
end

# Run main function and handle any unexpected errors
try
    main()
catch e
    println("Error in main function: $e")
    println(stacktrace())
    rethrow(e)  # Ensure the error is visible to caller
end