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
    local high_density_area = false
    local critical_obstacle_avoidance = false
    local exploration_factor = 0.5
    local obstacle_priority = 1.0
    local dynamic_replanning = false
    
    if isfile(OBS_INPUT_PATH)
        try
            obs_data = open(OBS_INPUT_PATH) do f
                JSON.parse(f)
            end
            obstacle_repulsion_weight = get(obs_data, "obstacle_repulsion_weight", 0.0)
            direct_path_clear = get(obs_data, "direct_path_clear", true)
            direct_path_suitability = get(obs_data, "direct_path_suitability", 1.0)
            high_density_area = get(obs_data, "high_density_area", false)
            critical_obstacle_avoidance = get(obs_data, "critical_obstacle_avoidance", false)
            exploration_factor = get(obs_data, "exploration_factor", 0.5)
            obstacle_priority = get(obs_data, "obstacle_priority", 1.0)
            dynamic_replanning = get(obs_data, "dynamic_replanning", false)
            
            println("Retrieved obstacle repulsion weight: $obstacle_repulsion_weight")
            println("Direct path clear: $direct_path_clear, suitability: $direct_path_suitability")
            println("High density area: $high_density_area")
            println("Critical obstacle avoidance: $critical_obstacle_avoidance")
            println("Exploration factor: $exploration_factor")
            println("Obstacle priority: $obstacle_priority")
            println("Dynamic replanning: $dynamic_replanning")
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
    
    # Start with a balanced base weight 
    target_preference_weight = 0.5
    
    # Get initial distance for reference (if available) or use current as fallback
    initial_distance = if haskey(data, "initial_distance")
        data["initial_distance"]
    else
        current_to_target_dist * 1.2  # Add 20% if not provided
    end
    # Calculate percentage of distance to target (how close we are)
    distance_percentage = (current_to_target_dist / initial_distance) * 100.0
    
    println("Planning: Distance percentage to target: $(round(distance_percentage, digits=1))% of initial $(round(initial_distance, digits=2))m")
    
    # Enhanced distance-based scaling using percentages rather than fixed distances
    if distance_percentage < 5.0  # Very close to target (within final 5% of journey)
        # When very close to target (final 5%), strongly favor direct approach
        target_preference_weight = 0.8  # High target preference
        println("Planning: Very close to target (<5% remaining), strongly prioritizing target approach: $(round(target_preference_weight, digits=2))")
    elseif distance_percentage < 15.0  # Close to target (5-15% of journey remaining)
        # Close to target but not final approach
        target_preference_weight = 0.7
        println("Planning: Close to target (5-15% remaining), prioritizing target approach: $(round(target_preference_weight, digits=2))")
    elseif distance_percentage < 33.0  # Approaching target (final third of journey)
        # In the final third of journey, gradually increase target preference
        # Scale from 0.5 to 0.65 as percentage decreases from 33% to 15%
        target_preference_weight = 0.5 + (0.15 * (33.0 - distance_percentage) / 18.0)
        println("Planning: Approaching target ($(round(distance_percentage, digits=1))% remaining), balanced weights: $(round(target_preference_weight, digits=2))")
    elseif distance_percentage > 75.0  # Still far from target (>75% of journey remaining)
        # When far from target, maintain higher obstacle avoidance (lower target preference)
        target_preference_weight = 0.4
        println("Planning: Far from target (>75% remaining), focusing on safe navigation: $(round(target_preference_weight, digits=2))")
    else
        # In the middle range (33-75%), use a moderate balanced approach
        # Scale from 0.4 to 0.5 as percentage decreases from 75% to 33%
        target_preference_weight = 0.4 + (0.1 * (75.0 - distance_percentage) / 42.0)
        println("Planning: Steady progress toward target ($(round(distance_percentage, digits=1))% remaining): $(round(target_preference_weight, digits=2))")
    end
    
    # Adjust based on obstacle density - less dramatic effect when close to target
    if high_density_area
        if distance_percentage < 5.0
            # When close to target, reduce penalty for high density
            target_preference_weight -= 0.1  # Reduced from 0.15
            println("High density area near target, small reduction: -0.1")
        else
            target_preference_weight -= 0.15
            println("High density area, standard reduction: -0.15")
        end
    end
    
    # Reduced impact of obstacle priority when close to target
    if obstacle_priority > 1.0
        if distance_percentage < 5.0
            # Less reduction when close to target
            reduction = min(0.15, (obstacle_priority - 1.0) * 0.08)
            target_preference_weight -= reduction
            println("Obstacle priority near target, reduced impact: -$(round(reduction, digits=2))")
        else
            reduction = min(0.2, (obstacle_priority - 1.0) * 0.1)
            target_preference_weight -= reduction
            println("Obstacle priority impact: -$(round(reduction, digits=2))")
        end
    end
    
    # Adjust based on direct path clearance
    if direct_path_clear && direct_path_suitability > 0.8
        # Clear path bonus
        bonus = direct_path_suitability * 0.2  # Scale bonus based on path quality
        target_preference_weight += bonus
        println("Clear direct path ($(round(direct_path_suitability, digits=2))), bonus: +$(round(bonus, digits=2))")
    elseif !direct_path_clear || direct_path_suitability < 0.5
        # Blocked path penalty - reduced when very close to target
        if distance_percentage < 5.0
            target_preference_weight -= 0.1  # Reduced penalty when very close
            println("Blocked path very near target, reduced penalty: -0.1")
        else
            target_preference_weight -= 0.15
            println("Blocked path, standard penalty: -0.15")
        end
    end
    
    # Factor in explicit obstacle repulsion - less impact when close to target
    if distance_percentage < 5.0
        target_preference_weight -= obstacle_repulsion_weight * 0.05  # Half effect when close
        println("Obstacle repulsion near target, reduced impact: -$(round(obstacle_repulsion_weight * 0.05, digits=2))")
    else
        target_preference_weight -= obstacle_repulsion_weight * 0.1
        println("Obstacle repulsion impact: -$(round(obstacle_repulsion_weight * 0.1, digits=2))")
    end    # Handle critical avoidance scenarios - reduced impact when close to target
    if critical_obstacle_avoidance || dynamic_replanning
        # Determine if we're in final approach to target (extremely close)
        final_approach = distance_percentage < 3.0 && high_density_area
        
        if final_approach
            # Special handling for final approach to prevent excessive obstacle avoidance
            # Allow higher target preference to ensure the drone can reach the goal
            target_preference_weight = min(max(target_preference_weight, 0.6), 0.75)
            println("🎯 Final approach in critical scenario, ensuring high target preference: $(round(target_preference_weight, digits=2))")
        elseif distance_percentage < 5.0
            # Higher cap when very close to target
            target_preference_weight = min(target_preference_weight, 0.45)  # Decreased from 0.5
            println("Critical scenario very near target, capping at 0.45")
        elseif distance_percentage < 8.0
            # Moderate cap when somewhat close
            target_preference_weight = min(target_preference_weight, 0.35)  # Decreased from 0.4
            println("Critical scenario near target, capping at 0.35")
        else
            target_preference_weight = min(target_preference_weight, 0.3)  # Decreased from 0.35
            println("Critical scenario, standard cap at 0.3")
        end
    end
      # Clamp to reasonable range - with higher minimum and maximum when closer to target
    if distance_percentage < 5.0
        # Higher minimum and maximum when close to target
        if high_density_area
            # Significantly increase target preference in high-density areas when very close to target
            # This ensures the drone will prioritize reaching the target over excessive obstacle avoidance
            target_preference_weight = max(0.5, min(0.9, target_preference_weight))
            println("Very close to target in high-density area, using elevated preference weight")
        else
            # Standard close-to-target adjustment in normal areas
            target_preference_weight = max(0.3, min(0.85, target_preference_weight))
        end
    else
        target_preference_weight = max(0.25, min(0.8, target_preference_weight))
    end
    
    # Additional target preference boost when extremely close to target in high-density areas
    target_preference_boost = get(data, "target_preference_boost", 0.0)
    if target_preference_boost > 0.0
        # Apply the boost from Python
        original_weight = target_preference_weight
        target_preference_weight += target_preference_boost * 0.2  # Apply 20% of the requested boost
        println("🎯 Final approach boost applied: +$(round(target_preference_boost * 0.2, digits=2)) (from $(round(original_weight, digits=2)) to $(round(target_preference_weight, digits=2)))")
    end
    
    println("Final target preference weight: $(round(target_preference_weight, digits=2))")
    println("Final obstacle avoidance weight: $(round(1.0 - target_preference_weight, digits=2))")
    
    # Create planner with preference model
    preference_model = PreferenceModel(
        distance_preference = target_preference_weight,
        path_preference = 1.0 - target_preference_weight  # Balance between these two primary preferences
    )
    
    # Determine safety margin based on context
    adaptive_safety_margin = MARGIN
      # Increase safety margin when in high density areas or with high obstacle priority
    if high_density_area
        adaptive_safety_margin *= 1.8  # Increased from 1.5
        println("Increased safety margin for high density area: $(adaptive_safety_margin)")
    end
    
    if obstacle_priority > 1.5
        adaptive_safety_margin *= (1.0 + (obstacle_priority - 1.0) * 0.3)  # Increased from 0.2
        println("Increased safety margin for high obstacle priority: $(adaptive_safety_margin)")
    end
    
    # Further increase for critical situations
    if critical_obstacle_avoidance || dynamic_replanning
        adaptive_safety_margin = max(adaptive_safety_margin * 1.5, 3.5)  # Increased values
        println("Increased safety margin for critical avoidance: $(adaptive_safety_margin)")
    end
    
    planner = ActionPlanner(preference_model)
    
    # Adjust waypoint sampling based on context
    adaptive_waypoint_count = WAYPOINT_SAMPLE_COUNT
    
    if high_density_area || obstacle_priority > 1.5
        # Use more samples in complex environments
        adaptive_waypoint_count = Int(WAYPOINT_SAMPLE_COUNT * 1.3)
        println("Increased waypoint sampling for complex environment: $(adaptive_waypoint_count)")
    end
    
    if critical_obstacle_avoidance || dynamic_replanning
        # Use significantly more samples during critical planning
        adaptive_waypoint_count = Int(WAYPOINT_SAMPLE_COUNT * 1.5)
        println("Increased waypoint sampling for critical planning: $(adaptive_waypoint_count)")
    end
      # Select the next waypoint using active inference
    println("\nSelecting next waypoint...")
    next_waypoint_result = select_action(
        planner, 
        beliefs, 
        num_samples=adaptive_waypoint_count, 
        safety_margin=adaptive_safety_margin,
        policy_length=POLICY_LENGTH,
        density_radius=get(data, "density_radius", 5.0),
        suitability_threshold=0.75  # Increased to 0.75 for safer path selection
    )
    
    # Extract next_waypoint and metadata from the result
    if isa(next_waypoint_result, Tuple) && length(next_waypoint_result) == 2
        next_waypoint, metadata = next_waypoint_result
        # Merge metadata into data dictionary for use in direct path comparison
        if isa(metadata, Dict)
            merge!(data, metadata)
            if haskey(metadata, "best_suitability")
                println("Best path suitability: $(round(metadata["best_suitability"], digits=2))")
            end
        end
    else
        next_waypoint = next_waypoint_result
    end
    
    # If we didn't get a good waypoint, try again with slightly lower threshold but still higher than default
    if isnothing(next_waypoint) || length(next_waypoint) < 3
        println("❌ Failed to find waypoint with high suitability, retrying with adjusted parameters")
          # Retry with lower threshold but still much higher than original
        next_waypoint_result = select_action(
            planner, 
            beliefs,
            num_samples=adaptive_waypoint_count * 2,  # Double the samples to find more alternatives
            safety_margin=adaptive_safety_margin * 0.85,  # Slightly reduced safety margin
            policy_length=POLICY_LENGTH,
            density_radius=get(data, "density_radius", 5.0),
            suitability_threshold=0.65  # Fallback threshold still higher than default but lower than primary
        )
        
        # Extract next_waypoint and metadata
        if isa(next_waypoint_result, Tuple) && length(next_waypoint_result) == 2
            next_waypoint, metadata = next_waypoint_result
            # Merge metadata into data dictionary
            if isa(metadata, Dict)
                merge!(data, metadata)
                if haskey(metadata, "best_suitability")
                    println("Best path suitability (fallback 1): $(round(metadata["best_suitability"], digits=2))")
                end
            end
        else
            next_waypoint = next_waypoint_result
        end
          if isnothing(next_waypoint) || length(next_waypoint) < 3
            println("⚠️ Second attempt failed, using lower threshold in emergency mode")
            
        # Last resort - focus on finding high suitability paths with more samples
            # Rather than lowering threshold further, we increase the sample count and search more thoroughly
            println("🔍 Searching more thoroughly for high quality paths...")
            next_waypoint_result = select_action(
                planner, 
                beliefs,                num_samples=adaptive_waypoint_count * 5,  # 5x samples for thorough search
                safety_margin=adaptive_safety_margin * 0.8,  # Slightly reduced but still safe
                policy_length=POLICY_LENGTH,
                density_radius=get(data, "density_radius", 5.0),
                suitability_threshold=0.6  # Emergency threshold, still relatively high
            )
            
            # Extract next_waypoint and metadata
            if isa(next_waypoint_result, Tuple) && length(next_waypoint_result) == 2
                next_waypoint, metadata = next_waypoint_result
                # Merge metadata into data dictionary
                if isa(metadata, Dict)
                    merge!(data, metadata)
                    if haskey(metadata, "best_suitability")
                        println("Best path suitability (fallback 2): $(round(metadata["best_suitability"], digits=2))")
                    end
                end
            else
                next_waypoint = next_waypoint_result
            end
        end
    end
      # Calculate waypoint in global coordinates
    global_waypoint = try
        if !isnothing(next_waypoint) && !isempty(next_waypoint) && isa(next_waypoint[1], Tuple) && length(next_waypoint[1][1]) == 3
            # Extract the first action (best by EFE) and convert it to a waypoint
            best_action = next_waypoint[1][1]
            action_magnitude = norm(best_action)
            
            # Calculate and log the action magnitude
            println("Selected action magnitude: $(round(action_magnitude, digits=2))m")
            
            # Convert action to waypoint by adding to current position
            drone_position + best_action
        else
            println("⚠️ No valid actions received from planner, using fallback")
            # Fallback: move towards target
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
      # Check if there's a clear path to target and not too far away
    target_in_range = current_to_target_dist < 25.0  # Only try direct path when within reasonable range
    direct_path_clear = get(data, "direct_path_clear", false)
    direct_path_suitability = get(data, "direct_path_suitability", 0.0)    # If direct path is clear and has good suitability, and not too distant, consider it
    if target_in_range && direct_path_clear
        # More strict suitability requirement when not close to target
        direct_path_threshold = current_to_target_dist < 5.0 ? 0.6 : 0.75
        
        # Check if direct path has sufficient suitability
        if direct_path_suitability > direct_path_threshold
            # Calculate if the planned path actually has better suitability
            # Only override if the direct path is truly better than the planned path
            best_planned_suitability = 0.0
            if !isnothing(next_waypoint) && !isempty(next_waypoint) && isa(next_waypoint[1], Tuple)
                # Get the planned waypoint for comparison
                planned_action = next_waypoint[1][1]
                planned_waypoint = drone_position + planned_action
                
                # Access data about the planned path from the planning module
                # We'll only use direct path if it's significantly better in terms of suitability
                best_idx = 1 # First action is the best one
                if haskey(data, "best_suitability")
                    best_planned_suitability = data["best_suitability"]
                end
            end
            
            # Only choose direct path if its suitability is better than the planned path
            # or if the planned path has unusually low suitability
            if direct_path_suitability > best_planned_suitability * 1.2 || best_planned_suitability < 0.6
                println("✅ Direct path to target is clear with superior suitability: $(round(direct_path_suitability, digits=2))! Using optimized path")
                
                # Calculate a direct waypoint toward target
                # Use a step size based on distance: longer steps when farther away, shorter when close
                step_size = min(current_to_target_dist * 0.5, 5.0)  # 50% of distance or max 5 meters
                step_size = max(step_size, 1.5)  # Ensure minimum step size of 1.5 meters
                
                # Create direct waypoint
                normalized_to_target = current_to_target / current_to_target_dist
                direct_waypoint = drone_position + normalized_to_target * step_size
                global_waypoint = direct_waypoint
            else
                println("ℹ️ Direct path available but planned path has better suitability ($(round(best_planned_suitability, digits=2)) vs $(round(direct_path_suitability, digits=2)))")
            end
        end
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