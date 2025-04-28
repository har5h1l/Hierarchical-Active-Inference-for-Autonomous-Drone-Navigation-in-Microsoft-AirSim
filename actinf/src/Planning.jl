module Planning

export ActionPlanner, select_action, generate_waypoints, calculate_efe
export simulate_transition, calculate_suitability_for_waypoint
export PreferenceModel, evaluate_preference

using LinearAlgebra
using StaticArrays
using ..StateSpace
using ..Inference

# Constants for suitability calculation and filtering
const SUITABILITY_THRESHOLD = 0.5
const OBSTACLE_WEIGHT = 0.7
const DENSITY_WEIGHT = 0.3
const CUTOFF_DISTANCE = 2.5  # Meters
const STEEPNESS_DISTANCE = 3.0
const CUTOFF_DENSITY = 0.2
const STEEPNESS_DENSITY = 10.0

"""
    PreferenceModel

Defines how the agent evaluates states, with explicit scoring functions for each state component.
Higher preference scores indicate more desirable states.
"""
struct PreferenceModel
    # Preference parameters for each state dimension
    distance_weight::Float64           # Weight for distance preference
    distance_scaling::Float64          # Scaling factor for distance
    angle_weight::Float64              # Weight for angle (azimuth/elevation) preferences
    angle_sharpness::Float64           # How sharply to prefer on-target angles
    suitability_weight::Float64        # Weight for environmental suitability
    suitability_threshold::Float64     # Minimum acceptable suitability
    
    # Normalization settings
    max_distance::Float64              # Maximum distance for normalization
end

"""
    PreferenceModel(; kwargs...)

Constructor with default parameters for the preference model.
"""
function PreferenceModel(;
    distance_weight = 2.0,         # Increased from 1.0 to emphasize distance more
    distance_scaling = 0.05,       # Reduced from 0.1 to make distance preference decay more gradually
    angle_weight = 0.5,           # Reduced from 0.8 to prioritize distance over angle
    angle_sharpness = 3.0,        # Reduced from 5.0 for smoother angle preferences
    suitability_weight = 1.0,      # Reduced from 1.5 to prioritize distance
    suitability_threshold = 0.3,
    max_distance = 50.0
)
    return PreferenceModel(
        distance_weight,
        distance_scaling,
        angle_weight,
        angle_sharpness,
        suitability_weight,
        suitability_threshold,
        max_distance
    )
end

"""
    evaluate_distance_preference(distance::Float64, model::PreferenceModel)::Float64

Evaluate preference for a given distance to target.
Modified to more strongly prefer shorter distances.
"""
function evaluate_distance_preference(distance::Float64, model::PreferenceModel)::Float64
    # Normalize distance to [0,1] range
    normalized_dist = min(distance / model.max_distance, 1.0)
    
    # Transform to preference score: 1 when distance=0, approaching 0 as distance increases
    # Using quadratic decay for stronger preference of shorter distances
    preference = (1.0 - normalized_dist)^2 * exp(-model.distance_scaling * distance)
    
    return preference
end

"""
    evaluate_angle_preference(angle::Float64)::Float64

Evaluate preference for a given angle (azimuth or elevation).
For S2/S3: 0 degrees (directly toward target) is best.
Includes a constant baseline preference (0.2) to gently bias the agent toward facing the target.
"""
function evaluate_angle_preference(angle::Float64, sharpness::Float64)::Float64
    # Convert angle to absolute value in radians (we care about magnitude, not direction)
    abs_angle = abs(angle)
    
    # Constant baseline preference - gently bias toward facing target
    constant_baseline = 0.2
    
    # Prefer angles closer to 0 (directly toward target)
    # Using cosine: 1 when angle=0, decreasing as angle increases
    # Raised to a power for sharper preference peak
    variable_preference = (cos(abs_angle) + 1) / 2
    variable_preference = variable_preference ^ sharpness
    
    # Combine the baseline and variable components
    # This ensures there's always some preference for facing the target,
    # but also maintains the existing angle preference structure
    preference = constant_baseline + (1.0 - constant_baseline) * variable_preference
    
    return preference
end

"""
    evaluate_suitability_preference(suitability::Float64, threshold::Float64)::Float64

Evaluate preference for environmental suitability.
For S4: higher is better, with threshold for minimum acceptable suitability.
"""
function evaluate_suitability_preference(suitability::Float64, threshold::Float64)::Float64
    # Apply threshold with smooth transition
    if suitability < threshold
        # Rapidly decreasing preference below threshold
        return suitability * (suitability / threshold) * 0.5
    else
        # Linear scaling above threshold, with bonus for exceeding threshold
        normalized = 0.5 + 0.5 * (suitability - threshold) / (1.0 - threshold)
        return normalized
    end
end

"""
    evaluate_preference(state::StateSpace.DroneState, model::PreferenceModel)::Float64

Calculate overall preference score for a drone state.
Higher scores indicate more desirable states.
"""
function evaluate_preference(state::StateSpace.DroneState, model::PreferenceModel)::Float64
    # Evaluate individual components
    distance_pref = evaluate_distance_preference(state.distance, model)
    azimuth_pref = evaluate_angle_preference(state.azimuth, model.angle_sharpness)
    elevation_pref = evaluate_angle_preference(state.elevation, model.angle_sharpness)
    suitability_pref = evaluate_suitability_preference(state.suitability, model.suitability_threshold)
    
    # Combine preferences with weights
    angle_combined = (azimuth_pref + elevation_pref) / 2
    
    total_preference = (
        model.distance_weight * distance_pref +
        model.angle_weight * angle_combined +
        model.suitability_weight * suitability_pref
    ) / (model.distance_weight + model.angle_weight + model.suitability_weight)
    
    return total_preference
end

"""
ActionPlanner contains parameters for action selection and planning
"""
struct ActionPlanner
    # Action space parameters
    max_step_size::Float64      # Maximum movement distance per step
    num_angles::Int             # Number of angles to consider
    num_step_sizes::Int         # Number of step sizes to consider
    
    # Free energy components weights
    pragmatic_weight::Float64   # Weight for goal-seeking behavior
    epistemic_weight::Float64   # Weight for uncertainty-reducing behavior
    risk_weight::Float64        # Weight for obstacle avoidance
    
    # Safety parameters
    safety_distance::Float64    # Minimum safe distance from obstacles
    density_weight::Float64     # Weight for obstacle density in suitability
    
    # Preference model
    preference_model::PreferenceModel
end

"""
    ActionPlanner(; kwargs...)

Constructor with default parameters
"""
function ActionPlanner(;
    max_step_size = 0.5,
    num_angles = 8,
    num_step_sizes = 3,
    pragmatic_weight = 1.0,
    epistemic_weight = 0.2,
    risk_weight = 2.0,
    safety_distance = 1.5,
    density_weight = 1.0,
    preference_model = PreferenceModel()
)
    return ActionPlanner(
        max_step_size,
        num_angles,
        num_step_sizes,
        pragmatic_weight,
        epistemic_weight,
        risk_weight,
        safety_distance,
        density_weight,
        preference_model
    )
end

"""
    generate_waypoints(current_position::SVector{3, Float64}, distance::Float64, num_angles::Int, num_elevations::Int)::Vector{SVector{3, Float64}}

Generate waypoints at a fixed radius from current_position in azimuth and elevation sweeps.
"""
function generate_waypoints(current_position::SVector{3, Float64}, distance::Float64, num_angles::Int, num_elevations::Int)::Vector{SVector{3, Float64}}
    waypoints = Vector{SVector{3, Float64}}()
    
    # Include current position as a waypoint (stay in place)
    push!(waypoints, current_position)
    
    for elev_idx in 1:num_elevations
        elevation = π * (elev_idx - 1) / (num_elevations - 1) - π/2  # from -pi/2 to pi/2
        
        for angle_idx in 1:num_angles
            azimuth = 2π * (angle_idx - 1) / num_angles
            
            dx = distance * cos(elevation) * cos(azimuth)
            dy = distance * cos(elevation) * sin(azimuth)
            dz = distance * sin(elevation)
            
            waypoint = current_position + SVector{3, Float64}(dx, dy, dz)
            push!(waypoints, waypoint)
        end
    end
    
    return waypoints
end

"""
    calculate_suitability_for_waypoint(waypoint::SVector{3, Float64}, obstacle_distance::Float64, obstacle_density::Float64, density_weight::Float64 = 1.0)::Float64

Calculate suitability score for a waypoint based on distance to nearest obstacle and obstacle density.
Higher suitability means the waypoint is safer to navigate to.
Uses sigmoid-like scaling for more predictable behavior.
"""
function calculate_suitability_for_waypoint(waypoint::SVector{3, Float64}, obstacle_distance::Float64, obstacle_density::Float64, density_weight::Float64 = 1.0)::Float64
    # Use constants for more consistent suitability calculation
    return StateSpace.calculate_suitability(
        obstacle_distance, 
        obstacle_density, 
        obstacle_weight=OBSTACLE_WEIGHT, 
        density_weight=DENSITY_WEIGHT*density_weight,
        cutoff_distance=CUTOFF_DISTANCE,
        steepness_distance=STEEPNESS_DISTANCE,
        cutoff_density=CUTOFF_DENSITY,
        steepness_density=STEEPNESS_DENSITY
    )
end

"""
    simulate_transition(state::StateSpace.DroneState, waypoint::SVector{3, Float64}, target_position::SVector{3, Float64}, 
                        voxel_grid::Vector{SVector{3, Float64}}, obstacle_distance::Float64, obstacle_density::Float64, density_weight::Float64 = 1.0)::StateSpace.DroneState

Simulate the predicted next state given the current state and a waypoint.
Computes distance to target, azimuth and elevation from waypoint to target,
and calculates suitability based on obstacle sensory data (voxel grid, distance and density).
"""
function simulate_transition(state::StateSpace.DroneState, waypoint::SVector{3, Float64}, target_position::SVector{3, Float64}, 
                            voxel_grid::Vector{SVector{3, Float64}}, obstacle_distance::Float64, obstacle_density::Float64, density_weight::Float64 = 1.0)::StateSpace.DroneState
    # Vector from waypoint to target
    to_target = target_position - waypoint
    dist_to_target = norm(to_target)
    
    # Compute azimuth and elevation angles
    azimuth = atan(to_target[2], to_target[1])  # atan(y, x)
    elevation = atan(to_target[3], sqrt(to_target[1]^2 + to_target[2]^2))
    
    # Calculate suitability using the voxel grid information for more accurate obstacle assessment
    local_obstacle_distance = obstacle_distance
    local_density = obstacle_density
    
    # If we have voxel data, use it to refine our suitability calculation
    if !isempty(voxel_grid)
        # Calculate distances from waypoint to each obstacle voxel
        distances_to_voxels = [norm(waypoint - voxel) for voxel in voxel_grid]
        
        # Find closest obstacle (if any voxels are available)
        if !isempty(distances_to_voxels)
            closest_obstacle_distance = minimum(distances_to_voxels)
            # Take the minimum of the sensed obstacle distance and the voxel-based calculation
            local_obstacle_distance = min(obstacle_distance, closest_obstacle_distance)
            
            # Calculate local density near the waypoint
            # Count voxels within 3m radius of the waypoint
            nearby_voxels = count(d -> d < 3.0, distances_to_voxels)
            local_volume = 4/3 * π * 3.0^3  # Volume of 3m radius sphere
            local_density = nearby_voxels / local_volume
        end
    end
    
    # Calculate suitability using the shared function for consistency
    suitability = StateSpace.calculate_suitability(local_obstacle_distance, local_density, 
                                                 obstacle_weight=0.7, density_weight=0.3*density_weight)
    
    # Construct new DroneState with updated fields
    return StateSpace.DroneState(
        distance = dist_to_target,
        azimuth = azimuth,
        elevation = elevation,
        suitability = suitability
    )
end

"""
    calculate_efe(state, beliefs, action, preference_model;
                 pragmatic_weight=1.0, epistemic_weight=0.2)

Calculate the Expected Free Energy focusing on pragmatic and epistemic values only.
Risk is handled separately through early elimination of unsafe waypoints.
"""
function calculate_efe(state::StateSpace.DroneState, 
                      beliefs::Inference.DroneBeliefs, 
                      action::SVector{3, Float64}, 
                      preference_model::PreferenceModel;
                      pragmatic_weight=1.0, 
                      epistemic_weight=0.2)
    # Use expected_state from beliefs
    expected = Inference.expected_state(beliefs)
    
    # Calculate pragmatic value using preference model
    preference_score = evaluate_preference(state, preference_model)
    
    # Add distance reduction bonus - strongly reward actions that decrease distance
    action_mag = norm(action)
    distance_reduction = expected.distance - state.distance
    distance_bonus = distance_reduction > 0 ? 2.0 * distance_reduction : 0.0
    
    # Modified pragmatic value with distance bonus
    pragmatic_value = -pragmatic_weight * (preference_score + distance_bonus) * action_mag
    
    # Calculate epistemic value from belief distributions
    dist_entropy = -sum(beliefs.distance_belief .* log.(beliefs.distance_belief .+ 1e-10))
    azim_entropy = -sum(beliefs.azimuth_belief .* log.(beliefs.azimuth_belief .+ 1e-10))
    elev_entropy = -sum(beliefs.elevation_belief .* log.(beliefs.elevation_belief .+ 1e-10))
    suit_entropy = -sum(beliefs.suitability_belief .* log.(beliefs.suitability_belief .+ 1e-10))
    dens_entropy = -sum(beliefs.density_belief .* log.(beliefs.density_belief .+ 1e-10))
    
    total_entropy = dist_entropy + azim_entropy + elev_entropy + suit_entropy + dens_entropy
    epistemic_value = -epistemic_weight * 0.5 * total_entropy * action_mag
    
    # Total Expected Free Energy (lower is better)
    # No risk value component anymore - handled by early filtering based on suitability
    return pragmatic_value + epistemic_value
end

"""
    select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner, 
                 current_position::SVector{3, Float64}, target_position::SVector{3, Float64};
                 obstacle_distance::Float64 = 10.0, obstacle_density::Float64 = 0.0, 
                 num_policies::Int = 5)

Select the best actions by first filtering out unsafe paths based on suitability,
then evaluating Expected Free Energy on remaining safe candidates.
Dynamically adjusts waypoint radius, policy length, and waypoint sampling based on environment suitability.
"""
function select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner, 
                     current_position::SVector{3, Float64}, target_position::SVector{3, Float64};
                     obstacle_distance::Float64 = 10.0, obstacle_density::Float64 = 0.0, 
                     num_policies::Int = 5)
    # Constants for adaptive parameters
    MIN_RADIUS = 0.5
    MAX_RADIUS = 3.0
    MIN_POLICY_LEN = 2
    MAX_POLICY_LEN = 5
    MIN_WAYPOINTS = 15
    MAX_WAYPOINTS = 75
    
    # Dynamically adjust parameters based on state suitability
    # For suitability, higher values (closer to 1) mean safer navigation conditions
    suitability_factor = clamp(state.suitability, 0.0, 1.0)  # Ensure it's in valid range
    
    # 1. Adjust waypoint radius (step size) - inverse relationship with suitability
    # Low suitability -> smaller radius (safer, shorter steps)
    # High suitability -> larger radius (faster exploitation)
    adaptive_radius = MIN_RADIUS + suitability_factor * (MAX_RADIUS - MIN_RADIUS)
    
    # 2. Adjust policy length - inverse relationship with suitability
    # Low suitability -> longer policy (more careful planning)
    # High suitability -> shorter policy (less planning needed)
    adaptive_policy_length = MAX_POLICY_LEN - suitability_factor * (MAX_POLICY_LEN - MIN_POLICY_LEN)
    adaptive_policy_length = round(Int, adaptive_policy_length)
    
    # 3. Adjust number of waypoints - inverse relationship with suitability
    # Low suitability -> more waypoints (greater caution, more exploration)
    # High suitability -> fewer waypoints (less exploration needed)
    adaptive_waypoint_count = MAX_WAYPOINTS - suitability_factor * (MAX_WAYPOINTS - MIN_WAYPOINTS)
    adaptive_waypoint_count = round(Int, adaptive_waypoint_count)
    
    # Calculate appropriate number of angles and elevations to achieve target waypoint count
    # Target is adaptive_waypoint_count total waypoints
    # Formula: num_angles * num_elevations + 1 ≈ adaptive_waypoint_count
    # where +1 accounts for the current position waypoint
    total_angle_divisions = max(4, round(Int, sqrt(adaptive_waypoint_count - 1)))
    num_angles = max(4, round(Int, total_angle_divisions * 1.5))  # More horizontal than vertical divisions
    num_elevations = max(2, round(Int, total_angle_divisions / 1.5))  # Fewer vertical divisions
    
    # Generate waypoints with adaptive radius for adaptive exploration
    waypoints = generate_waypoints(current_position, adaptive_radius, num_angles, num_elevations)
    
    # Calculate vector to target
    to_target = target_position - current_position
    target_direction = normalize(to_target)
    
    # Add direct-to-target waypoints with various step sizes
    for step in [0.3, 0.6, 0.9] .* adaptive_radius
        direct_waypoint = current_position + step * target_direction
        push!(waypoints, direct_waypoint)
    end
    
    # Step 1: Generate candidate waypoints and predict next states
    all_waypoints = waypoints
    all_states = Vector{StateSpace.DroneState}()
    all_actions = Vector{SVector{3, Float64}}()
    all_suitabilities = Vector{Float64}()
    
    println("Generated $(length(all_waypoints)) candidate waypoints")
    
    # Step 2: Predict next state for each candidate and calculate suitability
    for wp in all_waypoints
        next_state = simulate_transition(state, wp, target_position, beliefs.voxel_grid, obstacle_distance, obstacle_density, planner.density_weight)
        action = wp - current_position
        
        push!(all_states, next_state)
        push!(all_actions, action)
        push!(all_suitabilities, next_state.suitability)
    end
    
    # Step 3: Early elimination based on suitability threshold
    safe_indices = findall(s -> s >= SUITABILITY_THRESHOLD, all_suitabilities)
    
    # If no waypoints pass the suitability threshold, relax it to take the best available
    if isempty(safe_indices) && !isempty(all_suitabilities)
        # Find the waypoint with highest suitability if none pass threshold
        _, best_idx = findmax(all_suitabilities)
        safe_indices = [best_idx]  # Just use the best one
        println("Warning: No waypoints passed suitability threshold. Using best available.")
    end
    
    filtered_states = all_states[safe_indices]
    filtered_actions = all_actions[safe_indices]
    
    println("$(length(filtered_states)) waypoints passed suitability threshold ($(SUITABILITY_THRESHOLD))")
    
    # Step 4: Evaluate EFE on filtered (safe) waypoints
    if isempty(filtered_states)
        # Emergency fallback - just stay in place
        println("Emergency: No safe paths found. Staying in place.")
        return [(SVector{3, Float64}(0.0, 0.0, 0.0), 0.0)]
    end
    
    # For each safe candidate, calculate EFE
    efe_scores = Vector{Float64}()
    
    for (state_idx, next_state) in enumerate(filtered_states)
        action = filtered_actions[state_idx]
        
        # Calculate EFE without risk penalty (safety already ensured by filtering)
        efe = calculate_efe(
            next_state,
            beliefs,
            action,
            planner.preference_model,
            pragmatic_weight=planner.pragmatic_weight,
            epistemic_weight=planner.epistemic_weight
        )
        
        push!(efe_scores, efe)
    end
    
    # Step 5: Policy selection - choose the best actions based on EFE
    # Sort by EFE ascending (lower is better)
    sorted_indices = sortperm(efe_scores)
    
    # Use the adaptive policy length instead of fixed num_policies
    top_k = min(adaptive_policy_length, length(sorted_indices))
    
    # Create selected tuples of (action, efe)
    selected = [(filtered_actions[sorted_indices[i]], efe_scores[sorted_indices[i]]) for i in 1:top_k]
    
    # Step 7: Debug info
    if !isempty(sorted_indices) && !isempty(efe_scores)
        best_idx = sorted_indices[1]
        best_efe = efe_scores[best_idx]
        best_action = filtered_actions[best_idx]
        best_suitability = filtered_states[best_idx].suitability
        
        println("Best EFE: $(best_efe), Action: $(best_action), Suitability: $(best_suitability)")
    end
    
    return selected
end

end # module
