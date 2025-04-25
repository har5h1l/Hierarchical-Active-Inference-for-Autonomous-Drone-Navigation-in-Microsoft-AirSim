module Planning

export ActionPlanner, select_action, generate_waypoints, calculate_efe, simulate_transition, calculate_suitability_for_waypoint, PreferenceModel, evaluate_preference

using LinearAlgebra
using StaticArrays
using ..StateSpace
using ..Inference

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
    distance_weight = 1.0,
    distance_scaling = 0.1,
    angle_weight = 0.8,
    angle_sharpness = 5.0,
    suitability_weight = 1.5,
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
For S1: lower distance is better, returns high score for low distances.
"""
function evaluate_distance_preference(distance::Float64, model::PreferenceModel)::Float64
    # Normalize distance to [0,1] range (clamping at max distance)
    normalized_dist = min(distance / model.max_distance, 1.0)
    
    # Transform to preference score: 1 when distance=0, approaching 0 as distance increases
    # Exponential decay provides steeper preference for closer distances
    preference = exp(-model.distance_scaling * distance)
    
    return preference
end

"""
    evaluate_angle_preference(angle::Float64)::Float64

Evaluate preference for a given angle (azimuth or elevation).
For S2/S3: 0 degrees (directly toward target) is best.
"""
function evaluate_angle_preference(angle::Float64, sharpness::Float64)::Float64
    # Convert angle to absolute value in radians (we care about magnitude, not direction)
    abs_angle = abs(angle)
    
    # Prefer angles closer to 0 (directly toward target)
    # Using cosine: 1 when angle=0, decreasing as angle increases
    # Raised to a power for sharper preference peak
    preference = (cos(abs_angle) + 1) / 2
    preference = preference ^ sharpness
    
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
"""
function calculate_suitability_for_waypoint(waypoint::SVector{3, Float64}, obstacle_distance::Float64, obstacle_density::Float64, density_weight::Float64 = 1.0)::Float64
    # Safety factor increases with distance to obstacle (exp(-1/d))
    safety_factor = exp(-1.0 / max(obstacle_distance, 0.1))
    
    # Density factor decreases with higher obstacle density (exp(-density * weight))
    density_factor = exp(-obstacle_density * density_weight * 5.0)
    
    # Combine factors - both should be high for good suitability
    return safety_factor * density_factor
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
    
    # Safety factor increases with distance to obstacle (exp(-1/d))
    safety_factor = exp(-1.0 / max(local_obstacle_distance, 0.1))
    
    # Density factor decreases with higher obstacle density (exp(-density * weight))
    density_factor = exp(-local_density * density_weight * 5.0)
    
    # Combine factors - both should be high for good suitability
    suitability = safety_factor * density_factor
    
    # Construct new DroneState with updated fields
    return StateSpace.DroneState(
        distance = dist_to_target,
        azimuth = azimuth,
        elevation = elevation,
        suitability = suitability,
        obstacle_density = local_density
    )
end

"""
    calculate_efe(state, beliefs, action, preference_model;
                 pragmatic_weight=1.0, epistemic_weight=0.2, risk_weight=2.0)

Calculate the Expected Free Energy for a potential action using the preference model.
Balances pragmatic value (reaching target) with epistemic value (exploration) and risk (safety).
"""
function calculate_efe(state, beliefs, action, preference_model;
                      pragmatic_weight=1.0, epistemic_weight=0.2, risk_weight=2.0)
    # Use expected_state from beliefs
    expected = Inference.expected_state(beliefs)
    
    # Calculate pragmatic value using preference model (higher preference should lower EFE)
    preference_score = evaluate_preference(state, preference_model)
    
    # Invert preference score since we want to minimize EFE
    # Scale by action magnitude (prefer actions that improve preference)
    action_mag = norm(action)
    pragmatic_value = -pragmatic_weight * preference_score * action_mag
    
    # Calculate epistemic value - we want to reduce uncertainty where it matters
    # Extract entropy from belief distributions
    dist_entropy = -sum(beliefs.distance_belief .* log.(beliefs.distance_belief .+ 1e-10))
    azim_entropy = -sum(beliefs.azimuth_belief .* log.(beliefs.azimuth_belief .+ 1e-10))
    elev_entropy = -sum(beliefs.elevation_belief .* log.(beliefs.elevation_belief .+ 1e-10))
    suit_entropy = -sum(beliefs.suitability_belief .* log.(beliefs.suitability_belief .+ 1e-10))
    dens_entropy = -sum(beliefs.density_belief .* log.(beliefs.density_belief .+ 1e-10))
    
    # Total entropy
    total_entropy = dist_entropy + azim_entropy + elev_entropy + suit_entropy + dens_entropy
    
    # Scale by action magnitude (prefer larger movements when uncertain)
    epistemic_value = -epistemic_weight * total_entropy * action_mag
    
    # For risk, use suitability from state
    risk_value = risk_weight * (1 - state.suitability) * action_mag
    
    # Total Expected Free Energy (lower is better)
    efe = pragmatic_value + risk_value + epistemic_value
    
    return efe
end

"""
    select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner, 
                 current_position::SVector{3, Float64}, target_position::SVector{3, Float64};
                 obstacle_distance::Float64 = 10.0, obstacle_density::Float64 = 0.0, 
                 num_policies::Int = 5)

Select the best actions by minimizing expected free energy.
Generates continuous waypoints, computes suitability, simulates transitions,
calculates EFE, and returns top num_policies actions and their EFE values.
"""
function select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner, 
                     current_position::SVector{3, Float64}, target_position::SVector{3, Float64};
                     obstacle_distance::Float64 = 10.0, obstacle_density::Float64 = 0.0, 
                     num_policies::Int = 5)
    # Generate baseline continuous waypoints
    waypoints = generate_waypoints(current_position, planner.max_step_size, planner.num_angles, planner.num_step_sizes)
    
    # Compute suitability for each waypoint
    waypoint_suitabilities = [(wp, calculate_suitability_for_waypoint(wp, obstacle_distance, obstacle_density, planner.density_weight)) for wp in waypoints]
    
    # Sort by suitability descending
    sorted_waypoints = sort(waypoint_suitabilities, by = x -> x[2], rev=true)
    
    # Select top 100 waypoints by suitability (or fewer if less available)
    top_n = min(100, length(sorted_waypoints))
    top_waypoints = sorted_waypoints[1:top_n]
    
    # For each, simulate transition state and calculate EFE
    candidate_actions = Vector{Tuple{SVector{3, Float64}, Float64}}()
    
    for (wp, suitability) in top_waypoints
        # Simulate transition state
        next_state = simulate_transition(state, wp, target_position, beliefs.voxel_grid, obstacle_distance, obstacle_density, planner.density_weight)
        
        # Calculate action vector from current position to waypoint
        action = wp - current_position
        
        # Calculate EFE for this action using the preference model
        efe = calculate_efe(
            next_state,
            beliefs,
            action,
            planner.preference_model,
            pragmatic_weight=planner.pragmatic_weight,
            epistemic_weight=planner.epistemic_weight,
            risk_weight=planner.risk_weight
        )
        
        push!(candidate_actions, (action, efe))
    end
    
    # Sort candidate actions by EFE ascending (lower is better)
    sorted_candidates = sort(candidate_actions, by = x -> x[2])
    
    # Select top num_policies actions
    top_k = min(num_policies, length(sorted_candidates))
    selected = sorted_candidates[1:top_k]
    
    # Return top actions and their EFE values
    return selected
end

end # module
