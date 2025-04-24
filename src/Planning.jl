module Planning

export ActionPlanner, select_action, generate_waypoints, calculate_efe, simulate_transition, calculate_suitability_for_waypoint

using LinearAlgebra
using StaticArrays
using ..StateSpace
using ..Inference

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
    safety_distance = 1.5
)
    return ActionPlanner(
        max_step_size,
        num_angles,
        num_step_sizes,
        pragmatic_weight,
        epistemic_weight,
        risk_weight,
        safety_distance
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
    calculate_suitability_for_waypoint(waypoint::SVector{3, Float64})::Float64

Mock function to calculate suitability score for a waypoint.
Returns exp(-1.0 / distance_to_nearest_obstacle(waypoint)) for now.
"""
function calculate_suitability_for_waypoint(waypoint::SVector{3, Float64})::Float64
    # Mock distance to nearest obstacle - replace with actual sensing or map query
    function distance_to_nearest_obstacle(p::SVector{3, Float64})::Float64
        # Placeholder: assume no obstacles nearby, return large value
        return 10.0
    end
    
    dist = distance_to_nearest_obstacle(waypoint)
    return exp(-1.0 / dist)
end

"""
    simulate_transition(state::StateSpace.DroneState, waypoint::SVector{3, Float64}, target_position::SVector{3, Float64})::StateSpace.DroneState

Simulate the predicted next state given the current state and a waypoint.
Computes distance to target, azimuth and elevation from waypoint to target,
and leaves suitability to be filled later.
"""
function simulate_transition(state::StateSpace.DroneState, waypoint::SVector{3, Float64}, target_position::SVector{3, Float64})::StateSpace.DroneState
    # Vector from waypoint to target
    to_target = target_position - waypoint
    dist_to_target = norm(to_target)
    
    # Compute azimuth and elevation angles
    azimuth = atan(to_target[2], to_target[1])  # atan(y, x)
    elevation = atan(to_target[3], sqrt(to_target[1]^2 + to_target[2]^2))
    
    # Construct new DroneState with updated fields
    # Assuming DroneState has fields: distance, azimuth, elevation, suitability
    # Suitability left blank or zero for now
    return StateSpace.DroneState(
        distance = dist_to_target,
        azimuth = azimuth,
        elevation = elevation,
        suitability = 0.0
    )
end

"""
    calculate_efe(state, beliefs, action;
                 pragmatic_weight=1.0, epistemic_weight=0.2, risk_weight=2.0)

Calculate the Expected Free Energy for a potential action.
Balances pragmatic value (reaching target) with epistemic value (exploration) and risk (safety).
"""
function calculate_efe(state, beliefs, action;
                      pragmatic_weight=1.0, epistemic_weight=0.2, risk_weight=2.0)
    # Use expected_state from beliefs to determine predicted state qualities
    expected = Inference.expected_state(beliefs)
    
    # Calculate improvement in distance (negative because we want to minimize EFE)
    # Use the expected distance from beliefs and estimated improvement based on action
    pragmatic_value = -pragmatic_weight * norm(action) * cos(expected.azimuth)
    
    # Calculate epistemic value - we want to reduce uncertainty where it matters
    # Extract entropy from belief distributions
    dist_entropy = -sum(beliefs.distance_belief .* log.(beliefs.distance_belief .+ 1e-10))
    azim_entropy = -sum(beliefs.azimuth_belief .* log.(beliefs.azimuth_belief .+ 1e-10))
    elev_entropy = -sum(beliefs.elevation_belief .* log.(beliefs.elevation_belief .+ 1e-10))
    suit_entropy = -sum(beliefs.suitability_belief .* log.(beliefs.suitability_belief .+ 1e-10))
    
    # Total entropy
    total_entropy = dist_entropy + azim_entropy + elev_entropy + suit_entropy
    
    # Scale by action magnitude (prefer larger movements when uncertain)
    action_mag = norm(action)
    epistemic_value = -epistemic_weight * total_entropy * action_mag
    
    # For risk, use suitability from beliefs
    risk_value = risk_weight * (1 - expected.suitability) * action_mag
    
    # Total Expected Free Energy (lower is better)
    efe = pragmatic_value + risk_value + epistemic_value
    
    return efe
end

"""
    select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner, current_position::SVector{3, Float64}, target_position::SVector{3, Float64}; num_policies::Int = 5)

Select the best actions by minimizing expected free energy.
Generates continuous waypoints, computes suitability, simulates transitions,
calculates EFE, and returns top num_policies actions and their EFE values.
"""
function select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner, current_position::SVector{3, Float64}, target_position::SVector{3, Float64}; num_policies::Int = 5)
    # Generate baseline continuous waypoints
    waypoints = generate_waypoints(current_position, planner.max_step_size, planner.num_angles, planner.num_step_sizes)
    
    # Compute suitability for each waypoint
    waypoint_suitabilities = [(wp, calculate_suitability_for_waypoint(wp)) for wp in waypoints]
    
    # Sort by suitability descending
    sorted_waypoints = sort(waypoint_suitabilities, by = x -> x[2], rev=true)
    
    # Select top 100 waypoints by suitability (or fewer if less available)
    top_n = min(100, length(sorted_waypoints))
    top_waypoints = sorted_waypoints[1:top_n]
    
    # For each, simulate transition state and calculate EFE
    candidate_actions = Vector{Tuple{SVector{3, Float64}, Float64}}()
    
    for (wp, suitability) in top_waypoints
        # Simulate transition state
        next_state = simulate_transition(state, wp, target_position)
        
        # Update suitability in next_state
        next_state = StateSpace.DroneState(
            distance = next_state.distance,
            azimuth = next_state.azimuth,
            elevation = next_state.elevation,
            suitability = suitability
        )
        
        # Calculate action vector from current position to waypoint
        action = wp - current_position
        
        # Calculate EFE for this action
        efe = calculate_efe(
            next_state,
            beliefs,
            action,
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
