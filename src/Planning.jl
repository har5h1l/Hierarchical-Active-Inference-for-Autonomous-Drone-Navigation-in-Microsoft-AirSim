module Planning

export ActionPlanner, select_action, generate_actions, calculate_efe

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
    generate_actions(planner::ActionPlanner)

Generate a set of possible actions (3D vectors) to evaluate.
"""
function generate_actions(planner::ActionPlanner)
    actions = Vector{SVector{3, Float64}}()
    
    # Add "stay in place" action
    push!(actions, SVector{3, Float64}(0.0, 0.0, 0.0))
    
    # Generate actions in horizontal plane (XY)
    for angle_idx in 1:planner.num_angles
        angle = 2π * (angle_idx - 1) / planner.num_angles
        
        for step_idx in 1:planner.num_step_sizes
            step_size = planner.max_step_size * step_idx / planner.num_step_sizes
            
            # Calculate XY components
            dx = step_size * cos(angle)
            dy = step_size * sin(angle)
            
            # Add horizontal actions
            push!(actions, SVector{3, Float64}(dx, dy, 0.0))
            
            # Add actions with vertical components
            push!(actions, SVector{3, Float64}(dx, dy, step_size / 2))
            push!(actions, SVector{3, Float64}(dx, dy, -step_size / 2))
        end
    end
    
    return actions
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
    select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner)

Select the best action by minimizing expected free energy.
Returns the selected action and its EFE value.
"""
function select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner)
    # Generate potential actions
    actions = generate_actions(planner)
    
    # Evaluate each action using EFE
    best_action = nothing
    best_efe = Inf
    
    for action in actions
        efe = calculate_efe(
            state, 
            beliefs, 
            action,
            pragmatic_weight=planner.pragmatic_weight,
            epistemic_weight=planner.epistemic_weight,
            risk_weight=planner.risk_weight
        )
        
        if efe < best_efe
            best_efe = efe
            best_action = action
        end
    end
    
    # If no good action found, use a safe default
    if isnothing(best_action)
        best_action = SVector{3, Float64}(0.1, 0.0, 0.0)  # Move slightly forward as default
        best_efe = Inf
    end
    
    return best_action, best_efe
end

end # module
