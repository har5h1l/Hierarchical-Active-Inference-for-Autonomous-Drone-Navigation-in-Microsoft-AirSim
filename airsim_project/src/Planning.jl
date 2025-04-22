module Planning

export ActionPlanner, plan_next_action, generate_trajectory

using LinearAlgebra
using StaticArrays
using ..State
using ..Inference

"""
Parameters for the ActionPlanner
"""
struct ActionPlannerParams
    # Goal weighting parameters
    distance_weight::Float64
    obstacle_weight::Float64
    
    # Safety parameters
    safety_distance::Float64
    
    # Planning parameters
    max_step_size::Float64
    lookahead_steps::Int
end

"""
ActionPlanner combines state beliefs with preferences to select actions
"""
struct ActionPlanner
    params::ActionPlannerParams
end

"""
    ActionPlanner(; kwargs...)

Constructor for ActionPlanner with default parameters
"""
function ActionPlanner(;
    distance_weight = 1.0,
    obstacle_weight = 2.0,
    safety_distance = 1.5,
    max_step_size = 0.5,
    lookahead_steps = 5
)
    params = ActionPlannerParams(
        distance_weight,
        obstacle_weight,
        safety_distance,
        max_step_size,
        lookahead_steps
    )
    
    return ActionPlanner(params)
end

"""
    plan_next_action(planner::ActionPlanner, state::State, beliefs::StateBeliefs)

Generate the next action (movement vector) based on current state and beliefs.
Returns a 3D vector representing the movement to execute.
"""
function plan_next_action(planner::ActionPlanner, state::State, beliefs::StateBeliefs)
    # Generate potential actions
    potential_actions = generate_potential_actions(planner, state)
    
    # Evaluate each action
    best_action = nothing
    best_score = Inf
    
    for action in potential_actions
        # Calculate the expected free energy for this action
        score = evaluate_action(planner, action, state, beliefs)
        
        if score < best_score
            best_score = score
            best_action = action
        end
    end
    
    # If no suitable action found, return a safe default
    if isnothing(best_action)
        @warn "No suitable action found, using safe default"
        
        # Default action: small step toward target, but not too close to obstacles
        direction = state.target_position - state.position
        if norm(direction) > 0.001
            direction = direction / norm(direction) * min(planner.params.max_step_size / 2, 0.1)
        else
            direction = SVector{3, Float64}(0.0, 0.0, 0.0)
        end
        
        best_action = direction
    end
    
    return best_action
end

"""
    evaluate_action(planner::ActionPlanner, action::AbstractVector, state::State, beliefs::StateBeliefs)

Evaluate an action's expected value using active inference principles.
Returns a score where lower values are better.
"""
function evaluate_action(planner::ActionPlanner, action::AbstractVector, state::State, beliefs::StateBeliefs)
    # Predict next position after taking this action
    next_position = state.position + action
    
    # Check distance to target after action
    target_vector = state.target_position - next_position
    distance_to_target = norm(target_vector)
    
    # Check distances to obstacles after action
    obstacle_distances = [norm(next_position - obstacle) for obstacle in state.obstacles]
    min_obstacle_distance = isempty(obstacle_distances) ? Inf : minimum(obstacle_distances)
    
    # Calculate score components
    
    # 1. Target distance component (lower is better)
    distance_score = planner.params.distance_weight * distance_to_target
    
    # 2. Obstacle avoidance component (penalty for getting too close)
    obstacle_score = 0.0
    if min_obstacle_distance < planner.params.safety_distance
        obstacle_penalty = planner.params.obstacle_weight * 
                        (planner.params.safety_distance - min_obstacle_distance)^2
        obstacle_score += obstacle_penalty
    end
    
    # 3. Action magnitude penalty (prefer smaller movements)
    action_magnitude_penalty = 0.1 * norm(action)
    
    # Compute expected free energy (used in active inference)
    # Lower values are better
    expected_free_energy = distance_score + obstacle_score + action_magnitude_penalty
    
    return expected_free_energy
end

"""
    generate_potential_actions(planner::ActionPlanner, state::State)

Generate a set of potential actions to evaluate based on the current state.
Returns a list of 3D movement vectors.
"""
function generate_potential_actions(planner::ActionPlanner, state::State)
    actions = Vector{SVector{3, Float64}}()
    
    # Calculate direction to target
    direction_to_target = state.target_position - state.position
    distance_to_target = norm(direction_to_target)
    
    if distance_to_target > 0.001
        # Normalize direction vector
        normalized_direction = direction_to_target / distance_to_target
        
        # Add action directly toward target with different magnitudes
        for scale in [0.25, 0.5, 0.75, 1.0]
            step_size = min(planner.params.max_step_size * scale, distance_to_target)
            push!(actions, normalized_direction * step_size)
        end
        
        # Add variations around the direct path
        for angle in [-π/8, -π/16, π/16, π/8]
            # Rotate in XY plane
            rotation = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1]
            rotated_direction = rotation * normalized_direction
            
            for scale in [0.5, 0.75, 1.0]
                step_size = planner.params.max_step_size * scale
                push!(actions, rotated_direction * step_size)
            end
        end
        
        # Add variations in Z (altitude)
        for dz in [-0.2, -0.1, 0.1, 0.2]
            # Only add altitude changes if we're not too close to the ground
            if state.position[3] + dz > -10.0  # Assuming Z is negative going down
                step_size = min(planner.params.max_step_size * 0.75, distance_to_target)
                action = normalized_direction * step_size
                action = SVector{3, Float64}(action[1], action[2], action[3] + dz)
                push!(actions, action)
            end
        end
    end
    
    # Add hover/wait action
    push!(actions, SVector{3, Float64}(0.0, 0.0, 0.0))
    
    # If too close to an obstacle, add "back away" action
    if state.distance_to_obstacle < planner.params.safety_distance
        away_direction = state.position - state.nearest_obstacle
        if norm(away_direction) > 0.001
            away_direction = away_direction / norm(away_direction)
            push!(actions, away_direction * planner.params.max_step_size)
        end
    end
    
    return actions
end

"""
    generate_trajectory(planner::ActionPlanner, start_state::State, 
                       beliefs::StateBeliefs, num_steps::Int)

Generate a predicted trajectory over multiple steps.
Returns a list of predicted positions.
"""
function generate_trajectory(planner::ActionPlanner, start_state::State, 
                           beliefs::StateBeliefs, num_steps::Int)
    trajectory = [start_state.position]
    
    # Copy the state to modify it for predictions
    current_state = deepcopy(start_state)
    current_beliefs = deepcopy(beliefs)
    
    for _ in 1:num_steps
        # Plan next action
        action = plan_next_action(planner, current_state, current_beliefs)
        
        # Update predicted position
        new_position = current_state.position + action
        push!(trajectory, new_position)
        
        # Update state for next iteration
        current_state = State.State(
            new_position,
            current_state.orientation,
            current_state.target_position,
            norm(current_state.target_position - new_position),
            atan(new_position[2] - current_state.target_position[2], 
                 new_position[1] - current_state.target_position[1]),
            current_state.nearest_obstacle,
            norm(current_state.nearest_obstacle - new_position),
            current_state.obstacles,
            current_state.position_uncertainty,
            current_state.obstacle_uncertainty
        )
    end
    
    return trajectory
end

end # module
