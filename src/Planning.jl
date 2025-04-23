module Planning

export ActionPlanner, plan_next_action, generate_trajectory

using LinearAlgebra
using StaticArrays
using ..State
using ..Inference

"""
Parameters for the ActionPlanner with active inference components
"""
struct ActionPlannerParams
    # EFE weighting parameters
    pragmatic_weight::Float64    # Weight for pragmatic value (target-seeking)
    risk_weight::Float64         # Weight for risk aversion (obstacle avoidance)
    epistemic_weight::Float64    # Weight for exploratory behavior
    
    # Safety parameters
    safety_distance::Float64
    
    # Planning parameters
    max_step_size::Float64
    lookahead_steps::Int
    
    # Action space parameters
    num_directions::Int          # Number of directions to sample
    num_step_sizes::Int          # Number of step sizes to try per direction
end

"""
ActionPlanner integrates active inference principles for action selection
"""
struct ActionPlanner
    params::ActionPlannerParams
end

"""
    ActionPlanner(; kwargs...)

Constructor for ActionPlanner with default parameters
"""
function ActionPlanner(;
    pragmatic_weight = 1.0,
    risk_weight = 2.0,
    epistemic_weight = 0.2,
    safety_distance = 1.5,
    max_step_size = 0.5,
    lookahead_steps = 5,
    num_directions = 12,
    num_step_sizes = 3
)
    params = ActionPlannerParams(
        pragmatic_weight,
        risk_weight,
        epistemic_weight,
        safety_distance,
        max_step_size,
        lookahead_steps,
        num_directions,
        num_step_sizes
    )
    
    return ActionPlanner(params)
end

"""
    plan_next_action(planner::ActionPlanner, state::State, beliefs::StateBeliefs)

Generate the next action using active inference principles to minimize expected free energy.
Returns a 3D vector representing the movement to execute.
"""
function plan_next_action(planner::ActionPlanner, state::State, beliefs::StateBeliefs)
    # Generate potential actions
    potential_actions = generate_potential_actions(planner, state)
    
    # Evaluate each action using expected free energy
    best_action = nothing
    best_efe = Inf
    
    for action in potential_actions
        # Calculate expected free energy for this action
        efe = Inference.compute_free_energy(state, beliefs, action)
        
        if efe < best_efe
            best_efe = efe
            best_action = action
        end
    end
    
    # If no suitable action found, return a safe default
    if isnothing(best_action)
        @warn "No suitable action found, using safe default"
        
        # Default action: small step toward target, but not too close to obstacles
        direction = state.direction_to_target
        if norm(direction) > 0.001
            best_action = direction * min(planner.params.max_step_size / 2, 0.1)
        else
            best_action = SVector{3, Float64}(0.0, 0.0, 0.0)
        end
    end
    
    return best_action
end

"""
    generate_potential_actions(planner::ActionPlanner, state::State)

Generate a set of potential actions to evaluate based on active inference principles.
Returns a list of 3D movement vectors.
"""
function generate_potential_actions(planner::ActionPlanner, state::State)
    actions = Vector{SVector{3, Float64}}()
    
    # Target-directed actions
    generate_target_directed_actions!(actions, planner, state)
    
    # Exploratory actions to reduce uncertainty
    generate_exploratory_actions!(actions, planner, state)
    
    # Obstacle avoidance actions
    generate_obstacle_avoidance_actions!(actions, planner, state)
    
    # Add hover/wait action
    push!(actions, SVector{3, Float64}(0.0, 0.0, 0.0))
    
    return actions
end

"""
    generate_target_directed_actions!(actions, planner, state)

Generate actions that move toward the target (pragmatic value).
"""
function generate_target_directed_actions!(actions, planner::ActionPlanner, state::State)
    # Use the direction vector to target
    direction = state.direction_to_target
    
    if norm(direction) > 0.001
        # Add actions with different step sizes
        for scale in range(0.25, 1.0, length=planner.params.num_step_sizes)
            step_size = min(planner.params.max_step_size * scale, state.distance_to_target)
            push!(actions, direction * step_size)
        end
        
        # Add variations around the direct path
        for angle in range(-π/6, π/6, length=planner.params.num_directions ÷ 3)
            # Rotate in XY plane
            rotation = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1]
            rotated_direction = SVector{3, Float64}(rotation * direction)
            
            for scale in range(0.5, 1.0, length=planner.params.num_step_sizes-1)
                step_size = planner.params.max_step_size * scale
                push!(actions, rotated_direction * step_size)
            end
        end
    end
end

"""
    generate_exploratory_actions!(actions, planner, state)

Generate exploratory actions that help reduce uncertainty (epistemic value).
"""
function generate_exploratory_actions!(actions, planner::ActionPlanner, state::State)
    # Generate actions in all primary and diagonal directions
    base_directions = [
        SVector{3, Float64}(1.0, 0.0, 0.0),
        SVector{3, Float64}(0.0, 1.0, 0.0),
        SVector{3, Float64}(0.0, 0.0, 1.0),
        SVector{3, Float64}(1.0, 1.0, 0.0),
        SVector{3, Float64}(1.0, 0.0, 1.0),
        SVector{3, Float64}(0.0, 1.0, 1.0),
        SVector{3, Float64}(1.0, 1.0, 1.0)
    ]
    
    # Normalize and add with small step size
    for dir in base_directions
        normalized_dir = normalize(dir)
        push!(actions, normalized_dir * (planner.params.max_step_size * 0.5))
        
        # Also add the negative direction
        push!(actions, -normalized_dir * (planner.params.max_step_size * 0.5))
    end
end

"""
    generate_obstacle_avoidance_actions!(actions, planner, state)

Generate actions that avoid obstacles (risk aversion).
"""
function generate_obstacle_avoidance_actions!(actions, planner::ActionPlanner, state::State)
    # If close to an obstacle, add "back away" action
    if state.distance_to_obstacle < planner.params.safety_distance
        away_direction = state.position - state.nearest_obstacle
        if norm(away_direction) > 0.001
            away_direction = normalize(away_direction)
            push!(actions, away_direction * planner.params.max_step_size)
        end
    end
    
    # Generate actions perpendicular to obstacles
    for obstacle in state.obstacles
        if norm(obstacle - state.position) < planner.params.safety_distance * 2
            # Vector from drone to obstacle
            to_obstacle = obstacle - state.position
            
            if norm(to_obstacle) > 0.001
                # Generate perpendicular vectors (crude way: swap and negate components)
                perp1 = SVector{3, Float64}(to_obstacle[2], -to_obstacle[1], to_obstacle[3])
                perp2 = SVector{3, Float64}(to_obstacle[3], to_obstacle[2], -to_obstacle[1])
                
                # Normalize and add
                if norm(perp1) > 0.001
                    perp1 = normalize(perp1)
                    push!(actions, perp1 * (planner.params.max_step_size * 0.75))
                    push!(actions, -perp1 * (planner.params.max_step_size * 0.75))
                end
                
                if norm(perp2) > 0.001
                    perp2 = normalize(perp2)
                    push!(actions, perp2 * (planner.params.max_step_size * 0.75))
                    push!(actions, -perp2 * (planner.params.max_step_size * 0.75))
                end
            end
        end
    end
end

"""
    generate_trajectory(planner::ActionPlanner, start_state::State, 
                       beliefs::StateBeliefs, num_steps::Int)

Generate a predicted trajectory over multiple steps using active inference.
Returns a list of predicted positions.
"""
function generate_trajectory(planner::ActionPlanner, start_state::State, 
                           beliefs::StateBeliefs, num_steps::Int)
    trajectory = [start_state.position]
    
    # Copy state and beliefs for prediction
    current_state = deepcopy(start_state)
    current_beliefs = deepcopy(beliefs)
    
    for _ in 1:num_steps
        # Plan next action using active inference
        action = plan_next_action(planner, current_state, current_beliefs)
        
        # Update predicted position
        new_position = current_state.position + action
        push!(trajectory, new_position)
        
        # Update state for next iteration (simplified)
        new_distance = norm(current_state.target_position - new_position)
        new_direction = normalize(current_state.target_position - new_position)
        new_angle = atan(new_direction[2], new_direction[1])
        
        # Create a new simple state for the next prediction
        # This is a simplified version just for trajectory prediction
        current_state = State.State(
            new_position,
            current_state.orientation,
            current_state.target_position,
            new_distance,
            new_direction,
            new_angle,
            current_state.nearest_obstacle,
            max(0.1, current_state.distance_to_obstacle - norm(action)),
            current_state.obstacles,
            current_state.voxel_grid,
            current_state.position_uncertainty,
            current_state.obstacle_uncertainty
        )
        
        # Update beliefs based on predicted state
        current_beliefs = Inference.update_beliefs(current_beliefs, current_state, action)
    end
    
    return trajectory
end

end # module
