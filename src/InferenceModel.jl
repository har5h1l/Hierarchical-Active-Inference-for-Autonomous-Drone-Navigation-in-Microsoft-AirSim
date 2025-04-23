module InferenceModel

export drone_inference_model, run_inference, update_belief_states, calculate_efe

using RxInfer
using LinearAlgebra
using StaticArrays

"""
    drone_inference_model()

Define the generative model for active inference where:
- S1: Distance from target (scalar, integer-like)
- S2: Angle from target (directional vector: pitch, yaw, roll)
- S3: Obstacles - Coordinates (set of positions from voxel grid)
"""
@model function drone_inference_model()
    
    # Latent state priors
    s₁ ~ Normal(0, 10)                          # S1: Distance (meters)
    s₂ ~ MvNormal(zeros(3), 5.0 * I(3))         # S2: 3D angle vector
    s₃ ~ MvNormal(zeros(6), 10.0 * I(6))        # S3: Voxel-based obstacle layout (example: 3 nearest 2D points)

    # Observations modeled as noisy versions of these states
    o₁ ~ Normal(s₁, 1.0)                        # Noisy distance sensor
    o₂ ~ MvNormal(s₂, 0.5 * I(3))               # Noisy camera vector
    o₃ ~ MvNormal(s₃, 2.0 * I(6))               # Noisy voxel perception
end

"""
    run_inference(obs::NamedTuple)

Run inference on the generative model given observations.
Returns the posterior distributions of the latent states.
"""
function run_inference(obs::NamedTuple)
    model = drone_inference_model()
    result = infer(model, obs)
    return (
        s1 = mean(result.posteriors[:s₁]),
        s2 = mean(result.posteriors[:s₂]),
        s3 = mean(result.posteriors[:s₃]),
        s1_var = var(result.posteriors[:s₁]),
        s2_var = var(result.posteriors[:s₂]),
        s3_var = var(result.posteriors[:s₃])
    )
end

"""
    prepare_observations(state, noise_level=0.1)

Convert a drone state into observations for the inference model.
Adds noise to simulate sensor uncertainty.
"""
function prepare_observations(state, noise_level=0.1)
    # S1: Distance to target with noise
    distance = state.distance_to_target + randn() * noise_level
    
    # S2: Direction vector to target with noise
    direction = state.direction_to_target + SVector{3, Float64}(randn() * noise_level, randn() * noise_level, randn() * noise_level)
    
    # S3: Obstacle positions (simplified to 6D vector - 2 obstacles)
    # Take the two nearest obstacles or use zeros
    obstacle_vector = zeros(6)
    for i in 1:min(2, length(state.obstacles))
        obstacle = state.obstacles[i]
        obstacle_vector[3*i-2:3*i] .= [
            obstacle[1] + randn() * noise_level, 
            obstacle[2] + randn() * noise_level, 
            obstacle[3] + randn() * noise_level
        ]
    end
    
    # Return observations
    return (
        o₁ = distance,
        o₂ = direction,
        o₃ = obstacle_vector
    )
end

"""
    update_belief_states(state, previous_beliefs=nothing; noise_level=0.1)

Update beliefs about latent states based on current observations.
"""
function update_belief_states(state, previous_beliefs=nothing; noise_level=0.1)
    # Create observations from current state
    observations = prepare_observations(state, noise_level)
    
    # Run inference to get updated beliefs
    updated_beliefs = run_inference(observations)
    
    # If we have previous beliefs, we can incorporate them here
    if !isnothing(previous_beliefs)
        # Simple Bayesian update (could be more sophisticated)
        alpha = 0.7  # Weight for new observations vs prior beliefs
        
        # Weighted average of new and previous beliefs
        combined_beliefs = (
            s1 = alpha * updated_beliefs.s1 + (1-alpha) * previous_beliefs.s1,
            s2 = alpha * updated_beliefs.s2 + (1-alpha) * previous_beliefs.s2,
            s3 = alpha * updated_beliefs.s3 + (1-alpha) * previous_beliefs.s3,
            s1_var = (alpha * updated_beliefs.s1_var + (1-alpha) * previous_beliefs.s1_var),
            s2_var = (alpha * updated_beliefs.s2_var + (1-alpha) * previous_beliefs.s2_var),
            s3_var = (alpha * updated_beliefs.s3_var + (1-alpha) * previous_beliefs.s3_var)
        )
        
        return combined_beliefs
    end
    
    return updated_beliefs
end

"""
    calculate_efe(state, beliefs, action; 
                 pragmatic_weight=1.0, epistemic_weight=0.2, risk_weight=2.0)

Calculate the Expected Free Energy for a potential action.
Balances pragmatic value (reaching target) with epistemic value (exploration).
"""
function calculate_efe(state, beliefs, action; 
                      pragmatic_weight=1.0, epistemic_weight=0.2, risk_weight=2.0)
    # Predict next position after applying action
    next_position = state.position + action
    
    # Calculate predicted distance to target (pragmatic value)
    predicted_distance = norm(state.target_position - next_position)
    distance_improvement = state.distance_to_target - predicted_distance
    
    # Calculate predicted distance to obstacles (risk value)
    min_obstacle_distance = Inf
    for obstacle in state.obstacles
        dist = norm(obstacle - next_position)
        if dist < min_obstacle_distance
            min_obstacle_distance = dist
        end
    end
    
    # Calculate safety risk
    safety_margin = 1.5
    risk_value = 0.0
    if min_obstacle_distance < safety_margin
        risk_value = (safety_margin - min_obstacle_distance)^2
    end
    
    # Calculate epistemic value (reduction in uncertainty)
    # In a real implementation, we would predict how the action affects uncertainty
    # For simplicity, we'll use a heuristic based on current uncertainty
    current_uncertainty = beliefs.s1_var + sum(beliefs.s2_var) + sum(beliefs.s3_var)
    
    # Simple heuristic: actions that move into unexplored areas have higher epistemic value
    # Here we'll use a placeholder calculation
    # In a real implementation, you would compute how the action changes the uncertainty
    epistemic_value = current_uncertainty * min(1.0, norm(action))
    
    # Calculate Expected Free Energy components:
    # 1. Pragmatic value (lower distance to target is better)
    pragmatic_term = -pragmatic_weight * distance_improvement
    
    # 2. Risk value (avoid obstacles)
    risk_term = risk_weight * risk_value
    
    # 3. Epistemic value (explore to reduce uncertainty)
    epistemic_term = -epistemic_weight * epistemic_value
    
    # Total Expected Free Energy (lower is better)
    efe = pragmatic_term + risk_term + epistemic_term
    
    return efe
end

end # module 