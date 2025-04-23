module Inference

export StateBeliefs, initialize_beliefs, update_beliefs, predict_next_state, compute_free_energy

using RxInfer
using LinearAlgebra
using StaticArrays
using ..State

"""
struct StateBeliefs

A structure to store beliefs about the drone's state
"""
struct StateBeliefs
    # State 1: Distance from target (scalar)
    distance_mean::Float64
    distance_var::Float64
    
    # State 2: Angle/direction vector to target (3D vector)
    direction_mean::SVector{3, Float64}
    direction_var::SVector{3, Float64}
    
    # State 3: Obstacle positions (voxel grid representation)
    obstacle_positions_mean::Vector{SVector{3, Float64}}
    obstacle_positions_var::Vector{SVector{3, Float64}}
    
    # Additional state information for internal use
    position_mean::SVector{3, Float64}
    position_var::SVector{3, Float64}
    target_position_mean::SVector{3, Float64}
    target_position_var::SVector{3, Float64}
    
    # Dynamics parameters
    process_noise::Float64
    observation_noise::Float64
end

"""
    initialize_beliefs(initial_state::State)

Initialize the beliefs over the drone state with reasonable priors.
"""
function initialize_beliefs(initial_state::State)
    # State 1: Distance from target
    distance_mean = initial_state.distance_to_target
    distance_var = 0.1
    
    # State 2: Direction vector to target
    direction = normalize(initial_state.target_position - initial_state.position)
    direction_mean = direction
    direction_var = SVector{3, Float64}(0.1, 0.1, 0.1)
    
    # State 3: Obstacle positions in voxel grid
    obstacle_positions_mean = initial_state.obstacles
    obstacle_positions_var = [SVector{3, Float64}(0.2, 0.2, 0.2) for _ in initial_state.obstacles]
    
    # Additional state information
    position_mean = initial_state.position
    position_var = SVector{3, Float64}(0.1, 0.1, 0.1)
    
    target_position_mean = initial_state.target_position
    target_position_var = SVector{3, Float64}(0.1, 0.1, 0.1)
    
    process_noise = 0.01
    observation_noise = 0.05
    
    return StateBeliefs(
        distance_mean,
        distance_var,
        direction_mean,
        direction_var,
        obstacle_positions_mean,
        obstacle_positions_var,
        position_mean,
        position_var,
        target_position_mean,
        target_position_var,
        process_noise,
        observation_noise
    )
end

"""
    DroneInferenceModel

Define the generative model for active inference.
"""
@model function drone_inference_model()
    # Latent state priors
    s₁ ~ Normal(0, 10)                     # S1: Distance (meters)
    s₂ ~ MvNormal(zeros(3), 5.0 * I(3))    # S2: 3D angle/direction vector
    s₃ ~ MvNormal(zeros(6), 10.0 * I(6))   # S3: Voxel-based obstacle layout (simplified as 2 obstacle points in 3D)
    
    # Observations modeled as noisy versions of states
    o₁ ~ Normal(s₁, 1.0)                   # Noisy distance sensor
    o₂ ~ MvNormal(s₂, 0.5 * I(3))          # Noisy camera vector
    o₃ ~ MvNormal(s₃, 2.0 * I(6))          # Noisy voxel perception
end

"""
    run_inference(observations::NamedTuple)

Run Bayesian inference on the generative model given observations.
"""
function run_inference(observations::NamedTuple)
    model = drone_inference_model()
    result = infer(model, observations)
    
    return (
        s1 = mean(result.posteriors[:s₁]),
        s2 = mean(result.posteriors[:s₂]),
        s3 = mean(result.posteriors[:s₃])
    )
end

"""
    update_beliefs(beliefs::StateBeliefs, current_state::State, action::AbstractVector)

Update beliefs about the state using variational free energy minimization.
"""
function update_beliefs(beliefs::StateBeliefs, current_state::State, action::AbstractVector)
    # Update position based on action
    new_position_mean = beliefs.position_mean + action
    new_position_var = beliefs.position_var .+ beliefs.process_noise
    
    # Create observations for the inference model
    # State 1: Distance from target
    observed_distance = current_state.distance_to_target
    
    # State 2: Direction vector to target
    observed_direction = normalize(current_state.target_position - current_state.position)
    
    # State 3: Obstacle positions (simplified for this example - using first two obstacles or zeros)
    observed_obstacles = Vector{Float64}()
    for i in 1:min(2, length(current_state.obstacles))
        append!(observed_obstacles, current_state.obstacles[i])
    end
    
    # Pad with zeros if needed
    while length(observed_obstacles) < 6
        push!(observed_obstacles, 0.0)
    end
    
    # Run inference with observations
    observations = (
        o₁ = observed_distance,
        o₂ = observed_direction,
        o₃ = observed_obstacles
    )
    
    inference_result = run_inference(observations)
    
    # Extract inferred states
    inferred_distance = inference_result.s1
    inferred_direction = SVector{3, Float64}(inference_result.s2)
    
    # Extract obstacle information
    obstacle_vector = inference_result.s3
    inferred_obstacles = Vector{SVector{3, Float64}}()
    
    # Reconstruct obstacle positions
    for i in 1:2
        obstacle_pos = SVector{3, Float64}(
            obstacle_vector[3*i-2], 
            obstacle_vector[3*i-1], 
            obstacle_vector[3*i]
        )
        push!(inferred_obstacles, obstacle_pos)
    end
    
    # Additional obstacles from observation
    for i in 3:length(current_state.obstacles)
        push!(inferred_obstacles, current_state.obstacles[i])
    end
    
    # Return updated beliefs
    return StateBeliefs(
        inferred_distance,
        beliefs.distance_var,
        inferred_direction,
        beliefs.direction_var,
        inferred_obstacles,
        [SVector{3, Float64}(0.2, 0.2, 0.2) for _ in inferred_obstacles],
        new_position_mean,
        new_position_var,
        beliefs.target_position_mean,
        beliefs.target_position_var,
        beliefs.process_noise,
        beliefs.observation_noise
    )
end

"""
    predict_next_state(beliefs::StateBeliefs, action::AbstractVector; steps=1)

Predict the future state given current beliefs and an action.
"""
function predict_next_state(beliefs::StateBeliefs, action::AbstractVector; steps=1)
    # Initialize state prediction
    predicted_positions = Vector{SVector{3, Float64}}(undef, steps + 1)
    predicted_positions[1] = beliefs.position_mean
    
    for i in 1:steps
        # Motion model - simple linear dynamics
        new_position = predicted_positions[i] + action
        
        # Store prediction
        predicted_positions[i+1] = new_position
    end
    
    return predicted_positions
end

"""
    compute_free_energy(state::State, beliefs::StateBeliefs, action::AbstractVector)

Compute the expected free energy of a potential action given the current state and beliefs.
Incorporates pragmatic (target-seeking) and epistemic (uncertainty-reducing) value.
"""
function compute_free_energy(state::State, beliefs::StateBeliefs, action::AbstractVector)
    # Predict next position after taking this action
    predicted_position = beliefs.position_mean + action
    
    # Calculate distance to target after action
    predicted_distance = norm(predicted_position - beliefs.target_position_mean)
    
    # Calculate distance to obstacles after action
    obstacle_distances = [norm(predicted_position - obstacle) for obstacle in beliefs.obstacle_positions_mean]
    min_obstacle_distance = isempty(obstacle_distances) ? Inf : minimum(obstacle_distances)
    
    # Pragmatic value: lower distance to target is better
    pragmatic_value = -predicted_distance  # Negative because we want to minimize free energy
    
    # Risk value: penalize getting too close to obstacles
    safety_margin = 1.5  # Minimum safe distance from obstacles
    risk_value = 0.0
    if min_obstacle_distance < safety_margin
        risk_value = -10.0 * (safety_margin - min_obstacle_distance)^2
    end
    
    # Epistemic value: preference for reducing uncertainty
    # Higher uncertainty = higher epistemic value for exploration
    position_uncertainty = sum(beliefs.position_var)
    obstacle_uncertainty = length(beliefs.obstacle_positions_mean) > 0 ? 
                          sum(sum(var) for var in beliefs.obstacle_positions_var) : 1.0
    
    # Weight epistemic value (exploration) based on overall uncertainty
    epistemic_weight = 0.2  # Adjust to balance exploration vs. exploitation
    epistemic_value = -epistemic_weight * (position_uncertainty + obstacle_uncertainty)
    
    # Total expected free energy (negative because we want to minimize free energy)
    expected_free_energy = -(pragmatic_value + risk_value + epistemic_value)
    
    return expected_free_energy
end

end # module
