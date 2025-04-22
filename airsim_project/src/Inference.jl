module Inference

export StateBeliefs, initialize_beliefs, update_beliefs, predict_next_state

using RxInfer
using LinearAlgebra
using StaticArrays
using ..State

"""
struct StateBeliefs

A structure to store beliefs about the drone's state
"""
struct StateBeliefs
    # Position and orientation beliefs (mean and variance)
    position_mean::SVector{3, Float64}
    position_var::SVector{3, Float64}
    orientation_mean::SVector{3, Float64}
    orientation_var::SVector{3, Float64}
    
    # Target beliefs
    target_position_mean::SVector{3, Float64}
    target_position_var::SVector{3, Float64}
    
    # Obstacle beliefs
    obstacle_positions_mean::Vector{SVector{3, Float64}}
    obstacle_positions_var::Vector{SVector{3, Float64}}
    
    # Dynamics parameters
    process_noise::Float64
    observation_noise::Float64
end

"""
    initialize_beliefs(initial_state::State)

Initialize the beliefs over the drone state with reasonable priors.
"""
function initialize_beliefs(initial_state::State)
    position_mean = initial_state.position
    position_var = SVector{3, Float64}(0.1, 0.1, 0.1)
    
    orientation_mean = initial_state.orientation
    orientation_var = SVector{3, Float64}(0.1, 0.1, 0.1)
    
    target_position_mean = initial_state.target_position
    target_position_var = SVector{3, Float64}(0.1, 0.1, 0.1)
    
    obstacle_positions_mean = initial_state.obstacles
    obstacle_positions_var = [SVector{3, Float64}(0.2, 0.2, 0.2) for _ in initial_state.obstacles]
    
    process_noise = 0.01
    observation_noise = 0.05
    
    return StateBeliefs(
        position_mean,
        position_var,
        orientation_mean,
        orientation_var,
        target_position_mean,
        target_position_var,
        obstacle_positions_mean,
        obstacle_positions_var,
        process_noise,
        observation_noise
    )
end

"""
    update_beliefs(beliefs::StateBeliefs, current_state::State, action::AbstractVector)

Update beliefs about the state using variational free energy minimization.
"""
function update_beliefs(beliefs::StateBeliefs, current_state::State, action::AbstractVector)
    # Create the factor graph for inference
    g = FactorGraph()
    
    # Prior over position (from previous step)
    @RV position ~ MvNormal(beliefs.position_mean, diagm(beliefs.position_var))
    
    # Prior over orientation
    @RV orientation ~ MvNormal(beliefs.orientation_mean, diagm(beliefs.orientation_var))
    
    # Motion model (prediction)
    predicted_position_mean = beliefs.position_mean + action
    predicted_position_var = beliefs.position_var .+ beliefs.process_noise
    
    @RV predicted_position ~ MvNormal(predicted_position_mean, diagm(predicted_position_var))
    
    # Observation model (correction)
    position_observation = current_state.position
    @RV position_obs ~ MvNormal(position, diagm(SVector{3, Float64}(beliefs.observation_noise, beliefs.observation_noise, beliefs.observation_noise)))
    
    # Data binding
    Data(position_obs, position_observation)
    
    # Run variational inference
    result = inference(
        g, 
        [position, orientation, predicted_position],
        vi_params = VariationalInferenceParameters(
            free_energy_tol = 1e-6,
            max_iter = 100
        )
    )
    
    # Extract posterior beliefs
    position_posterior = getmarginal(result, position)
    position_mean = mean(position_posterior)
    position_var = var(position_posterior)
    
    orientation_posterior = getmarginal(result, orientation)
    orientation_mean = mean(orientation_posterior)
    orientation_var = var(orientation_posterior)
    
    # Update obstacle beliefs
    obstacle_positions_mean = current_state.obstacles
    obstacle_positions_var = [SVector{3, Float64}(0.2, 0.2, 0.2) for _ in current_state.obstacles]
    
    # Return updated beliefs
    return StateBeliefs(
        position_mean,
        position_var,
        orientation_mean,
        orientation_var,
        beliefs.target_position_mean,
        beliefs.target_position_var,
        obstacle_positions_mean,
        obstacle_positions_var,
        beliefs.process_noise,
        beliefs.observation_noise
    )
end

"""
    predict_next_state(beliefs::StateBeliefs, action::AbstractVector; steps=1)

Predict the future state given current beliefs and an action.
"""
function predict_next_state(beliefs::StateBeliefs, action::AbstractVector; steps=1)
    # Create the factor graph for prediction
    g = FactorGraph()
    
    # Prior over position
    @RV position ~ MvNormal(beliefs.position_mean, diagm(beliefs.position_var))
    
    # Initialize state prediction
    predicted_positions = Vector{SVector{3, Float64}}(undef, steps + 1)
    predicted_positions[1] = beliefs.position_mean
    
    for i in 1:steps
        # Motion model
        new_position_mean = predicted_positions[i] + action
        new_position_var = beliefs.position_var .+ (beliefs.process_noise * i)
        
        @RV new_position ~ MvNormal(new_position_mean, diagm(new_position_var))
        
        # Store prediction
        predicted_positions[i+1] = new_position_mean
    end
    
    return predicted_positions
end

"""
    compute_free_energy(state::State, beliefs::StateBeliefs, action::AbstractVector)

Compute the free energy of a potential action given the current state and beliefs.
"""
function compute_free_energy(state::State, beliefs::StateBeliefs, action::AbstractVector)
    # Predict next state
    predicted_position = beliefs.position_mean + action
    
    # Calculate distance to obstacles after action
    obstacle_distances = [norm(predicted_position - obstacle) for obstacle in state.obstacles]
    min_obstacle_distance = isempty(obstacle_distances) ? Inf : minimum(obstacle_distances)
    
    # Calculate distance to target after action
    distance_to_target = norm(predicted_position - beliefs.target_position_mean)
    
    # Compute free energy components
    accuracy = -distance_to_target  # Negative distance as we want to minimize distance
    complexity = 0.0
    
    # Add penalty for getting too close to obstacles
    safety_margin = 1.0  # Minimum safe distance from obstacles
    if min_obstacle_distance < safety_margin
        complexity += 10.0 * (safety_margin - min_obstacle_distance)^2
    end
    
    # Total free energy (to be minimized)
    free_energy = -accuracy + complexity
    
    return free_energy
end

end # module
