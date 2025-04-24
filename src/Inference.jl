module Inference

export DroneBeliefs, initialize_beliefs, update_beliefs_vfe, calculate_efe, expected_state

using Distributions
using LinearAlgebra
using StaticArrays
using RxInfer

# Constants for state space dimensions
const MAX_DISTANCE = 50      # Maximum distance in meters
const NUM_AZIMUTH = 24       # 24 azimuth bins (15° each)
const NUM_ELEVATION = 12     # 12 elevation bins (15° each)
const NUM_SUITABILITY = 11   # 11 suitability bins (0.1 each)

"""
DroneBeliefs represents probabilistic beliefs over the state space
"""
struct DroneBeliefs
    # Belief distributions over state variables
    distance_belief::Vector{Float64}      # P(distance)
    azimuth_belief::Vector{Float64}       # P(azimuth)
    elevation_belief::Vector{Float64}     # P(elevation)
    suitability_belief::Vector{Float64}   # P(suitability)
    
    # Joint belief (if factorized, this is just for convenience)
    joint_belief::Array{Float64, 4}       # P(distance, azimuth, elevation, suitability)
    
    # Observation likelihood models
    distance_likelihood::Matrix{Float64}    # P(obs_distance | s_distance)
    azimuth_likelihood::Matrix{Float64}     # P(obs_azimuth | s_azimuth)
    elevation_likelihood::Matrix{Float64}   # P(obs_elevation | s_elevation)
    suitability_likelihood::Matrix{Float64} # P(obs_obstacle | s_suitability)
end

"""
    initialize_beliefs()

Initialize beliefs with uniform priors over state space.
"""
function initialize_beliefs()
    # Create uniform priors
    distance_prior = ones(MAX_DISTANCE) / MAX_DISTANCE
    azimuth_prior = ones(NUM_AZIMUTH) / NUM_AZIMUTH
    elevation_prior = ones(NUM_ELEVATION) / NUM_ELEVATION
    suitability_prior = ones(NUM_SUITABILITY) / NUM_SUITABILITY
    
    # Initialize joint belief
    joint_prior = ones(MAX_DISTANCE, NUM_AZIMUTH, NUM_ELEVATION, NUM_SUITABILITY)
    joint_prior = joint_prior / sum(joint_prior)
    
    # Initialize likelihood matrices
    # For distance: Gaussian centered at true distance, with sigma=1m
    distance_likelihood = zeros(MAX_DISTANCE, MAX_DISTANCE)
    for i in 1:MAX_DISTANCE
        for j in 1:MAX_DISTANCE
            # Gaussian likelihood centered at actual state
            distance_likelihood[i, j] = exp(-0.5 * ((i - j) / 1.0)^2)
        end
        # Normalize
        if sum(distance_likelihood[i, :]) > 0
            distance_likelihood[i, :] = distance_likelihood[i, :] / sum(distance_likelihood[i, :])
        end
    end
    
    # For azimuth: Von Mises (approximated by wrapped Gaussian) with kappa=8
    azimuth_likelihood = zeros(NUM_AZIMUTH, NUM_AZIMUTH)
    for i in 1:NUM_AZIMUTH
        for j in 1:NUM_AZIMUTH
            # Calculate circular distance (shortest distance on circle)
            diff = min(abs(i - j), NUM_AZIMUTH - abs(i - j))
            # Wrapped Gaussian 
            azimuth_likelihood[i, j] = exp(-0.5 * (diff / 1.5)^2)
        end
        # Normalize
        azimuth_likelihood[i, :] = azimuth_likelihood[i, :] / sum(azimuth_likelihood[i, :])
    end
    
    # For elevation: similar to azimuth
    elevation_likelihood = zeros(NUM_ELEVATION, NUM_ELEVATION)
    for i in 1:NUM_ELEVATION
        for j in 1:NUM_ELEVATION
            # Calculate circular distance (shortest distance on circle)
            diff = min(abs(i - j), NUM_ELEVATION - abs(i - j))
            # Wrapped Gaussian 
            elevation_likelihood[i, j] = exp(-0.5 * (diff / 1.5)^2)
        end
        # Normalize
        elevation_likelihood[i, :] = elevation_likelihood[i, :] / sum(elevation_likelihood[i, :])
    end
    
    # For suitability: based on obstacle distance
    suitability_likelihood = zeros(NUM_SUITABILITY, NUM_SUITABILITY)
    for i in 1:NUM_SUITABILITY
        # Calculate expected obstacle distance for each suitability level
        for j in 1:NUM_SUITABILITY
            # Closer to diagonal = higher likelihood
            diff = abs(i - j)
            suitability_likelihood[i, j] = exp(-diff)
        end
        # Normalize
        suitability_likelihood[i, :] = suitability_likelihood[i, :] / sum(suitability_likelihood[i, :])
    end
    
    return DroneBeliefs(
        distance_prior,
        azimuth_prior,
        elevation_prior,
        suitability_prior,
        joint_prior,
        distance_likelihood,
        azimuth_likelihood,
        elevation_likelihood,
        suitability_likelihood
    )
end

"""
    calculate_suitability(obstacle_distance::Float64; beta::Float64=3.0)

Calculate suitability measure (0-1) based on distance to nearest obstacle.
Uses exponential transformation with parameter beta.
"""
function calculate_suitability(obstacle_distance::Float64; beta::Float64=3.0)
    # Exponential transformation: higher penalty as you get closer to obstacle
    # Returns value in [0,1] where 1 is most suitable (far from obstacles)
    return min(1.0, exp(-beta / (obstacle_distance + 1e-3)))
end

"""
    discretize_state(distance::Float64, azimuth::Float64, 
                    elevation::Float64, suitability::Float64)

Discretize continuous values according to our state space definition.
Returns tuple of discretized (integer) values.
"""
function discretize_state(distance::Float64, azimuth::Float64, 
                         elevation::Float64, suitability::Float64)
    # Discretize distance by 1m increments
    distance_idx = max(1, min(MAX_DISTANCE, Int(floor(distance))))
    
    # Discretize azimuth by 15° increments (24 possible values)
    # Convert from radians to 0-23 index (15° = π/12 radians)
    azimuth_normalized = mod(azimuth, 2π)  # Normalize to [0, 2π)
    azimuth_idx = Int(floor(azimuth_normalized / (π/12))) + 1
    azimuth_idx = max(1, min(NUM_AZIMUTH, azimuth_idx))
    
    # Discretize elevation by 15° increments (12 possible values)
    # Convert from radians to 0-11 index
    elevation_normalized = clamp(elevation, -π/2, π/2)  # Clamp to [-π/2, π/2]
    elevation_idx = Int(floor((elevation_normalized + π/2) / (π/12))) + 1
    elevation_idx = max(1, min(NUM_ELEVATION, elevation_idx))
    
    # Discretize suitability by 0.1 increments (0 to 1.0)
    suitability_idx = Int(floor(suitability * 10)) + 1
    suitability_idx = max(1, min(NUM_SUITABILITY, suitability_idx))
    
    return distance_idx, azimuth_idx, elevation_idx, suitability_idx
end

"""
    update_beliefs_vfe(beliefs::DroneBeliefs, observation)

Update beliefs using variational free energy minimization.
Returns updated beliefs.
"""
function update_beliefs_vfe(beliefs::DroneBeliefs, observation)
    # Extract observation data
    obs_distance = observation.obs_distance
    obs_azimuth = observation.obs_azimuth
    obs_elevation = observation.obs_elevation
    obs_nearest_obstacle = observation.obs_nearest_obstacle
    
    # Calculate suitability from nearest obstacle distance
    suitability = calculate_suitability(obs_nearest_obstacle)
    
    # Discretize observation
    obs_distance_idx, obs_azimuth_idx, obs_elevation_idx, obs_suitability_idx = 
        discretize_state(obs_distance, obs_azimuth, obs_elevation, suitability)
    
    # Extract likelihood vectors for this observation
    distance_lh = beliefs.distance_likelihood[obs_distance_idx, :]
    azimuth_lh = beliefs.azimuth_likelihood[obs_azimuth_idx, :]
    elevation_lh = beliefs.elevation_likelihood[obs_elevation_idx, :]
    suitability_lh = beliefs.suitability_likelihood[obs_suitability_idx, :]
    
    # VFE minimization (simplified as sequential Bayesian updates)
    # In full active inference, this would be iterative with proper message passing
    
    # Prior beliefs
    distance_prior = beliefs.distance_belief
    azimuth_prior = beliefs.azimuth_belief
    elevation_prior = beliefs.elevation_belief
    suitability_prior = beliefs.suitability_belief
    
    # Bayesian updates (factorized approximate posterior)
    distance_posterior = distance_lh .* distance_prior
    azimuth_posterior = azimuth_lh .* azimuth_prior
    elevation_posterior = elevation_lh .* elevation_prior 
    suitability_posterior = suitability_lh .* suitability_prior
    
    # Normalize
    distance_posterior = distance_posterior / sum(distance_posterior)
    azimuth_posterior = azimuth_posterior / sum(azimuth_posterior)
    elevation_posterior = elevation_posterior / sum(elevation_posterior)
    suitability_posterior = suitability_posterior / sum(suitability_posterior)
    
    # Reconstruct joint belief (factorized approximation)
    joint_belief = ones(MAX_DISTANCE, NUM_AZIMUTH, NUM_ELEVATION, NUM_SUITABILITY)
    
    for d in 1:MAX_DISTANCE, a in 1:NUM_AZIMUTH, e in 1:NUM_ELEVATION, s in 1:NUM_SUITABILITY
        joint_belief[d, a, e, s] = distance_posterior[d] * 
                                 azimuth_posterior[a] * 
                                 elevation_posterior[e] * 
                                 suitability_posterior[s]
    end
    
    # Return updated beliefs
    return DroneBeliefs(
        distance_posterior,
        azimuth_posterior,
        elevation_posterior,
        suitability_posterior,
        joint_belief,
        beliefs.distance_likelihood,
        beliefs.azimuth_likelihood,
        beliefs.elevation_likelihood,
        beliefs.suitability_likelihood
    )
end

"""
    expected_state(beliefs::DroneBeliefs)

Calculate expected values for each state dimension.
"""
function expected_state(beliefs::DroneBeliefs)
    # Calculate expected values
    exp_distance = sum(beliefs.distance_belief .* (1:MAX_DISTANCE))
    
    # For circular variables like azimuth and elevation, we need vector averaging
    azimuth_x = 0.0
    azimuth_y = 0.0
    for i in 1:NUM_AZIMUTH
        angle = (i - 1) * (2π / NUM_AZIMUTH)
        azimuth_x += cos(angle) * beliefs.azimuth_belief[i]
        azimuth_y += sin(angle) * beliefs.azimuth_belief[i]
    end
    exp_azimuth = atan(azimuth_y, azimuth_x)
    
    elevation_x = 0.0
    elevation_y = 0.0
    for i in 1:NUM_ELEVATION
        angle = (i - 1) * (π / (NUM_ELEVATION-1)) - π/2  # -π/2 to π/2
        elevation_x += cos(angle) * beliefs.elevation_belief[i]
        elevation_y += sin(angle) * beliefs.elevation_belief[i]
    end
    exp_elevation = atan(elevation_y, elevation_x)
    
    exp_suitability = sum(beliefs.suitability_belief .* ((0:10) ./ 10))
    
    return (
        distance = exp_distance,
        azimuth = exp_azimuth,
        elevation = exp_elevation,
        suitability = exp_suitability
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
    # Calculate predicted position after action
    predicted_position = state.position + action
    
    # Calculate predicted distance to target
    target_vector = state.target_position - predicted_position
    predicted_distance = norm(target_vector)
    
    # Calculate improvement in distance (negative because we want to minimize EFE)
    distance_improvement = state.raw_distance - predicted_distance
    pragmatic_value = -pragmatic_weight * distance_improvement
    
    # Get expected state values
    expected = expected_state(beliefs)
    
    # Calculate risk value (obstacle avoidance)
    risk_value = 0.0
    safety_distance = 1.5
    
    # Check if action brings us too close to obstacles
    for (voxel_coords, _) in state.voxel_grid
        # Convert voxel to world coordinates (approximate center)
        # Assuming 0.5 voxel size
        voxel_pos = SVector{3, Float64}(
            voxel_coords[1] * 0.5 + 0.25,
            voxel_coords[2] * 0.5 + 0.25,
            voxel_coords[3] * 0.5 + 0.25
        )
        
        # Calculate distance to obstacle
        obstacle_distance = norm(predicted_position - voxel_pos)
        
        # Add penalty if too close
        if obstacle_distance < safety_distance
            risk_value += risk_weight * (safety_distance - obstacle_distance)^2
        end
    end
    
    # Calculate entropy of beliefs (measure of uncertainty)
    dist_entropy = -sum(beliefs.distance_belief .* log.(beliefs.distance_belief .+ 1e-10))
    azim_entropy = -sum(beliefs.azimuth_belief .* log.(beliefs.azimuth_belief .+ 1e-10))
    elev_entropy = -sum(beliefs.elevation_belief .* log.(beliefs.elevation_belief .+ 1e-10))
    suit_entropy = -sum(beliefs.suitability_belief .* log.(beliefs.suitability_belief .+ 1e-10))
    
    # Total entropy
    total_entropy = dist_entropy + azim_entropy + elev_entropy + suit_entropy
    
    # Epistemic value - we want to reduce uncertainty where it matters
    # Scale by action magnitude (prefer larger movements when uncertain)
    action_mag = norm(action)
    epistemic_value = -epistemic_weight * total_entropy * action_mag
    
    # Total Expected Free Energy (lower is better)
    efe = pragmatic_value + risk_value + epistemic_value
    
    return efe
end

end # module
