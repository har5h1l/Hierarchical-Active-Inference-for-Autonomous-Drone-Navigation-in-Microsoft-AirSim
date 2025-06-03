module Inference

using StaticArrays
using RxInfer
using LinearAlgebra
using ..StateSpace

export DroneBeliefs, initialize_beliefs, update_beliefs!, expected_state
export serialize_beliefs, deserialize_beliefs, calculate_vfe, discretize_observation

"""
    DroneBeliefs

Container for factorized probabilistic beliefs about the drone state.
Uses categorical distributions for each state factor.
"""
mutable struct DroneBeliefs
    # Factorized categorical beliefs q(s) = q(s_location) × q(s_angle) × q(s_suitability)
    location_belief::Vector{Float64}      # q(s_location) - distance discretized
    angle_belief::Vector{Float64}         # q(s_angle) - combined azimuth/elevation  
    suitability_belief::Vector{Float64}   # q(s_suitability) - environmental quality
    
    # Discretization ranges for categorical states
    location_bins::Vector{Float64}        # Distance bins [0, max_distance]
    angle_bins::Vector{Float64}           # Angle bins [-π, π] for azimuth
    elevation_bins::Vector{Float64}       # Elevation bins [-π/2, π/2]
    suitability_bins::Vector{Float64}     # Suitability bins [0, 1]
    
    # Transition matrices B (learned from experience)
    B_location::Matrix{Float64}           # P(s_location_t | s_location_{t-1})
    B_angle::Matrix{Float64}              # P(s_angle_t | s_angle_{t-1})
    B_suitability::Matrix{Float64}        # P(s_suitability_t | s_suitability_{t-1})
    
    # Observation matrices A (sensory model)
    A_location::Matrix{Float64}           # P(o_distance | s_location)
    A_angle::Matrix{Float64}              # P(o_angle | s_angle)  
    A_suitability::Matrix{Float64}        # P(o_suitability | s_suitability)
    
    # Store sensory data for observation model updates
    voxel_grid::Vector{SVector{3, Float64}}
    
    # VFE computation cache
    vfe_cache::Dict{String, Float64}
end

"""
    initialize_beliefs(state::StateSpace.DroneState; num_bins=50)

Initialize uniform factorized beliefs and generative model matrices.
"""
function initialize_beliefs(state::StateSpace.DroneState; num_bins=50, voxel_grid=Vector{SVector{3, Float64}}(), obstacle_density=0.0)
    # Define categorical state discretizations
    max_distance = 120.0
    location_bins = collect(range(0.0, stop=max_distance, length=num_bins))
    angle_bins = collect(range(-π, stop=π, length=num_bins))
    elevation_bins = collect(range(-π/2, stop=π/2, length=num_bins))
    suitability_bins = collect(range(0.0, stop=1.0, length=num_bins))
    
    # Initialize uniform categorical beliefs
    location_belief = ones(num_bins) / num_bins
    angle_belief = ones(num_bins) / num_bins  
    suitability_belief = ones(num_bins) / num_bins
    
    # Initialize transition matrices B (identity + noise)
    B_location = initialize_transition_matrix(num_bins, 0.9)      # High self-transition
    B_angle = initialize_transition_matrix(num_bins, 0.8)         # Moderate self-transition
    B_suitability = initialize_transition_matrix(num_bins, 0.95)  # Very stable
    
    # Initialize observation matrices A (Gaussian observation model)
    A_location = initialize_observation_matrix(num_bins, location_bins, "distance")
    A_angle = initialize_observation_matrix(num_bins, angle_bins, "angle")
    A_suitability = initialize_observation_matrix(num_bins, suitability_bins, "suitability")
    
    # Create beliefs object
    beliefs = DroneBeliefs(
        location_belief,
        angle_belief, 
        suitability_belief,
        location_bins,
        angle_bins,
        elevation_bins,
        suitability_bins,
        B_location,
        B_angle,
        B_suitability,
        A_location,
        A_angle,
        A_suitability,
        voxel_grid,
        Dict{String, Float64}()
    )
    
    # Update with initial observation
    update_beliefs!(beliefs, state; obstacle_density=obstacle_density)
    
    return beliefs
end

"""
    initialize_transition_matrix(num_bins::Int, self_prob::Float64)

Initialize transition matrix B with diagonal dominance.
"""
function initialize_transition_matrix(num_bins::Int, self_prob::Float64)
    B = ones(num_bins, num_bins) * (1.0 - self_prob) / (num_bins - 1)
    for i in 1:num_bins
        B[i, i] = self_prob
    end
    return B
end

"""
    initialize_observation_matrix(num_bins::Int, bins::Vector{Float64}, obs_type::String)

Initialize observation matrix A with Gaussian observation model.
"""
function initialize_observation_matrix(num_bins::Int, bins::Vector{Float64}, obs_type::String)
    A = zeros(num_bins, num_bins)
    
    # Observation noise variance (depends on sensor type)
    if obs_type == "distance"
        σ = 2.0  # Distance sensor noise
    elseif obs_type == "angle" 
        σ = 0.2  # Angular noise
    else  # suitability
        σ = 0.1  # Suitability computation noise
    end
    
    for i in 1:num_bins  # observation bins
        for j in 1:num_bins  # state bins
            # Gaussian likelihood: P(obs_i | state_j)
            diff = bins[i] - bins[j]
            
            # Handle circular angles
            if obs_type == "angle" && abs(diff) > π
                diff = diff > 0 ? diff - 2π : diff + 2π
            end
            
            A[i, j] = exp(-0.5 * (diff / σ)^2)
        end
        # Normalize each row to be a proper probability distribution
        A[i, :] ./= sum(A[i, :])
    end
    
    return A
end

"""
    update_beliefs!(beliefs::DroneBeliefs, state::StateSpace.DroneState; kwargs...)

Variational Bayesian update using matrix-based VFE minimization.
"""
function update_beliefs!(beliefs::DroneBeliefs, state::StateSpace.DroneState; 
                        obstacle_density=0.0, voxel_grid=nothing, kwargs...)
    
    # Update voxel grid if provided
    if voxel_grid !== nothing
        empty!(beliefs.voxel_grid)
        append!(beliefs.voxel_grid, voxel_grid)
    end
    
    # Convert continuous observations to discrete observation indices
    obs_location_idx = discretize_observation(state.distance, beliefs.location_bins)
    obs_angle_idx = discretize_observation(state.azimuth, beliefs.angle_bins)
    obs_suitability_idx = discretize_observation(state.suitability, beliefs.suitability_bins)
    
    # Update suitability based on obstacle density and voxel data
    adjusted_suitability = calculate_adjusted_suitability(state.suitability, obstacle_density, beliefs.voxel_grid)
    obs_suitability_idx = discretize_observation(adjusted_suitability, beliefs.suitability_bins)
    
    # Variational message passing updates
    update_location_belief!(beliefs, obs_location_idx)
    update_angle_belief!(beliefs, obs_angle_idx)  
    update_suitability_belief!(beliefs, obs_suitability_idx)
    
    # Calculate and cache VFE
    vfe = calculate_vfe(beliefs, obs_location_idx, obs_angle_idx, obs_suitability_idx)
    beliefs.vfe_cache["current_vfe"] = vfe
    
    return beliefs
end

"""
    discretize_observation(value::Float64, bins::Vector{Float64})

Convert continuous observation to discrete bin index.
"""
function discretize_observation(value::Float64, bins::Vector{Float64})
    # Clamp value to bin range
    clamped_value = clamp(value, bins[1], bins[end])
    
    # Find closest bin
    distances = abs.(bins .- clamped_value)
    return argmin(distances)
end

"""
    calculate_adjusted_suitability(base_suitability::Float64, obstacle_density::Float64, 
                                  voxel_grid::Vector{SVector{3, Float64}})

Calculate suitability adjusted for obstacles and environmental factors.
"""
function calculate_adjusted_suitability(base_suitability::Float64, obstacle_density::Float64, 
                                       voxel_grid::Vector{SVector{3, Float64}})
    # Penalize suitability based on obstacle density
    density_penalty = obstacle_density * 0.8
    
    # Additional penalty for nearby obstacles (within 3m)
    if !isempty(voxel_grid)
        nearby_obstacles = count(v -> norm(v) < 3.0, voxel_grid)
        proximity_penalty = min(0.5, nearby_obstacles * 0.1)
    else
        proximity_penalty = 0.0
    end
    
    # Combined suitability (clamped to [0,1])
    adjusted = base_suitability * (1.0 - density_penalty - proximity_penalty)
    return clamp(adjusted, 0.0, 1.0)
end

"""
    update_location_belief!(beliefs::DroneBeliefs, obs_idx::Int)

Update location belief using variational message passing.
"""
function update_location_belief!(beliefs::DroneBeliefs, obs_idx::Int)
    # Prior from transition model: π = B^T * q_prev
    prior = beliefs.B_location' * beliefs.location_belief
    
    # Likelihood from observation model: L = A[obs_idx, :]
    likelihood = beliefs.A_location[obs_idx, :]
    
    # Posterior update: q_new ∝ prior ⊙ likelihood
    posterior = prior .* likelihood
    posterior ./= sum(posterior)  # Normalize
    
    # Temporal smoothing for stability
    α = 0.7  # Learning rate
    beliefs.location_belief .= α .* beliefs.location_belief .+ (1.0 - α) .* posterior
    
    # Ensure proper normalization
    beliefs.location_belief ./= sum(beliefs.location_belief)
end

"""
    update_angle_belief!(beliefs::DroneBeliefs, obs_idx::Int)

Update angle belief using variational message passing.
"""
function update_angle_belief!(beliefs::DroneBeliefs, obs_idx::Int)
    # Prior from transition model
    prior = beliefs.B_angle' * beliefs.angle_belief
    
    # Likelihood from observation model
    likelihood = beliefs.A_angle[obs_idx, :]
    
    # Posterior update with target preference weighting
    posterior = prior .* likelihood
    
    # Add target-seeking bias (prefer angles closer to 0)
    target_bias = exp.(-abs.(beliefs.angle_bins) ./ 0.5)  # Exponential decay from 0
    posterior .*= (1.0 .+ 0.3 .* target_bias)  # Moderate target bias
    
    posterior ./= sum(posterior)
    
    # Temporal smoothing
    α = 0.6  # Slightly faster learning for angles
    beliefs.angle_belief .= α .* beliefs.angle_belief .+ (1.0 - α) .* posterior
    beliefs.angle_belief ./= sum(beliefs.angle_belief)
end

"""
    update_suitability_belief!(beliefs::DroneBeliefs, obs_idx::Int)

Update suitability belief using variational message passing.
"""
function update_suitability_belief!(beliefs::DroneBeliefs, obs_idx::Int)
    # Prior from transition model
    prior = beliefs.B_suitability' * beliefs.suitability_belief
    
    # Likelihood from observation model
    likelihood = beliefs.A_suitability[obs_idx, :]
    
    # Posterior update with preference for high suitability
    posterior = prior .* likelihood
    
    # Add preference for higher suitability values
    suitability_preference = beliefs.suitability_bins .^ 1.5  # Quadratic preference
    posterior .*= (1.0 .+ 0.2 .* suitability_preference)
    
    posterior ./= sum(posterior)
    
    # Temporal smoothing
    α = 0.8  # Slower learning for suitability (more stable)
    beliefs.suitability_belief .= α .* beliefs.suitability_belief .+ (1.0 - α) .* posterior
    beliefs.suitability_belief ./= sum(beliefs.suitability_belief)
end

"""
    calculate_vfe(beliefs::DroneBeliefs, obs_location::Int, obs_angle::Int, obs_suitability::Int)

Calculate Variational Free Energy F = E_q[ln q(s) - ln p(o,s)].
"""
function calculate_vfe(beliefs::DroneBeliefs, obs_location::Int, obs_angle::Int, obs_suitability::Int)
    # Factorized entropy: H[q(s)] = H[q(s_loc)] + H[q(s_ang)] + H[q(s_suit)]
    entropy_location = -sum(beliefs.location_belief .* log.(beliefs.location_belief .+ 1e-10))
    entropy_angle = -sum(beliefs.angle_belief .* log.(beliefs.angle_belief .+ 1e-10))
    entropy_suitability = -sum(beliefs.suitability_belief .* log.(beliefs.suitability_belief .+ 1e-10))
    
    total_entropy = entropy_location + entropy_angle + entropy_suitability
    
    # Expected log-likelihood: E_q[ln p(o|s)]
    expected_ll_location = sum(beliefs.location_belief .* log.(beliefs.A_location[obs_location, :] .+ 1e-10))
    expected_ll_angle = sum(beliefs.angle_belief .* log.(beliefs.A_angle[obs_angle, :] .+ 1e-10))
    expected_ll_suitability = sum(beliefs.suitability_belief .* log.(beliefs.A_suitability[obs_suitability, :] .+ 1e-10))
    
    total_expected_ll = expected_ll_location + expected_ll_angle + expected_ll_suitability
    
    # Prior log-probability (assume uniform priors for simplicity)
    num_bins = length(beliefs.location_belief)
    prior_ll = -3 * log(num_bins)  # Log of uniform prior for 3 factors
    
    # VFE = -H[q(s)] - E_q[ln p(o|s)] - ln p(s)
    vfe = -total_entropy - total_expected_ll - prior_ll
    
    return vfe
end

"""
    expected_state(beliefs::DroneBeliefs)

Compute expected state from factorized beliefs.
"""
function expected_state(beliefs::DroneBeliefs)
    # Expected values from categorical beliefs
    exp_distance = sum(beliefs.location_belief .* beliefs.location_bins)
    exp_azimuth = weighted_average_circular(beliefs.angle_belief, beliefs.angle_bins)
    exp_elevation = 0.0  # Simplified: assume level flight
    exp_suitability = sum(beliefs.suitability_belief .* beliefs.suitability_bins)
    
    # Uncertainty-weighted suitability adjustment
    location_uncertainty = belief_uncertainty(beliefs.location_belief)
    angle_uncertainty = belief_uncertainty(beliefs.angle_belief)
    
    # Reduce suitability when uncertainty is high
    adjusted_suitability = exp_suitability * (1.0 - 0.3 * (location_uncertainty + angle_uncertainty))
    
    return StateSpace.DroneState(
        distance = exp_distance,
        azimuth = exp_azimuth,
        elevation = exp_elevation,
        suitability = clamp(adjusted_suitability, 0.0, 1.0)
    )
end

"""
    belief_uncertainty(belief::Vector{Float64})

Calculate normalized entropy (uncertainty) of a categorical belief.
"""
function belief_uncertainty(belief::Vector{Float64})
    entropy = -sum(p * log(max(p, 1e-10)) for p in belief)
    max_entropy = log(length(belief))
    return entropy / max_entropy
end

"""
    weighted_average_circular(belief::Vector{Float64}, angles::Vector{Float64})

Calculate expected value for circular quantities using unit vector method.
"""
function weighted_average_circular(belief::Vector{Float64}, angles::Vector{Float64})
    sin_avg = sum(belief .* sin.(angles))
    cos_avg = sum(belief .* cos.(angles))
    return atan(sin_avg, cos_avg)
end

"""
    serialize_beliefs(beliefs::DroneBeliefs)

Convert factorized beliefs to JSON-serializable format.
"""
function serialize_beliefs(beliefs::DroneBeliefs)
    # Convert voxel grid to arrays
    voxel_grid_arrays = [[v[1], v[2], v[3]] for v in beliefs.voxel_grid]
    
    return Dict(
        "location_belief" => Array(beliefs.location_belief),
        "angle_belief" => Array(beliefs.angle_belief),
        "suitability_belief" => Array(beliefs.suitability_belief),
        "location_bins" => Array(beliefs.location_bins),
        "angle_bins" => Array(beliefs.angle_bins),
        "elevation_bins" => Array(beliefs.elevation_bins),
        "suitability_bins" => Array(beliefs.suitability_bins),
        "B_location" => Array(beliefs.B_location),
        "B_angle" => Array(beliefs.B_angle),
        "B_suitability" => Array(beliefs.B_suitability),
        "A_location" => Array(beliefs.A_location),
        "A_angle" => Array(beliefs.A_angle),
        "A_suitability" => Array(beliefs.A_suitability),
        "voxel_grid" => voxel_grid_arrays,
        "vfe_cache" => beliefs.vfe_cache
    )
end

"""
    deserialize_beliefs(data::Dict)

Reconstruct factorized beliefs from serialized data.
"""
function deserialize_beliefs(data::Dict)
    # Set defaults for new factorized structure
    default_bins = 50
    default_belief = ones(default_bins) ./ default_bins
    default_location_bins = collect(range(0.0, stop=120.0, length=default_bins))
    default_angle_bins = collect(range(-π, stop=π, length=default_bins))
    default_elevation_bins = collect(range(-π/2, stop=π/2, length=default_bins))
    default_suitability_bins = collect(range(0.0, stop=1.0, length=default_bins))
    
    # Handle legacy format conversion
    if haskey(data, "distance_belief") && !haskey(data, "location_belief")
        # Convert from old format to new factorized format
        location_belief = get(data, "distance_belief", copy(default_belief))
        if length(location_belief) != default_bins
            # Resample to standard size
            location_belief = resample_belief(location_belief, default_bins)
        end
        
        angle_belief = get(data, "azimuth_belief", copy(default_belief))
        if length(angle_belief) != default_bins
            angle_belief = resample_belief(angle_belief, default_bins)
        end
        
        suitability_belief = get(data, "suitability_belief", copy(default_belief))
        if length(suitability_belief) != default_bins
            suitability_belief = resample_belief(suitability_belief, default_bins)
        end
    else
        # New factorized format
        location_belief = get(data, "location_belief", copy(default_belief))
        angle_belief = get(data, "angle_belief", copy(default_belief))
        suitability_belief = get(data, "suitability_belief", copy(default_belief))
    end
    
    # Get bins (with defaults)
    location_bins = get(data, "location_bins", default_location_bins)
    angle_bins = get(data, "angle_bins", default_angle_bins)
    elevation_bins = get(data, "elevation_bins", default_elevation_bins)
    suitability_bins = get(data, "suitability_bins", default_suitability_bins)
    
    # Get or initialize matrices
    B_location = get(data, "B_location", initialize_transition_matrix(default_bins, 0.9))
    B_angle = get(data, "B_angle", initialize_transition_matrix(default_bins, 0.8))
    B_suitability = get(data, "B_suitability", initialize_transition_matrix(default_bins, 0.95))
    
    A_location = get(data, "A_location", initialize_observation_matrix(default_bins, location_bins, "distance"))
    A_angle = get(data, "A_angle", initialize_observation_matrix(default_bins, angle_bins, "angle"))
    A_suitability = get(data, "A_suitability", initialize_observation_matrix(default_bins, suitability_bins, "suitability"))
    
    # Convert voxel grid
    voxel_grid = Vector{SVector{3, Float64}}()
    if haskey(data, "voxel_grid")
        for point in data["voxel_grid"]
            if length(point) == 3
                push!(voxel_grid, SVector{3, Float64}(point...))
            end
        end
    end
    
    # VFE cache
    vfe_cache = get(data, "vfe_cache", Dict{String, Float64}())
    
    return DroneBeliefs(
        location_belief,
        angle_belief,
        suitability_belief,
        location_bins,
        angle_bins,
        elevation_bins,
        suitability_bins,
        B_location,
        B_angle,
        B_suitability,
        A_location,
        A_angle,
        A_suitability,
        voxel_grid,
        vfe_cache
    )
end

"""
    resample_belief(belief::Vector{Float64}, target_size::Int)

Resample a belief distribution to a different size.
"""
function resample_belief(belief::Vector{Float64}, target_size::Int)
    if length(belief) == target_size
        return belief
    end
    
    # Simple linear interpolation resampling
    old_indices = range(1, stop=length(belief), length=length(belief))
    new_indices = range(1, stop=length(belief), length=target_size)
    
    resampled = zeros(target_size)
    for i in 1:target_size
        # Find closest indices for interpolation
        idx = new_indices[i]
        lower_idx = floor(Int, idx)
        upper_idx = min(ceil(Int, idx), length(belief))
        
        if lower_idx == upper_idx
            resampled[i] = belief[lower_idx]
        else
            # Linear interpolation
            α = idx - lower_idx
            resampled[i] = (1 - α) * belief[lower_idx] + α * belief[upper_idx]
        end
    end
    
    # Renormalize
    resampled ./= sum(resampled)
    return resampled
end

end # module Inference
