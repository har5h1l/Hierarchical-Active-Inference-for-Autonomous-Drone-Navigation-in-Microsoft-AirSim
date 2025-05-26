module Inference

using StaticArrays
using ..StateSpace

export DroneBeliefs, initialize_beliefs, update_beliefs!, expected_state
export serialize_beliefs, deserialize_beliefs

"""
    DroneBeliefs

Container for probabilistic beliefs about the drone state.
Each field is a discretized probability distribution.
"""
mutable struct DroneBeliefs
    distance_belief::Vector{Float64}
    azimuth_belief::Vector{Float64}
    elevation_belief::Vector{Float64}
    suitability_belief::Vector{Float64}
    density_belief::Vector{Float64}
    
    # Define discretization ranges for each variable
    distance_range::Vector{Float64}
    azimuth_range::Vector{Float64}
    elevation_range::Vector{Float64}
    suitability_range::Vector{Float64}
    density_range::Vector{Float64}
    
    # Store sensory data for better state transition modeling
    voxel_grid::Vector{SVector{3, Float64}}
end

"""
    initialize_beliefs(state::StateSpace.DroneState; num_bins=100)

Initialize uniform beliefs and update with current state observation.
"""
function initialize_beliefs(state::StateSpace.DroneState; num_bins=50, voxel_grid=Vector{SVector{3, Float64}}(), obstacle_density=0.0)
    # Define ranges for each state variable
    distance_range = collect(range(0.0, stop=125.0, length=(num_bins+25)))
    azimuth_range = collect(range(-π, stop=π, length=num_bins))
    elevation_range = collect(range(-π/2, stop=π/2, length=num_bins))
    suitability_range = collect(range(0.0, stop=1.0, length=num_bins))
    density_range = collect(range(0.0, stop=1.0, length=num_bins))
    
    # Initialize with uniform distributions using correct bin counts
    distance_belief = ones(num_bins+25) / (num_bins+25)  # 75 bins for distance
    azimuth_belief = ones(num_bins) / num_bins  # 50 bins for angles
    elevation_belief = ones(num_bins) / num_bins
    suitability_belief = ones(num_bins) / num_bins
    density_belief = ones(num_bins) / num_bins
    
    # Create beliefs object
    beliefs = DroneBeliefs(
        distance_belief,
        azimuth_belief,
        elevation_belief,
        suitability_belief,
        density_belief,
        distance_range,
        azimuth_range,
        elevation_range,
        suitability_range,
        density_range,
        voxel_grid
    )
    
    # Update with initial state
    update_beliefs!(beliefs, state; obstacle_density=obstacle_density)
    
    return beliefs
end

"""
    update_beliefs!(beliefs::DroneBeliefs, state::StateSpace.DroneState; kernel_width=0.1)

Update belief distributions based on new state observation.
Uses a Gaussian kernel to update beliefs with stronger weighting for target-related beliefs.
"""
function update_beliefs!(beliefs::DroneBeliefs, state::StateSpace.DroneState; kernel_width=0.1, voxel_grid=nothing, obstacle_density=0.0)
    # Adjust kernel widths based on state variables
    distance_kernel = kernel_width * (1.0 + 0.5 * state.distance/100.0)  # Wider kernel for larger distances
    azimuth_kernel = kernel_width * 2.0  # Wider kernel for angles
    elevation_kernel = kernel_width * 2.0
    
    # Update target-related beliefs with stronger influence
    update_belief_dim!(beliefs.distance_belief, beliefs.distance_range, state.distance, distance_kernel)
    update_belief_dim!(beliefs.azimuth_belief, beliefs.azimuth_range, state.azimuth, azimuth_kernel)
    update_belief_dim!(beliefs.elevation_belief, beliefs.elevation_range, state.elevation, elevation_kernel)
    
    # Add target preference weighting - fixed the maximum function error
    max_distance = maximum(beliefs.distance_range)
    target_weight = exp(-state.distance / 10.0)  # Stronger preference for closer positions to target
    beliefs.distance_belief .*= (1.0 .+ target_weight * (max_distance .- beliefs.distance_range) ./ max_distance)
    beliefs.distance_belief ./= sum(beliefs.distance_belief)
    
    # Update environment-related beliefs
    update_belief_dim!(beliefs.suitability_belief, beliefs.suitability_range, state.suitability, kernel_width)
    update_belief_dim!(beliefs.density_belief, beliefs.density_range, obstacle_density, kernel_width)
    
    # Update voxel grid if provided
    if voxel_grid !== nothing
        empty!(beliefs.voxel_grid)
        append!(beliefs.voxel_grid, voxel_grid)
    end
    
    return beliefs
end

"""
    update_belief_dim!(belief::Vector{Float64}, range_values::Vector{Float64}, 
                      observation::Float64, kernel_width::Float64)

Helper function to update a single belief dimension with adaptive kernel width.
"""
function update_belief_dim!(belief::Vector{Float64}, range_values::Vector{Float64}, 
                          observation::Float64, kernel_width::Float64)
    # Create likelihood based on Gaussian kernel
    likelihood = zeros(length(belief))
    
    # Check if this is an angular quantity by looking at the range
    is_angular = minimum(range_values) ≈ -π && maximum(range_values) ≈ π
    
    for i in 1:length(belief)
        # Handle circular quantities properly for angular values
        if is_angular
            distance = min(abs(range_values[i] - observation), 2π - abs(range_values[i] - observation))
        else
            distance = range_values[i] - observation
        end
        likelihood[i] = exp(-0.5 * (distance / kernel_width)^2)
    end
    
    # Normalize likelihood
    likelihood ./= sum(likelihood)
    
    # Bayesian update with temporal smoothing
    belief .= 0.8 .* belief .+ 0.2 .* likelihood
    
    # Re-normalize
    belief ./= sum(belief)
    
    return belief
end

"""
    expected_state(beliefs::DroneBeliefs)

Compute expected state from current beliefs with improved handling of circular quantities.
"""
function expected_state(beliefs::DroneBeliefs)
    # Calculate expected values with uncertainty consideration
    exp_distance = weighted_average(beliefs.distance_belief, beliefs.distance_range)
    exp_azimuth = weighted_average_circular(beliefs.azimuth_belief, beliefs.azimuth_range)
    exp_elevation = weighted_average(beliefs.elevation_belief, beliefs.elevation_range)
    
    # Calculate uncertainty-weighted suitability
    exp_suitability = weighted_average(beliefs.suitability_belief, beliefs.suitability_range)
    exp_density = weighted_average(beliefs.density_belief, beliefs.density_range)
    
    # Adjust suitability based on belief uncertainty
    distance_uncertainty = belief_uncertainty(beliefs.distance_belief)
    angle_uncertainty = belief_uncertainty(beliefs.azimuth_belief)
    
    # Reduce suitability when uncertainty is high
    adjusted_suitability = exp_suitability * (1.0 - 0.5 * (distance_uncertainty + angle_uncertainty))
    
    return StateSpace.DroneState(
        distance = exp_distance,
        azimuth = exp_azimuth,
        elevation = exp_elevation,
        suitability = adjusted_suitability
    )
end

"""
    belief_uncertainty(belief::Vector{Float64})

Calculate the uncertainty in a belief distribution.
Returns a value between 0 (certain) and 1 (maximum uncertainty).
"""
function belief_uncertainty(belief::Vector{Float64})
    # Calculate normalized entropy
    entropy = -sum(p * log(max(p, 1e-10)) for p in belief)
    max_entropy = log(length(belief))
    return entropy / max_entropy
end

"""
    weighted_average(belief::Vector{Float64}, values::Vector{Float64})

Calculate the expected value of a distribution.
"""
function weighted_average(belief::Vector{Float64}, values::Vector{Float64})
    return sum(belief .* values)
end

"""
    weighted_average_circular(belief::Vector{Float64}, values::Vector{Float64})

Calculate the expected value for circular quantities (like angles).
"""
function weighted_average_circular(belief::Vector{Float64}, values::Vector{Float64})
    # Convert to unit vectors and take weighted average
    sin_avg = sum(belief .* sin.(values))
    cos_avg = sum(belief .* cos.(values))
    
    # Convert back to angle
    return atan(sin_avg, cos_avg)
end

"""
    serialize_beliefs(beliefs::DroneBeliefs)

Convert beliefs to a JSON-serializable format.
"""
function serialize_beliefs(beliefs::DroneBeliefs)
    # Convert voxel grid to regular arrays for JSON serialization
    voxel_grid_arrays = [[v[1], v[2], v[3]] for v in beliefs.voxel_grid]
    
    return Dict(
        "distance_belief" => Array(beliefs.distance_belief),
        "azimuth_belief" => Array(beliefs.azimuth_belief),
        "elevation_belief" => Array(beliefs.elevation_belief),
        "suitability_belief" => Array(beliefs.suitability_belief),
        "density_belief" => Array(beliefs.density_belief),
        "distance_range" => Array(beliefs.distance_range),
        "azimuth_range" => Array(beliefs.azimuth_range),
        "elevation_range" => Array(beliefs.elevation_range),
        "suitability_range" => Array(beliefs.suitability_range),
        "density_range" => Array(beliefs.density_range),
        "voxel_grid" => voxel_grid_arrays
    )
end

"""
    deserialize_beliefs(data::Dict)

Reconstruct beliefs from serialized data.
"""
function deserialize_beliefs(data::Dict)
    # Set default values for missing data - updated to match new bin counts
    default_num_bins = 50
    default_distance_bins = default_num_bins + 25  # 75 bins for distance
    
    default_distance_belief = ones(default_distance_bins) ./ default_distance_bins
    default_angle_belief = ones(default_num_bins) ./ default_num_bins
    
    default_distance_range = collect(range(0.0, stop=125.0, length=default_distance_bins))
    default_angle_range = collect(range(-π, stop=π, length=default_num_bins))
    default_elev_range = collect(range(-π/2, stop=π/2, length=default_num_bins))
    default_unit_range = collect(range(0.0, stop=1.0, length=default_num_bins))
    
    # Get values with defaults
    distance_belief = get(data, "distance_belief", copy(default_distance_belief))
    azimuth_belief = get(data, "azimuth_belief", copy(default_angle_belief))
    elevation_belief = get(data, "elevation_belief", copy(default_angle_belief))
    suitability_belief = get(data, "suitability_belief", copy(default_angle_belief))
    density_belief = get(data, "density_belief", copy(default_angle_belief))
    
    distance_range = get(data, "distance_range", default_distance_range)
    azimuth_range = get(data, "azimuth_range", default_angle_range)
    elevation_range = get(data, "elevation_range", default_elev_range)
    suitability_range = get(data, "suitability_range", default_unit_range)
    density_range = get(data, "density_range", default_unit_range)
    
    # Convert voxel grid arrays to SVectors
    voxel_grid = Vector{SVector{3, Float64}}()
    if haskey(data, "voxel_grid")
        for point in data["voxel_grid"]
            if length(point) == 3
                push!(voxel_grid, SVector{3, Float64}(point...))
            end
        end
    end
    
    return DroneBeliefs(
        distance_belief,
        azimuth_belief,
        elevation_belief,
        suitability_belief,
        density_belief,
        distance_range,
        azimuth_range,
        elevation_range,
        suitability_range,
        density_range,
        voxel_grid
    )
end

end # module Inference
