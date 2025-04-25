module Inference

export DroneBeliefs, initialize_beliefs, update_beliefs!, expected_state, serialize_beliefs, deserialize_beliefs

using ..StateSpace
using StaticArrays

"""
    DroneBeliefs

Container for probabilistic beliefs about the drone state.
Each field is a discretized probability distribution.
"""
struct DroneBeliefs
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
    initialize_beliefs(state::StateSpace.DroneState; num_bins=50)

Initialize uniform beliefs and update with current state observation.
"""
function initialize_beliefs(state::StateSpace.DroneState; num_bins=50, voxel_grid=Vector{SVector{3, Float64}}())
    # Define ranges for each state variable
    distance_range = collect(range(0.0, stop=50.0, length=num_bins))
    azimuth_range = collect(range(-π, stop=π, length=num_bins))
    elevation_range = collect(range(-π/2, stop=π/2, length=num_bins))
    suitability_range = collect(range(0.0, stop=1.0, length=num_bins))
    density_range = collect(range(0.0, stop=1.0, length=num_bins))
    
    # Initialize with uniform distributions
    distance_belief = ones(num_bins) / num_bins
    azimuth_belief = ones(num_bins) / num_bins
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
    update_beliefs!(beliefs, state)
    
    return beliefs
end

"""
    update_beliefs!(beliefs::DroneBeliefs, state::StateSpace.DroneState; kernel_width=0.1, voxel_grid=nothing)

Update belief distributions based on new state observation.
Uses a Gaussian kernel to update beliefs.
"""
function update_beliefs!(beliefs::DroneBeliefs, state::StateSpace.DroneState; kernel_width=0.1, voxel_grid=nothing)
    # Apply Bayesian update for each state variable
    update_belief_dim!(beliefs.distance_belief, beliefs.distance_range, state.distance, kernel_width)
    update_belief_dim!(beliefs.azimuth_belief, beliefs.azimuth_range, state.azimuth, kernel_width*2)  # Wider kernel for angles
    update_belief_dim!(beliefs.elevation_belief, beliefs.elevation_range, state.elevation, kernel_width*2)
    update_belief_dim!(beliefs.suitability_belief, beliefs.suitability_range, state.suitability, kernel_width)
    update_belief_dim!(beliefs.density_belief, beliefs.density_range, state.obstacle_density, kernel_width)
    
    # Update voxel grid if provided
    if voxel_grid !== nothing
        # Replace the existing voxel grid with the new one
        beliefs.voxel_grid = voxel_grid
    end
    
    return beliefs
end

"""
    update_belief_dim!(belief::Vector{Float64}, range_values::Vector{Float64}, observation::Float64, kernel_width::Float64)

Helper function to update a single belief dimension.
"""
function update_belief_dim!(belief::Vector{Float64}, range_values::Vector{Float64}, observation::Float64, kernel_width::Float64)
    # Create likelihood based on Gaussian kernel
    likelihood = zeros(length(belief))
    for i in 1:length(belief)
        # Calculate squared distance (with periodic wrapping for angles if needed)
        distance = range_values[i] - observation
        
        # Apply Gaussian kernel
        likelihood[i] = exp(-0.5 * (distance / kernel_width)^2)
    end
    
    # Normalize likelihood
    likelihood ./= sum(likelihood)
    
    # Bayesian update (element-wise multiplication)
    belief .*= likelihood
    
    # Re-normalize
    belief ./= sum(belief)
    
    return belief
end

"""
    expected_state(beliefs::DroneBeliefs)

Compute expected state from current beliefs (weighted average).
"""
function expected_state(beliefs::DroneBeliefs)
    # Calculate expected values for each state dimension
    exp_distance = weighted_average(beliefs.distance_belief, beliefs.distance_range)
    exp_azimuth = weighted_average_circular(beliefs.azimuth_belief, beliefs.azimuth_range)
    exp_elevation = weighted_average(beliefs.elevation_belief, beliefs.elevation_range)
    exp_suitability = weighted_average(beliefs.suitability_belief, beliefs.suitability_range)
    exp_density = weighted_average(beliefs.density_belief, beliefs.density_range)
    
    # Return as a DroneState
    return StateSpace.DroneState(
        distance = exp_distance,
        azimuth = exp_azimuth,
        elevation = exp_elevation,
        suitability = exp_suitability,
        obstacle_density = exp_density
    )
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
    # We can't directly serialize the SVector voxel grid, 
    # so we convert to regular arrays for JSON serialization
    voxel_grid_arrays = []
    for voxel in beliefs.voxel_grid
        push!(voxel_grid_arrays, [voxel[1], voxel[2], voxel[3]])
    end
    
    return Dict(
        "distance_belief" => beliefs.distance_belief,
        "azimuth_belief" => beliefs.azimuth_belief,
        "elevation_belief" => beliefs.elevation_belief,
        "suitability_belief" => beliefs.suitability_belief,
        "density_belief" => beliefs.density_belief,
        "distance_range" => beliefs.distance_range,
        "azimuth_range" => beliefs.azimuth_range,
        "elevation_range" => beliefs.elevation_range,
        "suitability_range" => beliefs.suitability_range,
        "density_range" => beliefs.density_range,
        "voxel_grid" => voxel_grid_arrays
    )
end

"""
    deserialize_beliefs(data::Dict)

Reconstruct beliefs from serialized data.
"""
function deserialize_beliefs(data::Dict)
    # Convert the arrays back to SVectors for the voxel grid
    voxel_grid = Vector{SVector{3, Float64}}()
    if haskey(data, "voxel_grid")
        for point in data["voxel_grid"]
            if length(point) == 3
                push!(voxel_grid, SVector{3, Float64}(point[1], point[2], point[3]))
            end
        end
    end
    
    return DroneBeliefs(
        data["distance_belief"],
        data["azimuth_belief"],
        data["elevation_belief"],
        data["suitability_belief"],
        data["density_belief"],
        data["distance_range"],
        data["azimuth_range"],
        data["elevation_range"],
        data["suitability_range"],
        data["density_range"],
        voxel_grid
    )
end

end # module
