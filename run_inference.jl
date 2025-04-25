#!/usr/bin/env julia

# Add the current directory to the load path
push!(LOAD_PATH, @__DIR__)

# Import required modules
using JSON
using LinearAlgebra
using StaticArrays
import actinf.StateSpace
import actinf.Inference

# Constants and parameters
const INTERFACE_DIR = "./interface"
const OBS_INPUT_PATH = joinpath(INTERFACE_DIR, "obs_input.json")
const INFERRED_STATE_PATH = joinpath(INTERFACE_DIR, "inferred_state.json")

# Make sure interface directory exists
mkpath(INTERFACE_DIR)

function main()
    println("Starting inference pipeline...")

    # Check if input file exists
    if !isfile(OBS_INPUT_PATH)
        error("Input file $OBS_INPUT_PATH not found.")
    end

    # Read observation data from JSON
    println("Reading observations from $OBS_INPUT_PATH...")
    obs_data = JSON.parsefile(OBS_INPUT_PATH)

    # Extract required data
    drone_position = get(obs_data, "drone_position", [0.0, 0.0, 0.0])
    drone_orientation = get(obs_data, "drone_orientation", [1.0, 0.0, 0.0, 0.0])
    nearest_obstacle_distances = get(obs_data, "nearest_obstacle_distances", [100.0, 100.0])
    obstacle_density = get(obs_data, "obstacle_density", 0.0)
    target_location = get(obs_data, "target_location", [10.0, 0.0, -3.0])
    
    # Parse voxel grid - converting from array of arrays to Vector of SVector
    voxel_grid_raw = get(obs_data, "voxel_grid", [])
    voxel_grid = Vector{SVector{3, Float64}}()
    for point in voxel_grid_raw
        if length(point) == 3
            push!(voxel_grid, SVector{3, Float64}(point[1], point[2], point[3]))
        end
    end
    
    println("Processed $(length(voxel_grid)) voxels from sensor data")

    # Convert to SVector for StateSpace compatibility
    drone_pos_vec = SVector{3, Float64}(drone_position...)
    target_pos_vec = SVector{3, Float64}(target_location...)

    # Create DroneObservation object
    observation = StateSpace.DroneObservation(
        drone_position = drone_pos_vec,
        drone_orientation = SVector{4, Float64}(drone_orientation...),
        target_position = target_pos_vec,
        nearest_obstacle_distances = nearest_obstacle_distances,
        voxel_grid = voxel_grid,
        obstacle_density = obstacle_density
    )

    # Create current state from observation
    current_state = StateSpace.create_state_from_observation(observation)
    
    # Initialize or update beliefs
    println("Running inference...")
    beliefs = if isfile(INFERRED_STATE_PATH)
        # Load previous beliefs
        prev_beliefs_json = JSON.parsefile(INFERRED_STATE_PATH)
        prev_beliefs = Inference.deserialize_beliefs(prev_beliefs_json)
        
        # Update beliefs with current observation, including new voxel grid
        Inference.update_beliefs!(prev_beliefs, current_state, voxel_grid=voxel_grid)
    else
        # Initialize new beliefs with the voxel grid
        Inference.initialize_beliefs(current_state, voxel_grid=voxel_grid)
    end
    
    # Get expected state from beliefs
    expected_state = Inference.expected_state(beliefs)
    
    # Serialize the current state and beliefs to JSON
    output_data = Dict(
        "state" => Dict(
            "distance" => current_state.distance,
            "azimuth" => current_state.azimuth,
            "elevation" => current_state.elevation,
            "suitability" => current_state.suitability,
            "obstacle_density" => current_state.obstacle_density
        ),
        "expected_state" => Dict(
            "distance" => expected_state.distance,
            "azimuth" => expected_state.azimuth,
            "elevation" => expected_state.elevation,
            "suitability" => expected_state.suitability,
            "obstacle_density" => expected_state.obstacle_density
        ),
        "beliefs" => Inference.serialize_beliefs(beliefs),
        "drone_position" => drone_position,
        "target_position" => target_location,
        "nearest_obstacle_distances" => nearest_obstacle_distances,
        "obstacle_density" => obstacle_density,
        "voxel_count" => length(voxel_grid)
    )
    
    # Save to file
    println("Saving inferred state to $INFERRED_STATE_PATH...")
    open(INFERRED_STATE_PATH, "w") do f
        JSON.print(f, output_data, 2)  # Pretty print with 2-space indent
    end
    
    println("Inference complete.")
end

# Run the main function
main()