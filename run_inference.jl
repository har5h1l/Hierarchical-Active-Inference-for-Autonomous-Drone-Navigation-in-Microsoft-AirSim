#!/usr/bin/env julia

# Only activate if environment is not already active
if Base.active_project() != abspath(joinpath(@__DIR__, "Project.toml"))
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(path=joinpath(@__DIR__, "actinf"))
end

# Precompile necessary packages upfront
println("Precompiling packages...")
using JSON
using LinearAlgebra
using StaticArrays
@time begin
    using actinf
    # Force compilation of critical functions
    dummy_state = actinf.StateSpace.DroneState()
    dummy_beliefs = actinf.Inference.initialize_beliefs(dummy_state)
end
println("Precompilation complete!")

# Constants and parameters
const INTERFACE_DIR = abspath(joinpath(@__DIR__, "interface"))
const OBS_INPUT_PATH = abspath(joinpath(INTERFACE_DIR, "obs_input.json"))
const INFERRED_STATE_PATH = abspath(joinpath(INTERFACE_DIR, "inferred_state.json"))

# Make sure interface directory exists and is writable
try
    mkpath(INTERFACE_DIR)
    # Test write access by touching the file
    touch(INFERRED_STATE_PATH)
catch e
    error("Failed to create or access interface directory: $e")
end

function main()
    # Print startup message
    println("\nStarting drone navigation inference...")
    
    # Check if input file exists
    if !isfile(OBS_INPUT_PATH)
        error("Input file $OBS_INPUT_PATH not found.")
    end

    # Read observation data from JSON
    println("\nReading sensor data...")
    obs_data = JSON.parsefile(OBS_INPUT_PATH)

    # Extract and print initial positions
    drone_position = SVector{3, Float64}(get(obs_data, "drone_position", [0.0, 0.0, 0.0])...)
    target_location = SVector{3, Float64}(get(obs_data, "target_location", [10.0, 0.0, -3.0])...)
    
    println("\nInitial Positions (Global Coordinates):")
    println("Drone:  [$(round(drone_position[1], digits=2)), $(round(drone_position[2], digits=2)), $(round(drone_position[3], digits=2))]")
    println("Target: [$(round(target_location[1], digits=2)), $(round(target_location[2], digits=2)), $(round(target_location[3], digits=2))]")
    
    # Extract remaining data
    drone_orientation = SVector{4, Float64}(get(obs_data, "drone_orientation", [1.0, 0.0, 0.0, 0.0])...)
    nearest_obstacle_distances = get(obs_data, "nearest_obstacle_distances", [100.0, 100.0])
    obstacle_density = get(obs_data, "obstacle_density", 0.0)
    
    # Parse voxel grid more efficiently
    voxel_grid = [SVector{3, Float64}(point...) for point in get(obs_data, "voxel_grid", Vector{Vector{Float64}}()) if length(point) == 3]

    # Create DroneObservation object
    observation = DroneObservation(
        drone_position = drone_position,
        drone_orientation = drone_orientation,
        target_position = target_location,
        nearest_obstacle_distances = nearest_obstacle_distances,
        voxel_grid = voxel_grid,
        obstacle_density = obstacle_density
    )

    # Create current state from observation
    current_state = create_state_from_observation(observation)
    
    # Initialize or update beliefs
    println("\nUpdating belief state...")
    beliefs = try
        if isfile(INFERRED_STATE_PATH)
            prev_beliefs_json = JSON.parsefile(INFERRED_STATE_PATH)
            prev_beliefs = haskey(prev_beliefs_json, "beliefs") ? 
                          deserialize_beliefs(prev_beliefs_json["beliefs"]) :
                          initialize_beliefs(current_state, voxel_grid=voxel_grid)
            update_beliefs!(prev_beliefs, current_state; voxel_grid=voxel_grid)
        else
            initialize_beliefs(current_state, voxel_grid=voxel_grid)
        end
    catch e
        # If there's any error reading/parsing previous beliefs, initialize new ones
        initialize_beliefs(current_state, voxel_grid=voxel_grid)
    end
    
    # Get expected state from beliefs
    expected_drone_state = expected_state(beliefs)
    
    # Print navigation information
    println("\nNavigation State:")
    println("Current distance to target: $(round(current_state.distance, digits=2)) meters")
    println("Current azimuth to target: $(round(rad2deg(current_state.azimuth), digits=2))°")
    println("Current elevation to target: $(round(rad2deg(current_state.elevation), digits=2))°")
    println("Path suitability: $(round(current_state.suitability, digits=2))")
    println("Obstacle density: $(round(current_state.obstacle_density, digits=2))")
    
    # Print target and drone positions in global coordinates
    println("\nGlobal Coordinates:")
    println("Target position: [$(round(target_location[1], digits=2)), $(round(target_location[2], digits=2)), $(round(target_location[3], digits=2))]")
    println("Drone position: [$(round(drone_position[1], digits=2)), $(round(drone_position[2], digits=2)), $(round(drone_position[3], digits=2))]")
    println("Distance to target: $(round(current_state.distance, digits=2))")
    println("Azimuth to target: $(round(rad2deg(current_state.azimuth), digits=2))°")
    println("Elevation to target: $(round(rad2deg(current_state.elevation), digits=2))°")
    println("Path suitability: $(round(current_state.suitability, digits=2))\n")
    
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
            "distance" => expected_drone_state.distance,
            "azimuth" => expected_drone_state.azimuth,
            "elevation" => expected_drone_state.elevation,
            "suitability" => expected_drone_state.suitability,
            "obstacle_density" => expected_drone_state.obstacle_density
        ),
        "beliefs" => serialize_beliefs(beliefs),
        "drone_position" => [drone_position[1], drone_position[2], drone_position[3]],
        "target_position" => [target_location[1], target_location[2], target_location[3]],
        "nearest_obstacle_distances" => nearest_obstacle_distances,
        "obstacle_density" => obstacle_density,
        "voxel_count" => length(voxel_grid)
    )
    
    # Write to file with error handling
    try
        # Write JSON data without encoding parameter
        open(INFERRED_STATE_PATH, "w") do f
            JSON.print(f, output_data)
        end
    catch e
        # If writing fails, try to recreate directory and retry once
        mkpath(dirname(INFERRED_STATE_PATH))
        open(INFERRED_STATE_PATH, "w") do f
            JSON.print(f, output_data)
        end
    end
end

# Run the main function
main()