#!/usr/bin/env julia

println("Starting drone navigation inference script...")

# Check if we have been precompiled already
const IS_PRECOMPILED = haskey(ENV, "JULIA_ACTINF_PRECOMPILED")

# Only activate project if it's not already active
# This avoids redundant activation during navigation
if !IS_PRECOMPILED
    println("Not precompiled. Activating project environment...")
    import Pkg
    Pkg.activate(@__DIR__)
    println("Project activated")
else
    println("Using precompiled environment")
end

# Import packages without reloading if already loaded
function use_package(pkg_name, force_load=false)
    if !isdefined(Main, Symbol(pkg_name)) || force_load
        println("Loading $pkg_name...")
        @eval import $pkg_name
        println("Successfully loaded $pkg_name")
    else
        println("Using already loaded package: $pkg_name")
    end
end

# Core packages - only load if not already loaded
use_package(:JSON)
use_package(:LinearAlgebra)
use_package(:StaticArrays)

# Make StaticArrays available in main scope if needed
if !isdefined(Main, :SVector)
    @eval using StaticArrays: SVector
    println("Imported SVector into scope")
else
    println("SVector already in scope")
end

# Load actinf package if not already loaded
if !isdefined(Main, :actinf) 
    println("Loading actinf package...")
    try
        @eval using actinf
        println("Successfully loaded actinf package")
    catch e
        println("Error loading actinf package: $e")
        error("Cannot continue without actinf package")
    end
else
    println("Using already loaded actinf package")
end

# Import specific components from actinf to make them available in Main scope
using actinf.StateSpace: DroneObservation, create_state_from_observation
using actinf.Inference: DroneBeliefs, initialize_beliefs, update_beliefs!, expected_state, serialize_beliefs, deserialize_beliefs

# Constants and parameters with more robust path handling for cross-platform compatibility
const INTERFACE_DIR = joinpath(@__DIR__, "interface")
const OBS_INPUT_PATH = joinpath(INTERFACE_DIR, "obs_input.json")
const INFERRED_STATE_PATH = joinpath(INTERFACE_DIR, "inferred_state.json")

# Make sure interface directory exists and is writable
try
    # Create the directory if it doesn't exist
    if !isdir(INTERFACE_DIR)
        mkpath(INTERFACE_DIR)
        println("Created interface directory: $INTERFACE_DIR")
    end
    
    # Test write access by touching the file
    temp_file = joinpath(INTERFACE_DIR, "test_write_access.tmp")
    touch(temp_file)
    if isfile(temp_file)
        rm(temp_file)  # Clean up the test file if successful
    end
catch e
    println("Warning: Issue with interface directory: $e")
    # Don't error out, let's try to continue and create the directory in the write section
end

function main()
    # Print startup message
    println("\nStarting drone navigation inference...")
    
    # Check if input file exists
    if !isfile(OBS_INPUT_PATH)
        error("Input file $OBS_INPUT_PATH not found.")
    end

    # Read observation data from JSON with robust error handling
    println("\nReading sensor data...")
    
    local obs_data
    try
        obs_data = open(OBS_INPUT_PATH, "r") do file
            JSON.parse(file)
        end
        println("Successfully parsed observation data")
    catch e
        println("Error parsing JSON from $OBS_INPUT_PATH: $e")
        # Create fallback minimal data
        obs_data = Dict(
            "drone_position" => [0.0, 0.0, 0.0],
            "target_location" => [10.0, 0.0, -3.0],
            "drone_orientation" => [1.0, 0.0, 0.0, 0.0],
            "nearest_obstacle_distances" => [100.0, 100.0],
            "obstacle_density" => 0.0,
            "voxel_grid" => []
        )
        println("Using fallback observation data")
    end

    # Extract and print initial positions with better error checking
    drone_position = try
        SVector{3, Float64}(get(obs_data, "drone_position", [0.0, 0.0, 0.0])...)
    catch e
        println("Error parsing drone_position, using default: $e")
        SVector{3, Float64}(0.0, 0.0, 0.0)
    end
    
    target_location = try
        target_key = haskey(obs_data, "target_position") ? "target_position" : "target_location"
        SVector{3, Float64}(get(obs_data, target_key, [10.0, 0.0, -3.0])...)
    catch e
        println("Error parsing target position, using default: $e")
        SVector{3, Float64}(10.0, 0.0, -3.0)
    end

    # Extract remaining data with robust error handling
    drone_orientation = try
        SVector{4, Float64}(get(obs_data, "drone_orientation", [1.0, 0.0, 0.0, 0.0])...)
    catch e
        println("Error parsing drone_orientation, using default: $e")
        SVector{4, Float64}(1.0, 0.0, 0.0, 0.0)
    end
    
    nearest_obstacle_distances = try
        Float64.(get(obs_data, "obstacle_distances", get(obs_data, "nearest_obstacle_distances", [100.0, 100.0])))
    catch e
        println("Error parsing nearest_obstacle_distances, using default: $e")
        Float64[100.0, 100.0]
    end
    
    obstacle_density = try
        Float64(get(obs_data, "obstacle_density", 0.0))
    catch e
        println("Error parsing obstacle_density, using default: $e")
        0.0
    end
    
    # Calculate distance to target for adaptive behaviors
    distance_to_target = try
        norm(target_location - drone_position)
    catch
        10.0  # Default if calculation fails
    end
    
    # Determine if we're in a high density obstacle area
    obstacles_count = try
        if haskey(obs_data, "obstacles_count")
            obs_data["obstacles_count"]
        elseif haskey(obs_data, "obstacle_positions")
            length(obs_data["obstacle_positions"])
        elseif haskey(obs_data, "voxel_grid")
            length(obs_data["voxel_grid"])
        else
            0
        end
    catch
        0
    end
    
    # Calculate closest obstacle distance
    closest_obstacle = try
        if !isempty(nearest_obstacle_distances)
            minimum(nearest_obstacle_distances)
        else
            100.0  # Default large value if no obstacles
        end
    catch
        100.0
    end
    
    # Determine if we're in a high density area
    high_density_area = (obstacles_count > 10 || closest_obstacle < 3.0 || obstacle_density > 0.5)
    
    # Calculate obstacle avoidance factor based on context
    # Higher values = more obstacle avoidance priority
    obstacle_avoidance_factor = 1.5  # Base value
    
    # Get initial distance for reference (if available) or use current as fallback
    initial_distance = if haskey(obs_data, "initial_distance")
        obs_data["initial_distance"]
    else
        distance_to_target * 1.2  # Add 20% if not provided
    end
    # Calculate percentage of distance to target (how close we are)
    distance_percentage = (distance_to_target / initial_distance) * 100.0
    
    println("Distance percentage to target: $(round(distance_percentage, digits=1))% of initial $(round(initial_distance, digits=2))m")
    
    # Dynamic adjustment based on percentage of distance to target instead of fixed distances
    if distance_percentage < 10.0  # Very close (last 10% of journey)
        # When very close to target, reduce obstacle avoidance to prioritize reaching target
        # But maintain a minimum safety factor that scales with remaining distance
        target_proximity_scale = max(0.5, distance_percentage / 10.0)  # Scale from 0.5-1.0 based on percentage
        obstacle_avoidance_factor = 1.0 * target_proximity_scale
        println("Very close to target ($(round(distance_percentage, digits=1))%): Reduced obstacle avoidance to $(round(obstacle_avoidance_factor, digits=2))")
    elseif distance_percentage < 25.0  # Moderately close (10-25% of journey remaining)
        # When moderately close, gradual reduction
        target_proximity_scale = 0.8 + (0.2 * distance_percentage / 25.0)  # Scale from 0.8-1.0
        obstacle_avoidance_factor = 1.2 * target_proximity_scale
        println("Approaching target ($(round(distance_percentage, digits=1))%): Adjusted obstacle avoidance to $(round(obstacle_avoidance_factor, digits=2))")
    elseif distance_percentage > 75.0  # Still far from target (>75% of journey remaining)
        # When far from target, maintain higher obstacle avoidance
        obstacle_avoidance_factor = 1.8
    else
        # Linear interpolation between 25-75% with balanced values
        obstacle_avoidance_factor = 1.2 + (0.6 * (distance_percentage - 25.0) / 50.0)
    end
    
    # Special case: if direct path to target appears clear (from sensor data)
    # further reduce obstacle avoidance to encourage more direct approach
    direct_path_clear = get(obs_data, "direct_path_clear", false)
    direct_path_suitability = get(obs_data, "direct_path_suitability", 0.0)
    
    if direct_path_clear && direct_path_suitability > 0.75 && distance_percentage < 50.0
        # Clear path and closer than halfway - reduce obstacle avoidance further
        obstacle_avoidance_factor *= 0.8
        println("Clear direct path detected: Further reduced obstacle avoidance to $(round(obstacle_avoidance_factor, digits=2))")
    end
    
    # Adjustment in high density areas - less aggressive when close to target
    if high_density_area
        if distance_percentage < 10.0
            # Moderate increase when close to target in high density
            obstacle_avoidance_factor *= 1.3
            println("High density area near target: moderate increase to $(round(obstacle_avoidance_factor, digits=2))")
        else
            obstacle_avoidance_factor *= 1.5
        end
    end
    
    # For dynamic replanning scenarios, scale based on distance to target
    if get(obs_data, "dynamic_replanning", false)
        if distance_percentage < 10.0
            # Less aggressive replanning near target
            obstacle_avoidance_factor *= 1.5  # Reduced from 2.0
            println("Dynamic replanning near target: moderate increase to $(round(obstacle_avoidance_factor, digits=2))")
        else
            obstacle_avoidance_factor *= 2.0
            println("Dynamic replanning: Increased obstacle avoidance factor to $(obstacle_avoidance_factor)")
        end
    end
      # Critical obstacle avoidance - still maintain safety but scale based on target proximity
    if get(obs_data, "critical_obstacle_avoidance", false)
        if distance_percentage < 10.0
            # Even in critical situations, be more willing to take risks when very close to target
            obstacle_avoidance_factor *= 2.5  # Increased from 2.0
            println("CRITICAL OBSTACLE AVOIDANCE NEAR TARGET: factor = $(obstacle_avoidance_factor)")
        else
            obstacle_avoidance_factor *= 3.0  # Increased from 2.5
            println("CRITICAL OBSTACLE AVOIDANCE MODE: factor = $(obstacle_avoidance_factor)")
        end
    end
      # Adjust obstacle sensitivity based on obstacle avoidance factor
    # This directly impacts the weight given to obstacle density and distance in suitability calculation
    obstacle_distance_weight = 1.0 * obstacle_avoidance_factor  # Increased from 0.8 
    obstacle_density_weight = 0.4 * obstacle_avoidance_factor   # Increased from 0.3
    
    # Special case: if we're very close to target (<3m), slightly reduce obstacle weights
    # to allow reaching difficult targets, but never below a safe minimum
    if distance_percentage < 3.0 && !high_density_area
        minimum_obstacle_weight = 0.9  # Increased from 0.8
        obstacle_distance_weight = max(minimum_obstacle_weight, obstacle_distance_weight * 0.8)
        obstacle_density_weight = max(minimum_obstacle_weight * 0.6, obstacle_density_weight * 0.8)  # Increased from 0.5
        println("Close to target: Adjusted obstacle weights to $(obstacle_distance_weight) / $(obstacle_density_weight)")
    end
    
    # Handle obstacle positions with better key detection and error handling
    voxel_grid = Vector{SVector{3, Float64}}()
    try
        grid_data = Vector{Vector{Float64}}()
        
        # Try different possible keys for obstacle data
        if haskey(obs_data, "voxel_grid") && !isempty(obs_data["voxel_grid"])
            grid_data = obs_data["voxel_grid"]
            println("Using voxel_grid data with $(length(grid_data)) points")
        elseif haskey(obs_data, "obstacle_positions") && !isempty(obs_data["obstacle_positions"])
            grid_data = obs_data["obstacle_positions"]
            println("Using obstacle_positions data with $(length(grid_data)) points")
        end
        
        # Convert points to SVector format with validation
        for point in grid_data
            if length(point) >= 3
                push!(voxel_grid, SVector{3, Float64}(point[1], point[2], point[3]))
            end
        end
        
        println("Processed $(length(voxel_grid)) valid obstacle points")
    catch e
        println("Error processing obstacle positions: $e")
        voxel_grid = Vector{SVector{3, Float64}}()
    end
    
    println("\nInitial Positions (Global Coordinates):")
    println("Drone:  [$(round(drone_position[1], digits=2)), $(round(drone_position[2], digits=2)), $(round(drone_position[3], digits=2))]")
    println("Target: [$(round(target_location[1], digits=2)), $(round(target_location[2], digits=2)), $(round(target_location[3], digits=2))]")
    println("Distance to target: $(round(distance_to_target, digits=2)) meters")
    println("Obstacle density: $(round(obstacle_density, digits=2))")
    println("Obstacle avoidance factor: $(round(obstacle_avoidance_factor, digits=2))")
    println("High density area: $high_density_area")

    # Create DroneObservation object with enhanced obstacle awareness parameters
    observation = DroneObservation(
        drone_position = drone_position,
        drone_orientation = drone_orientation,
        target_position = target_location,
        nearest_obstacle_distances = nearest_obstacle_distances,
        voxel_grid = voxel_grid,
        obstacle_density = obstacle_density * obstacle_avoidance_factor,  # Scale density by avoidance factor
        high_density_area = high_density_area,
        obstacle_avoidance_factor = obstacle_avoidance_factor,
        distance_to_target = distance_to_target
    )

    # Create current state from observation
    current_state = create_state_from_observation(observation)
    
    # Initialize or update beliefs
    println("\nUpdating belief state...")
    beliefs = try
        if isfile(INFERRED_STATE_PATH)
            println("Found existing inferred state file: $INFERRED_STATE_PATH")
            prev_beliefs_json = open(INFERRED_STATE_PATH, "r") do file
                JSON.parse(file)
            end
            prev_beliefs = haskey(prev_beliefs_json, "beliefs") ? 
                          deserialize_beliefs(prev_beliefs_json["beliefs"]) :
                          initialize_beliefs(current_state, voxel_grid=voxel_grid, 
                                           obstacle_density=obstacle_density * obstacle_avoidance_factor)
            update_beliefs!(prev_beliefs, current_state; 
                          voxel_grid=voxel_grid, 
                          obstacle_density=obstacle_density * obstacle_avoidance_factor,
                          obstacle_weight=obstacle_avoidance_factor)
        else
            println("No existing inferred state file, initializing new beliefs")
            initialize_beliefs(current_state, 
                             voxel_grid=voxel_grid, 
                             obstacle_density=obstacle_density * obstacle_avoidance_factor)
        end
    catch e
        println("Error loading previous beliefs: $e")
        println("Initializing new beliefs")
        # If there's any error reading/parsing previous beliefs, initialize new ones
        initialize_beliefs(current_state, 
                         voxel_grid=voxel_grid, 
                         obstacle_density=obstacle_density * obstacle_avoidance_factor)
    end
    
    # Get expected state from beliefs
    expected_drone_state = expected_state(beliefs)
    
    # Print navigation information
    println("\nNavigation State:")
    println("Current distance to target: $(round(current_state.distance, digits=2)) meters")
    println("Current azimuth to target: $(round(rad2deg(current_state.azimuth), digits=2))°")
    println("Current elevation to target: $(round(rad2deg(current_state.elevation), digits=2))°")
    println("Path suitability: $(round(current_state.suitability, digits=2))")
    println("Obstacle density: $(round(obstacle_density, digits=2)) (scaled: $(round(obstacle_density * obstacle_avoidance_factor, digits=2)))")
    println("Nearest obstacle: $(isempty(nearest_obstacle_distances) ? "None" : "$(round(minimum(nearest_obstacle_distances), digits=2)) meters")")
    
    # Save updated beliefs to output file
    println("\nSaving updated beliefs...")
    try
        # Convert to JSON-compatible Dict
        output_dict = Dict(
            "state" => Dict(
                "distance" => current_state.distance,
                "azimuth" => current_state.azimuth,
                "elevation" => current_state.elevation,
                "suitability" => current_state.suitability
            ),
            "drone_position" => [drone_position[1], drone_position[2], drone_position[3]],
            "target_position" => [target_location[1], target_location[2], target_location[3]],
            "nearest_obstacle_distances" => nearest_obstacle_distances,
            "obstacle_density" => obstacle_density * obstacle_avoidance_factor,  # Use scaled density
            "distance_to_target" => distance_to_target,
            "high_density_area" => high_density_area,
            "obstacle_avoidance_factor" => obstacle_avoidance_factor,
            "beliefs" => serialize_beliefs(beliefs)
        )
        
        # Write to output file
        open(INFERRED_STATE_PATH, "w") do file
            JSON.print(file, output_dict)
        end
        println("Successfully saved inferred state")
    catch e
        println("Error saving inferred state: $e")
    end
    
    println("\nInference step completed successfully")
end

# Run main function and handle any unexpected errors
try
    main()
catch e
    println("Error in main function: $e")
    println(stacktrace())
    rethrow(e)  # Ensure the error is visible to caller
end