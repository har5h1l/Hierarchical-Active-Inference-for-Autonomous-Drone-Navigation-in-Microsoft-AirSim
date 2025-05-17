module Planning

export ActionPlanner, select_action, generate_waypoints, calculate_efe
export simulate_transition, calculate_suitability_for_waypoint
export PreferenceModel, evaluate_preference

using LinearAlgebra
using StaticArrays
using ..StateSpace
using ..Inference

# Constants for suitability calculation and filtering
# Make SUITABILITY_THRESHOLD global and mutable so it can be updated from zmq_server.jl
global SUITABILITY_THRESHOLD = 0.75  # Increased from 0.6 to 0.75 for safer path selection
const DEFAULT_OBSTACLE_WEIGHT = 0.8  # Increased from 0.7
const DEFAULT_DENSITY_WEIGHT = 0.4   # Increased from 0.3
const CUTOFF_DISTANCE = 3.0  # Increased from 2.5 Meters
const STEEPNESS_DISTANCE = 4.0  # Increased from 3.0
const CUTOFF_DENSITY = 0.15  # Reduced from 0.2
const STEEPNESS_DENSITY = 12.0  # Increased from 10.0

# Target approach thresholds
const CLOSE_TO_TARGET_THRESHOLD = 10.0  # Distance in meters to prioritize direct paths to target
const VERY_CLOSE_TO_TARGET_THRESHOLD = 5.0  # Distance in meters for even stronger target preference

# Minimum acceptable suitability for waypoints
const MIN_ACCEPTABLE_SUITABILITY = 0.5  # Increased from 0.2 - Waypoints below this are considered risky

"""
    PreferenceModel

Defines how the agent evaluates states, with explicit scoring functions for each state component.
Higher preference scores indicate more desirable states.
"""
struct PreferenceModel
    # Preference parameters for each state dimension
    distance_weight::Float64           # Weight for distance preference
    distance_scaling::Float64          # Scaling factor for distance
    angle_weight::Float64              # Weight for angle (azimuth/elevation) preferences
    angle_sharpness::Float64           # How sharply to prefer on-target angles
    suitability_weight::Float64        # Weight for environmental suitability
    suitability_threshold::Float64     # Minimum acceptable suitability
    
    # Normalization settings
    max_distance::Float64              # Maximum distance for normalization
end

"""
    PreferenceModel(; kwargs...)

Constructor with default parameters for the preference model.
"""
function PreferenceModel(;
    distance_weight = 2.0,         # Increased from 1.0 to emphasize distance more
    distance_scaling = 0.05,       # Reduced from 0.1 to make distance preference decay more gradually
    angle_weight = 0.5,           # Reduced from 0.8 to prioritize distance over angle
    angle_sharpness = 3.0,        # Reduced from 5.0 for smoother angle preferences
    suitability_weight = 1.0,      # Reduced from 1.5 to prioritize distance
    suitability_threshold = 0.55,
    max_distance = 50.0
)
    return PreferenceModel(
        distance_weight,
        distance_scaling,
        angle_weight,
        angle_sharpness,
        suitability_weight,
        suitability_threshold,
        max_distance
    )
end

"""
    evaluate_distance_preference(distance::Float64, model::PreferenceModel)::Float64

Evaluate preference for a given distance to target.
Modified to more strongly prefer shorter distances.
"""
function evaluate_distance_preference(distance::Float64, model::PreferenceModel)::Float64
    # Normalize distance to [0,1] range
    normalized_dist = min(distance / model.max_distance, 1.0)
    
    # Transform to preference score: 1 when distance=0, approaching 0 as distance increases
    # Using quadratic decay for stronger preference of shorter distances
    preference = (1.0 - normalized_dist)^2 * exp(-model.distance_scaling * distance)
    
    return preference
end

"""
    evaluate_angle_preference(angle::Float64)::Float64

Evaluate preference for a given angle (azimuth or elevation).
For S2/S3: 0 degrees (directly toward target) is best.
Includes a constant baseline preference (0.2) to gently bias the agent toward facing the target.
"""
function evaluate_angle_preference(angle::Float64, sharpness::Float64)::Float64
    # Convert angle to absolute value in radians (we care about magnitude, not direction)
    abs_angle = abs(angle)
    
    # Constant baseline preference - gently bias toward facing target
    constant_baseline = 0.2
    
    # Prefer angles closer to 0 (directly toward target)
    # Using cosine: 1 when angle=0, decreasing as angle increases
    # Raised to a power for sharper preference peak
    variable_preference = (cos(abs_angle) + 1) / 2
    variable_preference = variable_preference ^ sharpness
    
    # Combine the baseline and variable components
    # This ensures there's always some preference for facing the target,
    # but also maintains the existing angle preference structure
    preference = constant_baseline + (1.0 - constant_baseline) * variable_preference
    
    return preference
end

"""
    evaluate_suitability_preference(suitability::Float64, threshold::Float64)::Float64

Evaluate preference for environmental suitability.
For S4: higher is better, with threshold for minimum acceptable suitability.
"""
function evaluate_suitability_preference(suitability::Float64, threshold::Float64)::Float64
    # Apply threshold with smooth transition
    if suitability < threshold
        # Rapidly decreasing preference below threshold
        return suitability * (suitability / threshold) * 0.5
    else
        # Linear scaling above threshold, with bonus for exceeding threshold
        normalized = 0.5 + 0.5 * (suitability - threshold) / (1.0 - threshold)
        return normalized
    end
end

"""
    evaluate_preference(state::StateSpace.DroneState, model::PreferenceModel)::Float64

Calculate overall preference score for a drone state.
Higher scores indicate more desirable states.
"""
function evaluate_preference(state::StateSpace.DroneState, model::PreferenceModel)::Float64
    # Evaluate individual components
    distance_pref = evaluate_distance_preference(state.distance, model)
    azimuth_pref = evaluate_angle_preference(state.azimuth, model.angle_sharpness)
    elevation_pref = evaluate_angle_preference(state.elevation, model.angle_sharpness)
    suitability_pref = evaluate_suitability_preference(state.suitability, model.suitability_threshold)
    
    # Combine preferences with weights
    angle_combined = (azimuth_pref + elevation_pref) / 2
    
    # Dynamically adjust weights based on suitability
    # In high-density obstacle environments (low suitability), we want to:
    # 1. Increase weight for suitability (obstacle avoidance)
    # 2. Decrease weight for distance to target
    # But maintain angle preference to keep drone oriented properly
    
    # Default weights
    dist_weight = model.distance_weight
    suit_weight = model.suitability_weight
    
    # If suitability is low (high obstacle density), adjust weights to prioritize safety
    if state.suitability < 0.4
        # Calculate how much to reduce distance weight and increase suitability weight
        # More aggressive adjustment for very low suitability
        adjustment_factor = (0.4 - state.suitability) * 2.5  # Factor ranges from 0.0-1.0
        
        # Scale the distance weight down, more reduction for lower suitability
        dist_weight = max(model.distance_weight * (1.0 - adjustment_factor * 0.5), 0.5)
        
        # Scale the suitability weight up, more increase for lower suitability
        suit_weight = model.suitability_weight * (1.0 + adjustment_factor)
    end
    
    total_preference = (
        dist_weight * distance_pref +
        model.angle_weight * angle_combined +
        suit_weight * suitability_pref
    ) / (dist_weight + model.angle_weight + suit_weight)
    
    return total_preference
end

"""
ActionPlanner contains parameters for action selection and planning
"""
struct ActionPlanner
    # Action space parameters
    max_step_size::Float64      # Maximum movement distance per step
    num_angles::Int             # Number of angles to consider
    num_step_sizes::Int         # Number of step sizes to consider
    
    # Free energy components weights
    pragmatic_weight::Float64   # Weight for goal-seeking behavior
    epistemic_weight::Float64   # Weight for uncertainty-reducing behavior
    risk_weight::Float64        # Weight for obstacle avoidance
    
    # Safety parameters
    safety_distance::Float64    # Minimum safe distance from obstacles
    density_weight::Float64     # Weight for obstacle density in suitability
    
    # Preference model
    preference_model::PreferenceModel
end

"""
    ActionPlanner(; kwargs...)

Constructor with default parameters
"""
function ActionPlanner(;
    max_step_size = 0.5,
    num_angles = 8,
    num_step_sizes = 3,
    pragmatic_weight = 1.0,
    epistemic_weight = 0.2,
    risk_weight = 2.0,
    safety_distance = 1.5,
    density_weight = 1.0,
    preference_model = PreferenceModel()
)
    return ActionPlanner(
        max_step_size,
        num_angles,
        num_step_sizes,
        pragmatic_weight,
        epistemic_weight,
        risk_weight,
        safety_distance,
        density_weight,
        preference_model
    )
end

"""
    generate_waypoints(current_position::SVector{3, Float64}, distance::Float64, num_angles::Int, num_elevations::Int)::Vector{SVector{3, Float64}}

Generate waypoints at a fixed radius from current_position in azimuth and elevation sweeps.
Ensures all waypoints are at least at the specified distance from current position.
"""
function generate_waypoints(current_position::SVector{3, Float64}, distance::Float64, num_angles::Int, num_elevations::Int)::Vector{SVector{3, Float64}}
    waypoints = Vector{SVector{3, Float64}}()
    
    # Set minimum allowed distance to ensure meaningful movement
    min_distance = max(1.0, distance * 0.75)  # At least 1.0 meter or 75% of requested distance
    
    # We no longer include current position as a waypoint to avoid "stay in place" actions
    # that can cause the drone to get stuck with negligible movements
    
    for elev_idx in 1:num_elevations
        elevation = π * (elev_idx - 1) / (num_elevations - 1) - π/2  # from -pi/2 to pi/2
        
        for angle_idx in 1:num_angles
            azimuth = 2π * (angle_idx - 1) / num_angles
            
            # Calculate waypoint position
            dx = distance * cos(elevation) * cos(azimuth)
            dy = distance * cos(elevation) * sin(azimuth)
            dz = distance * sin(elevation)
            
            movement_vector = SVector{3, Float64}(dx, dy, dz)
            
            # Ensure minimum movement distance
            movement_magnitude = norm(movement_vector)
            if movement_magnitude < 0.01  # Safeguard against zero vectors
                continue  # Skip this waypoint
            end
            
            # Scale up vectors that are too short
            if movement_magnitude < min_distance
                scaling_factor = min_distance / movement_magnitude
                movement_vector = movement_vector * scaling_factor
            end
            
            waypoint = current_position + movement_vector
            push!(waypoints, waypoint)
        end
    end
    
    # If no waypoints were generated (unlikely but possible), add one in the upward direction
    if isempty(waypoints)
        println("⚠️ No waypoints generated - adding fallback options")
        # Add waypoints in cardinal directions
        for direction in [
            SVector{3, Float64}(min_distance, 0.0, 0.0),  # Forward
            SVector{3, Float64}(0.0, min_distance, 0.0),  # Right
            SVector{3, Float64}(-min_distance, 0.0, 0.0), # Back
            SVector{3, Float64}(0.0, -min_distance, 0.0)  # Left
        ]
            push!(waypoints, current_position + direction)
        end
    end
    
    return waypoints
end

"""
    calculate_suitability_for_waypoint(waypoint::SVector{3, Float64}, obstacle_distance::Float64, obstacle_density::Float64, density_weight::Float64 = 1.0)::Float64

Calculate suitability score for a waypoint based on distance to nearest obstacle and obstacle density.
Higher suitability means the waypoint is safer to navigate to.
Uses sigmoid-like scaling for more predictable behavior.
"""
function calculate_suitability_for_waypoint(waypoint::SVector{3, Float64}, obstacle_distance::Float64, obstacle_density::Float64, density_weight::Float64 = 1.0)::Float64
    # Use constants for more consistent suitability calculation
    return StateSpace.calculate_suitability(
        obstacle_distance, 
        obstacle_density, 
        obstacle_weight=DEFAULT_OBSTACLE_WEIGHT, 
        density_weight=DEFAULT_DENSITY_WEIGHT*density_weight,
        cutoff_distance=CUTOFF_DISTANCE,
        steepness_distance=STEEPNESS_DISTANCE,
        cutoff_density=CUTOFF_DENSITY,
        steepness_density=STEEPNESS_DENSITY
    )
end

"""
    simulate_transition(state::StateSpace.DroneState, waypoint::SVector{3, Float64}, target_position::SVector{3, Float64}, 
                        voxel_grid::Vector{SVector{3, Float64}}, obstacle_distance::Float64, obstacle_density::Float64, 
                        density_weight::Float64 = 1.0; obstacle_weight::Float64 = DEFAULT_OBSTACLE_WEIGHT)::StateSpace.DroneState

Simulate the predicted next state given the current state and a waypoint.
Computes distance to target, azimuth and elevation from waypoint to target,
and calculates suitability based on obstacle sensory data (voxel grid, distance and density).
"""
function simulate_transition(state::StateSpace.DroneState, waypoint::SVector{3, Float64}, target_position::SVector{3, Float64}, 
                            voxel_grid::Vector{SVector{3, Float64}}, obstacle_distance::Float64, obstacle_density::Float64, 
                            density_weight::Float64 = 1.0; obstacle_weight::Float64 = DEFAULT_OBSTACLE_WEIGHT)::StateSpace.DroneState
    # Vector from waypoint to target
    to_target = target_position - waypoint
    dist_to_target = norm(to_target)
    
    # Also calculate current distance from current position to target 
    # (we need this to see if we're getting closer)
    current_position = waypoint - (waypoint - state.distance * normalize(to_target))
    current_to_target_dist = norm(target_position - current_position)
    
    # Compute azimuth and elevation angles
    azimuth = atan(to_target[2], to_target[1])  # atan(y, x)
    elevation = atan(to_target[3], sqrt(to_target[1]^2 + to_target[2]^2))
    
    # Calculate suitability using the voxel grid information for more accurate obstacle assessment
    local_obstacle_distance = obstacle_distance
    localGo_density = obstacle_density
    
    # If we have voxel data, use it to refine our suitability calculation
    if !isempty(voxel_grid)
        # Calculate distances from waypoint to each obstacle voxel
        distances_to_voxels = [norm(waypoint - voxel) for voxel in voxel_grid]
        
        # Find closest obstacle (if any voxels are available)
        if !isempty(distances_to_voxels)
            closest_obstacle_distance = minimum(distances_to_voxels)
            # Take the minimum of the sensed obstacle distance and the voxel-based calculation
            local_obstacle_distance = min(obstacle_distance, closest_obstacle_distance)
      # Calculate local density near the waypoint
            # Count voxels within 3m radius of the waypoint
            nearby_voxels = count(d -> d < 3.0, distances_to_voxels)
            local_volume = 4/3 * π * 3.0^3  # Volume of 3m radius sphere
            localGo_density = nearby_voxels / local_volume
            
            # Enhanced obstacle avoidance: Check if there's a high density cluster in any direction
            # Look for dense clusters of obstacles by dividing space into octants
            if nearby_voxels > 5
                # Calculate vectors from waypoint to each voxel
                voxel_vectors = [voxel - waypoint for voxel in voxel_grid]
                
                # Function to check which octant a vector belongs to
                function get_octant(v)
                    x_positive = v[1] >= 0
                    y_positive = v[2] >= 0
                    z_positive = v[3] >= 0
                    return (x_positive, y_positive, z_positive)
                end
                
                # Count voxels in each octant
                octant_counts = Dict{Tuple{Bool, Bool, Bool}, Int}()
                for (i, v) in enumerate(voxel_vectors)
                    if distances_to_voxels[i] < 5.0  # Only consider voxels within 5m
                        octant = get_octant(v)
                        octant_counts[octant] = get(octant_counts, octant, 0) + 1
                    end
                end
                
                # Check if any octant has a high concentration of obstacles
                max_octant_count = maximum(values(octant_counts); init=0)
                if max_octant_count > 5  # If an octant has more than 5 voxels
                    # Further penalize the density estimate to avoid this area
                    density_penalty = max_octant_count / nearby_voxels * 0.5
                    localGo_density = localGo_density * (1.0 + density_penalty)
                    
                    # Also reduce the effective obstacle distance if dense obstacles are present
                    if max_octant_count > 10 && local_obstacle_distance > 1.5
                        local_obstacle_distance = local_obstacle_distance * 0.85
                    end
                end
            end
        end
    end
    
    # Dynamically adjust obstacle weight based on distance to target
    # Gradually reduce obstacle weight as we get closer to target
    adaptive_obstacle_weight = obstacle_weight
    if dist_to_target < 5.0
        # Reduce obstacle weight when very close to target to prioritize reaching it
        dist_factor = max(0.7, dist_to_target / 5.0)  # Scale from 0.7-1.0 based on distance
        adaptive_obstacle_weight = obstacle_weight * dist_factor
    end
    
    # Calculate base suitability with adaptive obstacle weight
    suitability = StateSpace.calculate_suitability(
        local_obstacle_distance, 
        localGo_density, 
        obstacle_weight=adaptive_obstacle_weight, 
        density_weight=DEFAULT_DENSITY_WEIGHT*density_weight
    )
      # Special case: When close to target, evaluate direct path to target
    if current_to_target_dist <= CLOSE_TO_TARGET_THRESHOLD
        # If this waypoint would bring us closer to the target
        if dist_to_target < current_to_target_dist
            # More aggressive suitability boosting when close to target
            # The closer we are, the more boost we apply
            proximity_factor = 1.0
            
            if current_to_target_dist < VERY_CLOSE_TO_TARGET_THRESHOLD
                # When very close to target (<5m), apply stronger boost
                proximity_factor = 1.5
                
                # Calculate how directly this waypoint leads to the target
                # Higher score when waypoint is directly toward target
                to_waypoint = waypoint - current_position
                to_waypoint_norm = normalize(to_waypoint)
                to_target_norm = normalize(target_position - current_position)
                directness = dot(to_waypoint_norm, to_target_norm)  # 1.0 = perfectly aligned
                
                # If waypoint leads almost directly to target, boost suitability more
                if directness > 0.8
                    proximity_factor += (directness - 0.8) * 2.0
                end
                
                # If this would get us extremely close to target, massive boost
                if dist_to_target < 2.0
                    proximity_factor += (2.0 - dist_to_target)
                end
            end
            
            # Check if it's a clear path (no obstacles near)
            has_clear_path = local_obstacle_distance >= CUTOFF_DISTANCE
            
            # Improved clear path detection - check if path to target has no obstacles
            # Calculate the direct vector to target
            to_target_vector = target_position - waypoint
            to_target_distance = norm(to_target_vector)
            to_target_direction = normalize(to_target_vector)
            
            # Consider path is clear if:
            # 1. No obstacles are near the waypoint OR
            # 2. Obstacle distance is greater than distance to target OR
            # 3. Obstacles are not between waypoint and target
            
            # Check if obstacles are likely between waypoint and target
            obstacles_between = false
            
            if !isempty(voxel_grid) && to_target_distance > 1.0
                # Find voxels that are potentially in our path
                # Project each voxel onto the line to target
                for voxel in voxel_grid
                    # Vector from waypoint to voxel
                    waypoint_to_voxel = voxel - waypoint
                    
                    # Distance from waypoint to voxel
                    dist_to_voxel = norm(waypoint_to_voxel)
                    
                    # Only consider voxels closer than the target
                    if dist_to_voxel > to_target_distance
                        continue
                    end
                      # Project voxel onto the path to target
                    projection = dot(waypoint_to_voxel, to_target_direction)
                    
                    # Only consider voxels in front of us toward the target
                    if projection <= 0
                        continue
                    end
                    
                    # Calculate perpendicular distance from voxel to path
                    # |v - (v·d)d| where d is unit direction vector
                    # Use a more conservative perpendicular distance calculation that considers
                    # the drone's size and potential drift during movement                    proj_point = waypoint + projection * to_target_direction
                    perp_dist = norm(voxel - proj_point)
                    
                    # Consider as obstacle if close to direct path
                    # Increased the perpendicular distance threshold from 1.0 to 1.5 for safety
                    if perp_dist < 1.5 && projection < to_target_distance
                        obstacles_between = true
                        break
                    end
                end
            end
            
            # Path is clear if no obstacles are between waypoint and target
            direct_path_clear = has_clear_path || !obstacles_between
            
            # If path is clear or very close to target, boost suitability
            if direct_path_clear || current_to_target_dist < VERY_CLOSE_TO_TARGET_THRESHOLD
                # Boost target weight and preference when path is clear and we're close
                suitability_boost = 0.2 * proximity_factor
                suitability = min(1.0, suitability + suitability_boost)
                
                # If very close to target, boost even more
                if current_to_target_dist < VERY_CLOSE_TO_TARGET_THRESHOLD
                    suitability = min(1.0, suitability + 0.1 * proximity_factor)
                end
            end
            
            # Special case for almost reaching target - boost suitability unless obstacles are very close
            if dist_to_target < 2.0 && local_obstacle_distance > 1.0
                target_reach_boost = 0.3 * (1.0 - dist_to_target/2.0)  # More boost as we get closer
                suitability = min(1.0, suitability + target_reach_boost)
            end
            
            # Penalize paths with obstacles between waypoint and target when close to target
            if current_to_target_dist < VERY_CLOSE_TO_TARGET_THRESHOLD && obstacles_between
                obst_penalty = 0.3 * min(1.0, projection / to_target_distance)
                suitability = max(0.1, suitability - obst_penalty)
            end
        end
    end
    
    # Construct new DroneState with updated fields
    return StateSpace.DroneState(
        distance = dist_to_target,
        azimuth = azimuth,
        elevation = elevation,
        suitability = suitability
    )
end

"""
    calculate_efe(state, beliefs, action, preference_model;
                 pragmatic_weight=1.0, epistemic_weight=0.2)

Calculate the Expected Free Energy focusing on pragmatic and epistemic values only.
Risk is handled separately through early elimination of unsafe waypoints.
"""
function calculate_efe(state::StateSpace.DroneState, 
                      beliefs::Inference.DroneBeliefs, 
                      action::SVector{3, Float64}, 
                      preference_model::PreferenceModel;
                      pragmatic_weight=1.0, 
                      epistemic_weight=0.2,
                      obstacle_density=0.0,
                      obstacle_distance=10.0)
    # Use expected_state from beliefs
    expected = Inference.expected_state(beliefs)
    
    # Calculate pragmatic value using preference model
    preference_score = evaluate_preference(state, preference_model)
    
    # Adjust weights based on obstacle density - in high density environments, 
    # reduce preference for target approach and increase emphasis on exploration
    adjusted_pragmatic_weight = pragmatic_weight
    adjusted_epistemic_weight = epistemic_weight
    
    # Define the density and distance thresholds for adjustment
    high_density_threshold = 0.3  # Obstacle density above this is considered high
    close_obstacle_threshold = 3.0  # Obstacles closer than this are considered close
    
    # Check if we're in a high obstacle density environment
    in_high_density = obstacle_density > high_density_threshold
    obstacles_close = obstacle_distance < close_obstacle_threshold
    
    # Apply adjustment if either condition is met
    if in_high_density || obstacles_close
        # Calculate adjustment factor (0.0 to 1.0)
        density_factor = min(1.0, obstacle_density / 0.5)  # Normalize to 0-1 range
        distance_factor = max(0.0, 1.0 - (obstacle_distance / close_obstacle_threshold))
        
        # Use the more significant factor
        adjustment_factor = max(density_factor, distance_factor)
        
        # Reduce pragmatic weight (target seeking) in high density areas
        adjusted_pragmatic_weight = pragmatic_weight * (1.0 - adjustment_factor * 0.4)
        
        # Increase epistemic weight (exploration) in high density areas
        adjusted_epistemic_weight = epistemic_weight * (1.0 + adjustment_factor)
    end
    
    # Add distance reduction bonus - strongly reward actions that decrease distance
    action_mag = norm(action)
    distance_reduction = expected.distance - state.distance
    distance_bonus = distance_reduction > 0 ? 2.0 * distance_reduction : 0.0
    
    # Modified pragmatic value with distance bonus
    pragmatic_value = -pragmatic_weight * (preference_score + distance_bonus) * action_mag
    
    # Calculate epistemic value from belief distributions
    dist_entropy = -sum(beliefs.distance_belief .* log.(beliefs.distance_belief .+ 1e-10))
    azim_entropy = -sum(beliefs.azimuth_belief .* log.(beliefs.azimuth_belief .+ 1e-10))
    elev_entropy = -sum(beliefs.elevation_belief .* log.(beliefs.elevation_belief .+ 1e-10))
    suit_entropy = -sum(beliefs.suitability_belief .* log.(beliefs.suitability_belief .+ 1e-10))
    dens_entropy = -sum(beliefs.density_belief .* log.(beliefs.density_belief .+ 1e-10))
    
    total_entropy = dist_entropy + azim_entropy + elev_entropy + suit_entropy + dens_entropy
    epistemic_value = -epistemic_weight * 0.5 * total_entropy * action_mag
    
    # Total Expected Free Energy (lower is better)
    # No risk value component anymore - handled by early filtering based on suitability
    return pragmatic_value + epistemic_value
end

"""
    select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner, 
                 current_position::SVector{3, Float64}, target_position::SVector{3, Float64};
                 obstacle_distance::Float64 = 10.0, obstacle_density::Float64 = 0.0, 
                 num_policies::Int = 5, obstacle_weight::Float64 = DEFAULT_OBSTACLE_WEIGHT,
                 suitability_threshold::Float64 = 0.75)

Select the best actions by first filtering out unsafe paths based on suitability,
then evaluating Expected Free Energy on remaining safe candidates.
Dynamically adjusts waypoint radius, policy length, and waypoint sampling based on environment suitability.
"""
function select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner, 
                     current_position::SVector{3, Float64}, target_position::SVector{3, Float64};
                     obstacle_distance::Float64 = 10.0, obstacle_density::Float64 = 0.0, 
                     num_policies::Int = 5, obstacle_weight::Float64 = DEFAULT_OBSTACLE_WEIGHT,
                     suitability_threshold::Float64 = 0.75)  # Add explicit parameter with strong default
      MIN_RADIUS = 1.0  # Increased from 0.5 to ensure minimum action length of 1 meter
    MAX_RADIUS = 3.5  # Slightly increased maximum radius for more exploration
    MIN_POLICY_LEN = 2
    MAX_POLICY_LEN = 5
    MIN_WAYPOINTS = 25  # Increased from 15 to provide more options in dense environments
    MAX_WAYPOINTS = 100 # Increased from 75 to allow more thorough exploration when needed
    
    # Calculate distance to target
    current_to_target = target_position - current_position
    current_to_target_dist = norm(current_to_target)
    target_direction = normalize(current_to_target)
    
    # Increase waypoint radius when close to target to allow reaching it in one step
    close_to_target = current_to_target_dist <= CLOSE_TO_TARGET_THRESHOLD
    
    # Dynamically adjust parameters based on state suitability
    # For suitability, higher values (closer to 1) mean safer navigation conditions
    suitability_factor = clamp(state.suitability, 0.0, 1.0)  # Ensure it's in valid range
      # 1. Adjust waypoint radius (step size) - inverse relationship with suitability
    # Low suitability -> smaller radius (safer, shorter steps)
    # High suitability -> larger radius (faster exploitation)
    adaptive_radius = MIN_RADIUS + suitability_factor * (MAX_RADIUS - MIN_RADIUS)
    
    # Adapt step size based on obstacle conditions
    # When obstacles are close or density is high, ensure we have a substantial minimum step
    # to avoid getting stuck with tiny movements
    obstacle_factor = clamp(1.0 - (obstacle_density * 5.0), 0.3, 1.0)  # Scale down with density
    distance_factor = clamp(obstacle_distance / 10.0, 0.3, 1.0)        # Scale up with distance
    
    # Combine obstacle factors - in high-density or close-obstacle scenarios, 
    # maintain minimum step size but don't go too large
    environment_factor = (obstacle_factor + distance_factor) / 2.0
    
    # Ensure minimum radius of 1.0 meter in all cases
    min_required_radius = 1.0 + (1.0 - environment_factor) * 0.5  # 1.0 to 1.5 meters
    adaptive_radius = max(adaptive_radius, min_required_radius)
    
    # If close to target, ensure radius is at least enough to reach the target
    if close_to_target
        adaptive_radius = max(adaptive_radius, current_to_target_dist * 1.1)
    end
    
    # 2. Adjust policy length - inverse relationship with suitability
    # Low suitability -> longer policy (more careful planning)
    # High suitability -> shorter policy (less planning needed)
    adaptive_policy_length = MAX_POLICY_LEN - suitability_factor * (MAX_POLICY_LEN - MIN_POLICY_LEN)
    adaptive_policy_length = round(Int, adaptive_policy_length)
      # 3. Adjust number of waypoints - inverse relationship with suitability
    # Low suitability -> more waypoints (greater caution, more exploration)
    # High suitability -> fewer waypoints (less exploration needed)
    adaptive_waypoint_count = MAX_WAYPOINTS - suitability_factor * (MAX_WAYPOINTS - MIN_WAYPOINTS)
    
    # Increase waypoints significantly in high obstacle density or when obstacles are near
    # This provides more options to find viable paths in complex environments
    if obstacle_density > 0.1 || obstacle_distance < 5.0
        density_scale = clamp(obstacle_density * 10.0, 1.0, 3.0)  # Scale from 1.0 to 3.0
        distance_scale = clamp(5.0 / max(obstacle_distance, 1.0), 1.0, 2.5)  # Scale from 1.0 to 2.5
        obstacle_scale = max(density_scale, distance_scale)
        
        # Apply scaling to waypoint count, being more aggressive when both density is high and obstacles are close
        if obstacle_density > 0.1 && obstacle_distance < 5.0
            obstacle_scale = obstacle_scale * 1.2  # Additional 20% boost
        end
        
        adaptive_waypoint_count = min(adaptive_waypoint_count * obstacle_scale, MAX_WAYPOINTS)
        println("⚠️ Increasing waypoint count due to obstacles - Scale: $(round(obstacle_scale, digits=2))")
    end
    
    # Make sure we have a reasonable number of waypoints
    adaptive_waypoint_count = max(MIN_WAYPOINTS, round(Int, adaptive_waypoint_count))
    
    # Adjust parameters based on distance to target
    if current_to_target_dist > 20.0
        # When far from target, use more exploration
        adaptive_waypoint_count = min(round(Int, adaptive_waypoint_count * 1.2), MAX_WAYPOINTS)
        adaptive_radius = max(adaptive_radius, MIN_RADIUS * 2)  # Ensure larger steps for faster progress
    elseif close_to_target
        # When close to target, focus sampling more toward target direction
        adaptive_waypoint_count = min(round(Int, adaptive_waypoint_count * 1.1), MAX_WAYPOINTS)
        adaptive_radius = max(adaptive_radius, current_to_target_dist * 0.8)  # Allow getting closer to target
    end
    
    println("Adaptive planning parameters:")
    println("  Radius: $(adaptive_radius) m")
    println("  Policy Length: $(adaptive_policy_length) steps")
    println("  Waypoint Count: $(adaptive_waypoint_count) waypoints")
    
    # Generate candidate waypoints with adaptive radius
    all_waypoints = Vector{SVector{3, Float64}}()
    
    # Core waypoints - general sampling around current position
    general_waypoints = generate_waypoints(
        current_position, 
        adaptive_radius, 
        # Ensure reasonable amount of waypoints 
        round(Int, adaptive_waypoint_count * 0.7),
        3  # Elevations
    )
    append!(all_waypoints, general_waypoints)
    
    # Target-biased waypoints - sampling in direction of target
    # These help ensure we always consider paths that lead toward the target
    # Add a biased set of waypoints in the target direction
    target_biased_waypoints = generate_biased_waypoints(
        current_position,
        target_position,
        adaptive_radius,
        round(Int, adaptive_waypoint_count * 0.3)  # Use 30% of waypoints for target-biased sampling
    )
    append!(all_waypoints, target_biased_waypoints)
    
    # Step 1: Process all candidate waypoints
    all_states = Vector{StateSpace.DroneState}()
    all_actions = Vector{SVector{3, Float64}}()
    all_suitabilities = Vector{Float64}()
    all_distances = Vector{Float64}()
    
    println("Generated $(length(all_waypoints)) candidate waypoints")
      # Step 2: Predict next state for each candidate and calculate suitability
    for wp in all_waypoints
        # Calculate action vector and magnitude
        action = wp - current_position
        action_magnitude = norm(action)
        
        # Skip waypoints that are too close to current position (negligible movement)
        if action_magnitude < 1.0
            continue
        end
        
        next_state = simulate_transition(
            state, 
            wp, 
            target_position, 
            beliefs.voxel_grid, 
            obstacle_distance, 
            obstacle_density, 
            planner.density_weight,
            obstacle_weight=obstacle_weight
        )
        
        push!(all_states, next_state)
        push!(all_actions, action)
        push!(all_suitabilities, next_state.suitability)
        push!(all_distances, next_state.distance)
    end
    
    # Use the provided suitability threshold, not the global one
    # Also adjust when close to target for more precision
    effective_threshold = close_to_target ? 
                          suitability_threshold * 0.8 : # Slightly more permissive when close
                          suitability_threshold
    
    println("Using suitability threshold: $(effective_threshold)")
    
    # Find indices of all waypoints that meet minimum suitability
    safe_indices = findall(s -> s >= effective_threshold, all_suitabilities)
    
    # IMPROVED SAFETY STRATEGY: Always prioritize higher suitability paths
    # If we found safe paths, take the top 30% by suitability
    if !isempty(safe_indices)
        println("Found $(length(safe_indices)) waypoints above threshold $(effective_threshold)")
        
        # Sort indices by suitability (descending)
        sorted_by_suitability = sort(safe_indices, by=i -> all_suitabilities[i], rev=true)
          # Take the top 30% most suitable paths (with a minimum of 5)
        top_count = max(5, round(Int, length(sorted_by_suitability) * 0.3))
        top_indices = sorted_by_suitability[1:min(top_count, length(sorted_by_suitability))]
        
        # Find paths with very high suitability (clear paths)
        high_suitability_indices = findall(i -> all_suitabilities[i] > 0.8, sorted_by_suitability)
        has_clear_paths = !isempty(high_suitability_indices)
        
        # In high density environments, prioritize clear paths over target proximity
        if obstacle_density > 0.3 && has_clear_paths
            # Find the subset of high suitability paths that don't increase distance too much
            acceptable_distance_indices = filter(i -> 
                all_distances[sorted_by_suitability[i]] < current_to_target_dist * 1.2, 
                high_suitability_indices)
            
            if !isempty(acceptable_distance_indices)
                println("High density environment: Prioritizing $(length(acceptable_distance_indices)) clear paths")
                # Replace top indices with high suitability paths
                clear_path_indices = sorted_by_suitability[acceptable_distance_indices]
                top_indices = clear_path_indices
            end
        end
        
        # If using the top 30% would discard paths that also move closer to the target,
        # include some of these target-approaching paths even if they aren't in the top 30%
        # by suitability
        if length(sorted_by_suitability) > top_count
            # Find paths that bring us closer to target
            remaining_indices = sorted_by_suitability[top_count+1:end]
            closer_to_target_indices = filter(i -> all_distances[i] < current_to_target_dist, remaining_indices)
            
            # Sort these by distance to target (ascending)
            if !isempty(closer_to_target_indices)
                sorted_by_distance = sort(closer_to_target_indices, by=i -> all_distances[i])
                
                # Add up to 5 more paths that get closer to target with acceptable suitability
                # This ensures we don't neglect target approach when focusing on suitability
                additional_count = min(5, length(sorted_by_distance))
                append!(top_indices, sorted_by_distance[1:additional_count])
                
                println("Added $additional_count additional paths that approach target")
            end
        end
        
        # These are our final filtered indices
        safe_indices = top_indices
    end
    
    # If no safe paths found, dynamically lower threshold to find at least some viable path
    if isempty(safe_indices)
        println("No waypoints meet threshold $(effective_threshold), dynamically lowering threshold")
        
        # Try a range of decreasing thresholds
        for fallback_threshold in [0.4, 0.3, 0.2]
            safe_indices = findall(s -> s >= fallback_threshold, all_suitabilities)
            if !isempty(safe_indices)
                println("Found $(length(safe_indices)) waypoints with fallback threshold $(fallback_threshold)")
                
                # If we're using a lower threshold, take only the very top paths by suitability
                sorted_by_suitability = sort(safe_indices, by=i -> all_suitabilities[i], rev=true)
                top_indices = sorted_by_suitability[1:min(3, length(sorted_by_suitability))]
                safe_indices = top_indices
                break
            end
        end
    end
    
    # If still no safe paths found, use the most suitable path available
    if isempty(safe_indices)
        println("WARNING: No safe paths found, using best available")
        best_index = argmax(all_suitabilities)
        safe_indices = [best_index]
    end
    
    # Filter states and actions to only the safe indices
    filtered_states = all_states[safe_indices]
    filtered_actions = all_actions[safe_indices]
    filtered_suitabilities = all_suitabilities[safe_indices]
    filtered_distances = all_distances[safe_indices]
    
    println("Filtered to $(length(filtered_states)) safe waypoints")
      # Step 3: Calculate Expected Free Energy (EFE) for filtered paths
    efe_scores = Float64[]
    
    # Use current distance as reference with a margin for initial distance
    # (Fixed previous error with undefined obs_data variable)
    initial_distance = current_to_target_dist * 1.2  # Add 20% as a margin
      for (i, next_state) in enumerate(filtered_states)
        action = filtered_actions[i]
        next_distance = filtered_distances[i]
        
        # Check if this is a high suitability path (clear path)
        is_high_suitability = filtered_suitabilities[i] > 0.8
        
        # Calculate EFE using the existing function
        efe = calculate_efe(
            next_state,
            beliefs,
            action,
            planner.preference_model,
            pragmatic_weight=planner.pragmatic_weight,
            epistemic_weight=planner.epistemic_weight,
            obstacle_density=obstacle_density,  # Pass obstacle density to EFE calculation
            obstacle_distance=obstacle_distance  # Pass obstacle distance to EFE calculation
        )
        
        # Apply special bonus for high suitability paths in high density environments
        # This ensures clear paths are strongly preferred when in dense obstacle areas
        if is_high_suitability && (obstacle_density > 0.3 || obstacle_distance < 3.0)
            # Calculate density factor (more bonus in higher density)
            density_factor = clamp(obstacle_density * 2.0, 0.5, 1.5)
            
            # Stronger bonus when both density is high and obstacles are close
            if obstacle_density > 0.3 && obstacle_distance < 3.0
                high_suitability_bonus = 2.0 * density_factor
            else
                high_suitability_bonus = 1.5 * density_factor
            end
            
            println("Applying high suitability bonus ($(round(high_suitability_bonus, digits=2))x) in dense environment")
            efe = efe * high_suitability_bonus
        end
        
        # Use percentage-based approach for distance bonus calculation
        if next_distance < current_to_target_dist
            # Progressive bonus based on percentage improvement rather than absolute distance
            distance_percentage_improvement = ((current_to_target_dist - next_distance) / initial_distance) * 100.0
            
            # More significant bonuses for larger percentage improvements
            if distance_percentage_improvement > 5.0  # Significant improvement (>5% of total journey)
                distance_bonus = 1.5 + (distance_percentage_improvement / 5.0)  # Base bonus plus scaled improvement
            else
                distance_bonus = 1.0 + (distance_percentage_improvement / 2.0)  # Smaller bonus for minor improvements
            end
            
            # Apply progressive suitability bonus with higher multiplier
            suitability_bonus = 1.0 + (filtered_suitabilities[i] * 1.5)  # Reward high suitability paths more
            
            # Calculate percentage of journey completed
            journey_percentage_complete = 100.0 - ((next_distance / initial_distance) * 100.0)
            
            # Apply journey progress bonus - more aggressive in final stages
            if journey_percentage_complete > 90.0  # Final 10% of journey
                journey_bonus = 1.0 + ((journey_percentage_complete - 90.0) / 5.0)  # Up to 3x bonus in final approach
            elseif journey_percentage_complete > 75.0  # Last quarter of journey
                journey_bonus = 1.0 + ((journey_percentage_complete - 75.0) / 15.0)  # Up to 2x bonus 
            else
                journey_bonus = 1.0  # No additional bonus earlier in journey
            end
            
            # Apply combined bonuses
            efe = efe * distance_bonus * suitability_bonus * journey_bonus
        end
        
        # Additional suitability threshold for paths getting closer to target in final approach
        journey_percentage_complete = 100.0 - ((next_distance / initial_distance) * 100.0) 
        
        # Apply more lenient suitability requirements when very close to target
        if journey_percentage_complete > 90.0 && next_distance < current_to_target_dist
            # In final 10% of journey, accept slightly lower suitability if making progress
            min_acceptable_suitability = suitability_threshold * 0.8
            if filtered_suitabilities[i] >= min_acceptable_suitability
                # Apply a final approach bonus to encourage completing the journey
                final_approach_bonus = 1.0 + ((journey_percentage_complete - 90.0) / 2.0)
                efe = efe * final_approach_bonus
            end
        end
        
        push!(efe_scores, efe)
    end
    
    # Step 5: Policy selection - choose the best actions based on EFE
    # Sort by EFE ascending (lower is better)
    sorted_indices = sortperm(efe_scores)
      # Use the adaptive policy length instead of fixed num_policies
    top_k = min(adaptive_policy_length, length(sorted_indices))
    
    # Create selected tuples of (action, efe)
    selected = [(filtered_actions[sorted_indices[i]], efe_scores[sorted_indices[i]]) for i in 1:top_k]
    
    # Post-processing: Filter out any actions with negligible movement
    # This ensures replanning isn't triggered for tiny movements
    MIN_ACTION_MAGNITUDE = 1.0  # Minimum 1 meter movement
    valid_actions = []
    
    for (action, efe) in selected
        action_magnitude = norm(action)
        
        if action_magnitude >= MIN_ACTION_MAGNITUDE
            push!(valid_actions, (action, efe))
        else
            println("⚠️ Filtering out negligible action with magnitude $(round(action_magnitude, digits=2))m")
            # If this was the best action, print warning
            if action == filtered_actions[sorted_indices[1]]
                println("⚠️ Warning: Best action had negligible movement - attempting to find alternatives")
            end
        end
    end
    
    # If all actions were filtered out, use the best available regardless of magnitude
    if isempty(valid_actions) && !isempty(selected)
        println("⚠️ All actions filtered due to negligible movement - using best available")
        best_action = selected[1]
        push!(valid_actions, best_action)
        
        # Also add a scaled-up version of the best action to ensure movement
        action, efe = best_action
        action_magnitude = norm(action)
        
        if action_magnitude > 0.01  # Avoid division by zero
            scaling_factor = MIN_ACTION_MAGNITUDE / action_magnitude
            scaled_action = action * scaling_factor
            println("ℹ️ Adding scaled-up action with magnitude $(round(norm(scaled_action), digits=2))m")
            push!(valid_actions, (scaled_action, efe * 1.1))  # Slightly worse EFE for scaled action
        end
    end
    
    # Step 7: Debug info
    if !isempty(sorted_indices) && !isempty(efe_scores)
        best_idx = sorted_indices[1]
        best_efe = efe_scores[best_idx]
        best_action = filtered_actions[best_idx]
        best_suitability = filtered_states[best_idx].suitability
        best_distance = filtered_distances[best_idx]
          println("Best EFE: $(best_efe), Action: $(best_action), Suitability: $(best_suitability)")
        println("Current distance to target: $(current_to_target_dist), Next distance: $(best_distance)")
    end
    
    # Return the valid actions (filtered to ensure minimum movement distance)
    return valid_actions
end

# New helper function to generate waypoints biased toward the target
function generate_biased_waypoints(current_position::SVector{3, Float64}, 
                                  target_position::SVector{3, Float64},
                                  radius::Float64,
                                  num_waypoints::Int)::Vector{SVector{3, Float64}}
    
    # Calculate direction to target
    to_target = target_position - current_position
    distance_to_target = norm(to_target)
    
    # Set minimum action length for meaningful movement
    min_step_size = max(1.0, radius * 0.75)  # At least 1.0 meter or 75% of radius
    
    # If target is very close, return the target position
    if distance_to_target < min_step_size
        return [target_position]
    end
    
    direction_to_target = to_target / distance_to_target
    
    # Generate waypoints
    waypoints = Vector{SVector{3, Float64}}()
    
    # First, add a direct waypoint toward the target with appropriate step size
    direct_step = min(radius, distance_to_target * 0.8)  # 80% of distance or max radius
    
    # Ensure minimum step size
    direct_step = max(direct_step, min_step_size)
    
    direct_waypoint = current_position + direction_to_target * direct_step
    push!(waypoints, direct_waypoint)
    
    # Add waypoints with slight variations around the target direction
    # to provide options while still maintaining target-oriented movement
    for i in 1:num_waypoints-1
        # Vary the angle from the target direction (narrower cone than general sampling)
        max_angle = π/6  # 30-degree cone
        angle_x = (rand() * 2 - 1) * max_angle
        angle_y = (rand() * 2 - 1) * max_angle
        angle_z = (rand() * 2 - 1) * max_angle * 0.5  # Less vertical variation
        
        # Create rotation matrix
        Rx = [1 0 0; 0 cos(angle_x) -sin(angle_x); 0 sin(angle_x) cos(angle_x)]
        Ry = [cos(angle_y) 0 sin(angle_y); 0 1 0; -sin(angle_y) 0 cos(angle_y)]
        Rz = [cos(angle_z) -sin(angle_z) 0; sin(angle_z) cos(angle_z) 0; 0 0 1]
        
        R = Rx * Ry * Rz
        
        # Apply rotation to target direction
        perturbed_direction = R * [direction_to_target[1], direction_to_target[2], direction_to_target[3]]
        perturbed_direction = perturbed_direction / norm(perturbed_direction)
        
        # Vary the step size but ensure it meets minimum
        base_step_size = rand() * radius * 0.8 + radius * 0.2  # Between 20% and 100% of radius
        step_size = max(base_step_size, min_step_size)
        
        # Calculate waypoint
        waypoint = current_position + SVector{3, Float64}(perturbed_direction) * step_size
        push!(waypoints, waypoint)
    end
    
    return waypoints
end

end  # End of Planning module
