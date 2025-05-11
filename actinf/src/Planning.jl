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
global SUITABILITY_THRESHOLD = 0.5  # Can be overridden by server
const DEFAULT_OBSTACLE_WEIGHT = 0.7  # Default weight for obstacle distance
const DEFAULT_DENSITY_WEIGHT = 0.3   # Default weight for obstacle density
const CUTOFF_DISTANCE = 2.5  # Meters
const STEEPNESS_DISTANCE = 3.0
const CUTOFF_DENSITY = 0.2
const STEEPNESS_DENSITY = 10.0

# Target approach thresholds
const CLOSE_TO_TARGET_THRESHOLD = 10.0  # Distance in meters to prioritize direct paths to target
const VERY_CLOSE_TO_TARGET_THRESHOLD = 5.0  # Distance in meters for even stronger target preference

# Minimum acceptable suitability for waypoints - increased to ensure safety
const MIN_ACCEPTABLE_SUITABILITY = 0.3  # Waypoints below this are considered risky

# New constants for preventing crash near target
const EMERGENCY_SUITABILITY_THRESHOLD = 0.25  # Absolute minimum suitability to consider a path
const MIN_OBSTACLE_DISTANCE = 1.2  # Minimum obstacle distance in meters to allow paths

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
    suitability_threshold = 0.3,
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
    
    total_preference = (
        model.distance_weight * distance_pref +
        model.angle_weight * angle_combined +
        model.suitability_weight * suitability_pref
    ) / (model.distance_weight + model.angle_weight + model.suitability_weight)
    
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
"""
function generate_waypoints(current_position::SVector{3, Float64}, distance::Float64, num_angles::Int, num_elevations::Int)::Vector{SVector{3, Float64}}
    waypoints = Vector{SVector{3, Float64}}()
    
    # Include current position as a waypoint (stay in place)
    push!(waypoints, current_position)
    
    for elev_idx in 1:num_elevations
        elevation = π * (elev_idx - 1) / (num_elevations - 1) - π/2  # from -pi/2 to pi/2
        
        for angle_idx in 1:num_angles
            azimuth = 2π * (angle_idx - 1) / num_angles
            
            dx = distance * cos(elevation) * cos(azimuth)
            dy = distance * cos(elevation) * sin(azimuth)
            dz = distance * sin(elevation)
            
            waypoint = current_position + SVector{3, Float64}(dx, dy, dz)
            push!(waypoints, waypoint)
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
    local_density = obstacle_density
    
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
            local_density = nearby_voxels / local_volume
        end
    end
    
    # Calculate base suitability
    suitability = StateSpace.calculate_suitability(
        local_obstacle_distance, 
        local_density, 
        obstacle_weight=obstacle_weight, 
        density_weight=DEFAULT_DENSITY_WEIGHT*density_weight
    )
    
    # Special case: When close to target, evaluate direct path to target
    if current_to_target_dist <= CLOSE_TO_TARGET_THRESHOLD
        # If this waypoint would bring us closer to the target
        if dist_to_target < current_to_target_dist
            # Check if it's a clear path (no obstacles near)
            has_clear_path = local_obstacle_distance >= CUTOFF_DISTANCE
            
            # If path is clear or very close to target, boost suitability
            if has_clear_path || current_to_target_dist < VERY_CLOSE_TO_TARGET_THRESHOLD
                # Boost target weight and preference when path is clear and we're close
                suitability_boost = 0.2 
                suitability = min(1.0, suitability + suitability_boost)
                
                # If very close to target, boost even more
                if current_to_target_dist < VERY_CLOSE_TO_TARGET_THRESHOLD
                    suitability = min(1.0, suitability + 0.1)
                end
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
                      epistemic_weight=0.2)
    # Use expected_state from beliefs
    expected = Inference.expected_state(beliefs)
    
    # Calculate pragmatic value using preference model
    preference_score = evaluate_preference(state, preference_model)
    
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
                 num_policies::Int = 5, obstacle_weight::Float64 = DEFAULT_OBSTACLE_WEIGHT)

Select the best actions by first filtering out unsafe paths based on suitability,
then evaluating Expected Free Energy on remaining safe candidates.
Dynamically adjusts waypoint radius, policy length, and waypoint sampling based on environment suitability.
"""
function select_action(state::StateSpace.DroneState, beliefs::Inference.DroneBeliefs, planner::ActionPlanner, 
                     current_position::SVector{3, Float64}, target_position::SVector{3, Float64};
                     obstacle_distance::Float64 = 10.0, obstacle_density::Float64 = 0.0, 
                     num_policies::Int = 5, obstacle_weight::Float64 = DEFAULT_OBSTACLE_WEIGHT)
    # Constants for adaptive parameters
    MIN_RADIUS = 0.5
    MAX_RADIUS = 3.0
    MIN_POLICY_LEN = 2
    MAX_POLICY_LEN = 5
    MIN_WAYPOINTS = 15
    MAX_WAYPOINTS = 75
    
    # Calculate distance to target
    current_to_target = target_position - current_position
    current_to_target_dist = norm(current_to_target)
    target_direction = normalize(current_to_target)
    
    # Increase waypoint radius when close to target to allow reaching it in one step
    close_to_target = current_to_target_dist <= CLOSE_TO_TARGET_THRESHOLD
    
    # Dynamically adjust parameters based on state suitability
    # For suitability, higher values (closer to 1) mean safer navigation conditions
    suitability_factor = clamp(state.suitability, 0.0, 1.0)  # Ensure it's in valid range
    
    # 1. Adjust waypoint radius (step size) based on both suitability and obstacle density
    # Low suitability -> smaller radius (safer, shorter steps)
    # High suitability -> larger radius (faster exploitation)
    
    # Scale radius based on both obstacle density and closest obstacle distance
    obstacle_density_factor = 1.0 - clamp(obstacle_density, 0.0, 0.9)  # Higher density reduces steps
    obstacle_distance_factor = clamp(obstacle_distance / 10.0, 0.0, 1.0)  # Normalize distance to 0-1 range
    
    # Combined safety factor - weighs both density and distance
    combined_safety_factor = 0.7 * obstacle_distance_factor + 0.3 * obstacle_density_factor
    
    # More gradual scaling of waypoint radius based on safety factors
    adaptive_radius = MIN_RADIUS + combined_safety_factor * (MAX_RADIUS - MIN_RADIUS)
    
    # If close to target, ensure radius is at least enough to reach the target
    # But don't override safety constraints when very close to obstacles
    if close_to_target
        if obstacle_distance > MIN_OBSTACLE_DISTANCE
            # Allow reaching target in one step if safe
            adaptive_radius = max(adaptive_radius, min(current_to_target_dist * 1.1, MAX_RADIUS))
        else
            # When close to obstacles, prioritize safety with smaller steps
            adaptive_radius = min(adaptive_radius, 1.0)
            println("Close to target but also close to obstacles, using conservative radius: $(adaptive_radius)")
        end
    end
    
    println("Adaptive radius: $(adaptive_radius)m, Obstacle distance: $(obstacle_distance)m, Density: $(obstacle_density)")
    
    # 2. Adjust policy length - inverse relationship with suitability
    # Low suitability -> longer policy (more careful planning)
    # High suitability -> shorter policy (less planning needed)
    adaptive_policy_length = MAX_POLICY_LEN - combined_safety_factor * (MAX_POLICY_LEN - MIN_POLICY_LEN)
    adaptive_policy_length = round(Int, adaptive_policy_length)
    
    # 3. Adjust number of waypoints - inverse relationship with suitability
    # Low suitability -> more waypoints (greater caution, more exploration)
    # High suitability -> fewer waypoints (less exploration needed)
    adaptive_waypoint_count = MAX_WAYPOINTS - combined_safety_factor * (MAX_WAYPOINTS - MIN_WAYPOINTS)
    adaptive_waypoint_count = round(Int, adaptive_waypoint_count)
    
    # Calculate appropriate number of angles and elevations to achieve target waypoint count
    # Target is adaptive_waypoint_count total waypoints
    # Formula: num_angles * num_elevations + 1 ≈ adaptive_waypoint_count
    # where +1 accounts for the current position waypoint
    total_angle_divisions = max(4, round(Int, sqrt(adaptive_waypoint_count - 1)))
    num_angles = max(4, round(Int, total_angle_divisions * 1.5))  # More horizontal than vertical divisions
    num_elevations = max(2, round(Int, total_angle_divisions / 1.5))  # Fewer vertical divisions
    
    # Generate waypoints with adaptive radius for adaptive exploration
    waypoints = generate_waypoints(current_position, adaptive_radius, num_angles, num_elevations)
    
    # Add direct-to-target waypoints with various step sizes
    # When close to target, add more direct waypoints with finer gradations
    if close_to_target
        # More steps when close to target for finer control
        step_ratios = if current_to_target_dist < VERY_CLOSE_TO_TARGET_THRESHOLD
            [0.2, 0.4, 0.6, 0.8, 1.0]  # Very fine steps when very close
        else
            [0.3, 0.6, 0.9, 1.0]  # Regular steps when somewhat close
        end
        
        for ratio in step_ratios
            # Create direct waypoint toward target
            step_size = ratio * min(adaptive_radius, current_to_target_dist * 1.1)
            direct_waypoint = current_position + step_size * target_direction
            push!(waypoints, direct_waypoint)
        end
    else
        # Standard approach when not close
        for step in [0.3, 0.6, 0.9] .* adaptive_radius
            direct_waypoint = current_position + step * target_direction
            push!(waypoints, direct_waypoint)
        end
    end
    
    # Step 1: Generate candidate waypoints and predict next states
    all_waypoints = waypoints
    all_states = Vector{StateSpace.DroneState}()
    all_actions = Vector{SVector{3, Float64}}()
    all_suitabilities = Vector{Float64}()
    all_distances = Vector{Float64}()
    
    println("Generated $(length(all_waypoints)) candidate waypoints")
    
    # Step 2: Predict next state for each candidate and calculate suitability
    for wp in all_waypoints
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
        action = wp - current_position
        
        push!(all_states, next_state)
        push!(all_actions, action)
        push!(all_suitabilities, next_state.suitability)
        push!(all_distances, next_state.distance)
    end
    
    # Step 3: Early elimination based on dynamic suitability threshold
    # If close to target, use a more nuanced approach to balance safety and goal-reaching
    local_suitability_threshold = SUITABILITY_THRESHOLD
    
    # Adjust threshold based on proximity to target
    if close_to_target
        # Scale threshold based on closeness, but never below emergency threshold
        distance_factor = current_to_target_dist / CLOSE_TO_TARGET_THRESHOLD
        # More forgiving threshold when closer to target, but with a hard safety floor
        local_suitability_threshold = max(
            EMERGENCY_SUITABILITY_THRESHOLD,  
            SUITABILITY_THRESHOLD * (0.7 + 0.3 * distance_factor)
        )
        println("Close to target, adjusted suitability threshold to $(local_suitability_threshold)")
    end
    
    # Filter waypoints based on suitability
    safe_indices = findall(s -> s >= local_suitability_threshold, all_suitabilities)
    
    # If no waypoints passed the threshold but we're close to target,
    # try to find at least one that meets the emergency threshold
    if isempty(safe_indices) && close_to_target
        safe_indices = findall(s -> s >= EMERGENCY_SUITABILITY_THRESHOLD, all_suitabilities)
        println("Using emergency suitability threshold: $(EMERGENCY_SUITABILITY_THRESHOLD)")
    end
    
    # If still empty, try to at least find the safest option
    if isempty(safe_indices)
        # Find the waypoint with the highest suitability
        _, max_idx = findmax(all_suitabilities)
        # Only use it if it meets minimum acceptable standards
        if all_suitabilities[max_idx] >= MIN_ACCEPTABLE_SUITABILITY
            safe_indices = [max_idx]
            println("No satisfactory waypoints found. Using best available with suitability: $(all_suitabilities[max_idx])")
        else
            println("WARNING: All waypoints have very low suitability (best: $(all_suitabilities[max_idx]))")
        end
    end
    
    # Step 4: If still no safe waypoints, generate a minimal safe action
    if isempty(safe_indices)
        # Generate a minimal safe action by moving away from nearest obstacle
        println("Generating minimal safe action due to lack of safe waypoints")
        
        # If we have obstacle distance information, try to move away from nearest obstacle
        if obstacle_distance < 5.0
            # Calculate direction to nearest obstacle from belief state
            # This is a simplification; in practice you would use the actual obstacle position
            # For our purposes, a random direction with a safety bias is sufficient
            
            # Generate a short step in a random direction with slight upward bias (in NED, that's negative z)
            theta = rand() * 2π
            step_size = MIN_RADIUS / 2  # Very short step
            safe_action = SVector{3, Float64}(
                step_size * cos(theta),  # x component
                step_size * sin(theta),  # y component
                -0.2 * step_size         # z component (slightly upward in NED)
            )
            
            # Create a safe waypoint from current position
            safe_waypoint = current_position + safe_action
            
            # Simulate this safe action
            safe_state = simulate_transition(
                state, 
                safe_waypoint, 
                target_position, 
                beliefs.voxel_grid, 
                obstacle_distance, 
                obstacle_density, 
                planner.density_weight,
                obstacle_weight=obstacle_weight
            )
            
            # Use this safe action
            return [(safe_waypoint, safe_state, safe_action)]
        end
        
        # If we get here, we have no obstacle information and no safe waypoints
        # Return current position as a fallback (stay in place)
        println("WARNING: No safe waypoints found, will hover in place")
        stay_action = SVector{3, Float64}(0.0, 0.0, 0.0)
        return [(current_position, state, stay_action)]
    end
    
    # Use only safe waypoints for further evaluation
    safe_waypoints = all_waypoints[safe_indices]
    safe_states = all_states[safe_indices]
    safe_actions = all_actions[safe_indices]
    
    # Step 5: Rank the safe waypoints by expected free energy
    efes = [calculate_efe(s, state, beliefs, planner.preference_model,
                          planner.pragmatic_weight, planner.epistemic_weight, 
                          safe_actions[i]) for (i, s) in enumerate(safe_states)]
    
    # Create policy items (waypoint, state, action) sorted by EFE
    sorted_indices = sortperm(efes)
    sorted_policies = [(safe_waypoints[i], safe_states[i], safe_actions[i], efes[i], all_suitabilities[safe_indices[i]]) 
                       for i in sorted_indices]
    
    # Create and return the top policies
    top_n = min(num_policies, length(sorted_policies))
    return [(p[1], p[2], p[3]) for p in sorted_policies[1:top_n]]
end

end # module
