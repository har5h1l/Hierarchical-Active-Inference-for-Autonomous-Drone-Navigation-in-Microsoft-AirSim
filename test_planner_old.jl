
#!/usr/bin/env julia

# Simple path planning script for testing with old behavior (always takes direct path)
println("Starting OLD test planning script...")

# Constants
const OBS_INPUT_PATH = joinpath("interface", "obs_input.json")
const NEXT_WP_PATH = joinpath("interface", "next_waypoint.json")

# Create interface directory if it doesn't exist
mkpath("interface")

# Load observation data
obs_data = open(OBS_INPUT_PATH) do f
    using JSON
    JSON.parse(f)
end

# Get drone and target positions
drone_pos = obs_data["drone_position"]
target_pos = obs_data["target_position"]

# Calculate direct path (no obstacle avoidance)
direction = target_pos - drone_pos
distance = sqrt(sum((direction .^ 2)))
normalized_dir = direction ./ distance
step_size = min(3.0, distance * 0.6)
next_waypoint = drone_pos + normalized_dir .* step_size

# Write the waypoint to the output file
open(NEXT_WP_PATH, "w") do f
    x, y, z = next_waypoint
    json_str = """{"next_waypoint": [$x, $y, $z]}"""
    write(f, json_str)
end

println("Direct waypoint saved")
