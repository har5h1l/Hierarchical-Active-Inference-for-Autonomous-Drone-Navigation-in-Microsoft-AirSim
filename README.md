# Hierarchical Active Inference for Autonomous Drone Navigation in Microsoft AirSim with Environmentally Aware Adaptive Planning

This project implements autonomous drone navigation in the AirSim simulator using Active Inference, a computational framework based on the Free Energy Principle. The system interfaces AirSim with Julia for probabilistic inference using RxInfer.jl, with ZeroMQ handling inter-process communication.

## Project Structure

### Code Organization
- `actinf/` - Julia package implementing the active inference framework
  - `src/actinf.jl` - Main module definition
  - `src/StateSpace.jl` - State space representation for drone navigation
  - `src/Inference.jl` - Belief updating and probabilistic inference
  - `src/Planning.jl` - Action selection and planning using Expected Free Energy
  - `zmq_server.jl` - ZeroMQ server for Julia-Python communication
- `airsim/` - Python interface to AirSim simulator
  - `Sensory_Input_Processing.py` - Processing sensor data from AirSim
  - `settings.json` - AirSim configuration
- `run_inference.jl` - Entry point for belief state inference
- `run_planning.jl` - Entry point for action planning
- `precompile.jl` - Script to precompile dependencies for faster startup
- `rebuild.jl` - Script to rebuild the Julia package
- `test.py` - Python controller for simulation execution

### System Components

#### 1. State Space Representation (StateSpace.jl)
The drone's state is represented by the `DroneState` structure with four key dimensions:
- **Distance**: Distance to target (meters)
- **Azimuth**: Horizontal angle to target (radians)
- **Elevation**: Vertical angle to target (radians)
- **Suitability**: Environmental safety measure (0-1)

Raw sensory data is encapsulated in `DroneObservation` objects that contain:
- Drone position (x, y, z)
- Drone orientation (quaternion)
- Target position (x, y, z)
- Nearest obstacle distances
- Voxel grid representation of obstacles
- Obstacle density

The system handles coordinate transformations between global and egocentric reference frames using quaternion-based rotations.

#### 2. Belief Updating and Inference (Inference.jl)
The system maintains probabilistic beliefs about each state dimension:
- Beliefs are represented as discretized probability distributions
- `DroneBeliefs` structure maintains distributions and discretization ranges
- Temporal smoothing is applied for stable belief evolution
- Circular quantities (angles) are handled with appropriate distance metrics

The belief updating process:
1. Initialize uniform distributions across state dimensions
2. Update with Gaussian kernels centered on observations
3. Apply Bayesian updates with adaptive kernel widths
4. Calculate expected state from current beliefs

#### 3. Action Planning (Planning.jl)
The planning module selects optimal actions using a two-stage process:

**Stage 1: Safety Filtering**
- Generate candidate waypoints in a sphere around current position
- Direct-to-target waypoints are included with various step sizes
- Predict next state for each waypoint
- Calculate suitability score using sigmoid-based functions
- Discard unsafe waypoints (suitability < threshold)

**Stage 2: EFE Optimization**
- For remaining safe waypoints, calculate Expected Free Energy (EFE)
- EFE balances pragmatic value (progress toward goal) and epistemic value (uncertainty reduction)
- Select waypoints with lowest EFE scores

#### 4. Environmental Perception (Sensory_Input_Processing.py)
The `EnvironmentScanner` class processes raw sensor data from AirSim:
- Lidar point cloud processing
- Obstacle detection and clustering
- Density calculation in local regions
- Voxel grid representation of the environment

#### 5. Communication Architecture
The system uses ZeroMQ for efficient inter-process communication:
- **ZMQ Server** (zmq_server.jl): Persistent Julia process that maintains compiled state
- **ZMQ Interface** (Python): Handles communication with the Julia server
- Health monitoring with heartbeats
- Automatic server recovery if communication fails

## Detailed Algorithms and Formulas

### 1. Suitability Calculation
Suitability is a safety metric that combines obstacle distance and density information:

```
suitability = obstacle_weight * safety_factor + density_weight * density_factor
```

Where:
- `safety_factor = 1.0 / (1.0 + exp(-steepness_distance * (obstacle_distance - cutoff_distance)))`
- `density_factor = 1.0 / (1.0 + exp(steepness_density * (obstacle_density - cutoff_density)))`

Parameters:
- `obstacle_weight = 0.7` (Default weight for obstacle distance)
- `density_weight = 0.3` (Default weight for obstacle density)
- `cutoff_distance = 2.5` (Meters - threshold for rapid safety decrease)
- `steepness_distance = 3.0` (Controls transition sharpness)
- `cutoff_density = 0.2` (Density threshold)
- `steepness_density = 10.0` (Controls density transition sharpness)

### 2. Belief Updating
Beliefs are updated using a Bayesian approach with Gaussian kernels:

```
belief = 0.8 * prior_belief + 0.2 * new_likelihood
```

Where:
- `new_likelihood` is a Gaussian kernel centered on the observation
- Circular quantities (angles) use special distance calculations
- Kernel width adapts based on distance and state variables
- Target preference is incorporated to bias toward goal

### 3. Preference Calculation
The system evaluates states using a preference model with multiple components:

**Distance Preference:**
```
preference = (1.0 - normalized_dist)^2 * exp(-distance_scaling * distance)
```

**Angle Preference:**
```
preference = baseline + (1.0 - baseline) * (cos(abs_angle) + 1)^sharpness / 2
```

**Suitability Preference:**
```
if suitability < threshold:
    preference = suitability * (suitability / threshold) * 0.5
else:
    preference = 0.5 + 0.5 * (suitability - threshold) / (1.0 - threshold)
```

### 4. Expected Free Energy Calculation
EFE combines pragmatic and epistemic components:

```
EFE = pragmatic_value + epistemic_value
```

Where:
- `pragmatic_value = -pragmatic_weight * (preference_score + distance_bonus) * action_magnitude`
- `epistemic_value = -epistemic_weight * 0.5 * total_entropy * action_magnitude`
- `total_entropy` is calculated from belief distributions
- Lower EFE values are better

### 5. Adaptive Planning Parameters
The system dynamically adjusts planning parameters based on environmental suitability:

**Waypoint Radius (Step Size):**
```
adaptive_radius = MIN_RADIUS + suitability_factor * (MAX_RADIUS - MIN_RADIUS)
```
- Range: 0.5m (MIN_RADIUS) to 3.0m (MAX_RADIUS)
- Low suitability → smaller radius (safer, shorter steps)
- High suitability → larger radius (faster progress)

**Policy Length:**
```
adaptive_policy_length = MAX_POLICY_LEN - suitability_factor * (MAX_POLICY_LEN - MIN_POLICY_LEN)
```
- Range: 2 (MIN_POLICY_LEN) to 5 (MAX_POLICY_LEN) steps
- Low suitability → longer policy (more careful planning)
- High suitability → shorter policy (less planning needed)

**Waypoint Sampling:**
```
adaptive_waypoint_count = MAX_WAYPOINTS - suitability_factor * (MAX_WAYPOINTS - MIN_WAYPOINTS)
```
- Range: 15 (MIN_WAYPOINTS) to 75 (MAX_WAYPOINTS)
- Low suitability → more waypoints (greater exploration)
- High suitability → fewer waypoints (more direct paths)

## Navigation Process Flow

### 1. Initialization
- Start AirSim simulator
- Launch ZeroMQ server for Julia-Python communication
- Connect to the drone and take off
- Define target location

### 2. Main Navigation Loop
For each iteration:

1. **Sensory Processing**
   - Get current drone position and orientation
   - Scan environment for obstacles using LiDAR
   - Calculate obstacle distances and density

2. **Belief Inference**
   - Create DroneObservation from sensory data
   - Convert to egocentric DroneState
   - Update belief distributions
   - Calculate expected state

3. **Action Planning**
   - Generate candidate waypoints with adaptive radius
   - Filter out unsafe waypoints based on suitability
   - Calculate EFE for remaining waypoints
   - Select optimal next waypoint

4. **Execution**
   - Move drone to selected waypoint
   - Monitor for obstacles during movement
   - Abort and replan if unexpected obstacles appear

5. **Termination Check**
   - Check if target has been reached
   - Update trajectory visualization

## Real-time Obstacle Avoidance

During movement execution, the system continuously monitors for obstacles:

1. **Obstacle Detection**
   - Continuously scans surroundings using LiDAR
   - Detects obstacles within configurable safety threshold (2.0-2.5m)

2. **Emergency Protocol** when obstacles are detected:
   - Abort current motion
   - Hover safely in place
   - Trigger new inference and planning cycle

## Special Handling for Goal Approach

When the drone is close to the target:

1. **Direct Path Prioritization**
   - When distance < 10m, the system adds more direct target waypoints
   - Suitability boost for direct paths when clear of obstacles
   - Finer step size gradations for precise target approach

2. **Enhanced Target Preference**
   - When distance < 5m, target preference weight increases
   - Direct routes to target receive stronger suitability bonuses
   - Shorter steps for final positioning accuracy

## Error Handling and Robustness

The system implements several mechanisms for robust operation:

1. **ZMQ Communication Resilience**
   - Automatic socket reset on communication failures
   - Server health monitoring with heartbeats
   - Automatic restart if server becomes unresponsive

2. **Precompilation**
   - Julia code is precompiled to minimize startup latency
   - Package dependencies are pre-loaded
   - Status monitoring during precompilation

3. **Fallback Mechanisms**
   - If no safe waypoints are found, select best available option
   - When close to target, balance safety with goal progress
   - Emergency hover in place if all options are unsafe

## Requirements

- Julia 1.7+
- Python 3.7+
- AirSim simulator
- ZMQ, RxInfer.jl, PyCall.jl, and other dependencies

## Setup

1. Install AirSim following the [official documentation](https://microsoft.github.io/AirSim/build_windows/)
2. Install Julia dependencies:
```julia
using Pkg
Pkg.add(["ZMQ", "PyCall", "RxInfer", "StaticArrays", "LinearAlgebra", "JSON"])
```
3. Install Python dependencies:
```python
pip install airsim numpy zmq matplotlib sklearn
```
4. Configure AirSim settings.json for your simulation environment

## Running the System

1. Start AirSim simulator
2. Run the Python controller:
```
python test.py
```

This will:
- Precompile Julia dependencies
- Start the ZeroMQ server
- Initialize the drone in AirSim
- Execute the navigation sequence with real-time visualization

## Advanced Configuration

The system can be fine-tuned via several key parameters:

1. **Safety Parameters**
   - `SUITABILITY_THRESHOLD`: Minimum acceptable waypoint suitability (default: 0.5)
   - `MARGIN`: Safety margin for obstacle avoidance (default: 1.5m)
   - `CUTOFF_DISTANCE`: Distance threshold for safety calculations (default: 2.5m)

2. **Planning Parameters**
   - `WAYPOINT_SAMPLE_COUNT`: Number of waypoints to generate (default: 75)
   - `POLICY_LENGTH`: Number of steps to plan ahead (default: 3)
   - `DENSITY_RADIUS`: Radius for density evaluation (default: 5.0m)

3. **Navigation Parameters**
   - `TARGET_LOCATION`: Destination coordinates [x, y, z] (default: [-20.0, -20.0, -30.0])
   - `ARRIVAL_THRESHOLD`: Distance to consider target reached (default: 1.2m)
   - `MAX_ITERATIONS`: Maximum navigation steps (default: 100)
