# Hierarchical Active Inference for Autonomous Drone Navigation in Microsoft AirSim with Environmentally Aware Adaptive Planning

This project implements autonomous drone navigation in the AirSim simulator using Active Inference, a computational framework based on the Free Energy Principle. The system interfaces AirSim with Julia for probabilistic inference using RxInfer.jl, with ZeroMQ handling inter-process communication.

## Project Structure

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

## Communication Architecture

The system uses ZeroMQ (0MQ) for efficient inter-process communication:

1. **ZeroMQ Server** (`zmq_server.jl`):
   - Runs as a persistent Julia process
   - Maintains loaded Julia modules and compiled state between calls
   - Significantly reduces latency compared to the previous approach

2. **Communication Protocol**:
   - Request/Reply pattern for synchronous operations
   - Serialized data transfer using JSON
   - Status tracking with `zmq_server_running.status`

## Active Inference Framework

### State Space Representation

The drone's state is represented by the `DroneState` structure with four key dimensions:

- **Distance**: Distance to target (meters)
- **Azimuth**: Horizontal angle to target (radians)
- **Elevation**: Vertical angle to target (radians)
- **Suitability**: Environmental safety measure (0-1)

States are computed from `DroneObservation` objects that contain raw sensory data including drone position, orientation, target position, obstacle distances, and obstacle density.

### Suitability Calculation

The suitability metric is a crucial safety measure that indicates how navigable a location is:

- Higher values (closer to 1.0) indicate safer navigation conditions
- Calculated using two main components with sigmoid-like scaling for predictable behavior:
  - **Obstacle Distance**: Using sigmoid function `1.0 / (1.0 + exp(-steepness_distance * (obstacle_distance - cutoff_distance)))`
  - **Obstacle Density**: Using inverted sigmoid function `1.0 / (1.0 + exp(steepness_density * (obstacle_density - cutoff_density)))`
- Configurable parameters control transition sharpness and cutoff points:
  ```
  OBSTACLE_WEIGHT = 0.7
  DENSITY_WEIGHT = 0.3
  CUTOFF_DISTANCE = 2.5  # Meters
  STEEPNESS_DISTANCE = 3.0
  CUTOFF_DENSITY = 0.2
  STEEPNESS_DENSITY = 10.0
  ```

### Belief Updating

The `Inference` module maintains probabilistic beliefs about each state dimension:

- Beliefs are represented as discretized probability distributions
- Updates incorporate new observations using Bayesian inference
- Circular quantities (angles) are handled with appropriate distance metrics
- `DroneBeliefs` structure maintains both the distributions and their discretization ranges
- Temporal smoothing is applied to ensure stable belief evolution

### Two-Stage Action Selection

Action selection now uses a two-stage process that separates safety from optimization:

1. **Coarse Elimination (Safety)**:
   - Generate candidate waypoints around current position
   - Predict next state for each candidate waypoint
   - Calculate suitability score for each waypoint
   - **Immediately discard** waypoints with suitability below `SUITABILITY_THRESHOLD` (default: 0.5)
   - Only waypoints with acceptable suitability proceed to EFE evaluation

2. **Fine-Grained Selection (Optimization)**:
   - For remaining safe waypoints, calculate Expected Free Energy (EFE)
   - EFE now only balances two components (no risk penalty):
     - **Pragmatic Value**: Progress toward goal (distance reduction)
     - **Epistemic Value**: Uncertainty reduction (information gain)
   - Select waypoints with lowest EFE scores

This approach ensures:
- Dangerous paths are never considered during final planning
- EFE correctly focuses on optimal goal-seeking and exploration among safe options
- Risk avoidance is handled structurally in the planning process

### Adaptive Planning Parameters

Planning parameters dynamically adjust based on the environment's safety (suitability):

1. **Waypoint Radius (Step Size)**
   - Range: 0.5m (MIN_RADIUS) to 3.0m (MAX_RADIUS)
   - Low suitability → smaller radius (safer, shorter steps)
   - High suitability → larger radius (faster progress)

2. **Policy Length**
   - Range: 2 (MIN_POLICY_LEN) to 5 (MAX_POLICY_LEN) steps
   - Low suitability → longer policy (more careful planning)
   - High suitability → shorter policy (less planning needed)

3. **Waypoint Sampling**
   - Range: 15 (MIN_WAYPOINTS) to 75 (MAX_WAYPOINTS)
   - Low suitability → more waypoints (greater exploration)
   - High suitability → fewer waypoints (more direct paths)

### Real-time Obstacle Avoidance

During movement execution, the system continuously monitors for obstacles:

1. **Obstacle Detection**
   - Continuously scans surroundings using LiDAR and depth sensors
   - Detects obstacles within configurable safety threshold (2.0-2.5m)

2. **Emergency Protocol** when obstacles are detected:
   - Abort current motion
   - Hover safely in place
   - Trigger new inference and planning cycle

### Planning Process Flow

The complete planning process now follows this flow:

1. **Waypoint Generation** (no change)
   - Generate spherical distribution of waypoints based on adaptive radius
   - Include target-directed waypoints at various step sizes
   - Include "stay-in-place" option

2. **Predict Next State** for each candidate
   - Calculate updated distance, azimuth and elevation to target
   - Compute suitability score using sigmoid-like scaling

3. **Early Elimination Based on Suitability**
   - Discard waypoints with suitability < SUITABILITY_THRESHOLD
   - Report number of waypoints passing safety filter

4. **EFE Evaluation on Filtered Waypoints**
   - Calculate EFE based on pragmatic and epistemic values only
   - No risk penalty (handled by filtering)

5. **Policy Selection**
   - Select top-k actions with lowest EFE (k = adaptive_policy_length)
   - Return policy as sequence of actions

## Requirements

- Julia 1.7+
- Python 3.7+
- AirSim simulator
- RxInfer.jl, PyCall.jl, ZMQ.jl, and other dependencies

## Setup

1. Install AirSim following the [official documentation](https://microsoft.github.io/AirSim/build_windows/)
2. Install Julia dependencies:
```julia
using Pkg
Pkg.add(["ZMQ", "PyCall", "RxInfer", "Images", "StaticArrays", "LinearAlgebra", "JSON"])
```
3. Configure AirSim settings.json for your simulation environment

## Running the System

1. Start AirSim simulator
2. Start the ZeroMQ server:
```
julia actinf/zmq_server.jl
```
3. Run the Python controller:
```
python test.py
```

This will:
- Initialize the drone in AirSim
- Start the ZeroMQ communication channel
- Process sensory observations
- Call Julia for inference and planning through ZeroMQ
- Execute actions with real-time obstacle avoidance

## Preference Model

The system uses a configurable preference model to evaluate states:
- Distance preference: Favors states closer to target
- Angle preference: Favors alignment with target (with constant baseline preference)
- Suitability preference: Favors safe regions away from obstacles

Weights between these preferences can be adjusted in the `PreferenceModel` configuration.
