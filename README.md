# Hierarchical Active Inference for Autonomous Drone Navigation in Microsoft AirSim with Environmentally Aware Adaptive Planning

This project implements autonomous drone navigation in the AirSim simulator using Active Inference, a computational framework based on the Free Energy Principle. The system interfaces AirSim with Julia for probabilistic inference using RxInfer.jl.

## Project Structure

- `actinf/` - Julia package implementing the active inference framework
  - `src/actinf.jl` - Main module definition
  - `src/StateSpace.jl` - State space representation for drone navigation
  - `src/Inference.jl` - Belief updating and probabilistic inference
  - `src/Planning.jl` - Action selection and planning using Expected Free Energy
- `airsim/` - Python interface to AirSim simulator
  - `Sensory_Input_Processing.py` - Processing sensor data from AirSim
  - `settings.json` - AirSim configuration
- `interface/` - JSON files for inter-process communication
- `run_inference.jl` - Entry point for belief state inference
- `run_planning.jl` - Entry point for action planning
- `test.py` - Python controller for simulation execution

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
- Calculated using two main components:
  - **Obstacle Distance**: Exponential decay function `exp(-1/d)` gives higher values for greater distances
  - **Obstacle Density**: Exponential function `exp(-5*density)` penalizes crowded areas
- Obstacle distance is weighted more heavily (70%) than density (30%)

### Belief Updating

The `Inference` module maintains probabilistic beliefs about each state dimension:

- Beliefs are represented as discretized probability distributions
- Updates incorporate new observations using Bayesian inference
- Circular quantities (angles) are handled with appropriate distance metrics
- `DroneBeliefs` structure maintains both the distributions and their discretization ranges
- Temporal smoothing is applied to ensure stable belief evolution

### Action Selection

Action selection is based on minimizing Expected Free Energy (EFE), which balances:

1. **Pragmatic Value**: Preference for states that satisfy the agent's goals
   - Distance reduction (targeting)
   - Alignment with target (azimuth/elevation) with a baseline preference (0.2) toward facing target
   - Environmental safety (obstacle avoidance)

2. **Epistemic Value**: Information-seeking behavior to reduce uncertainty
   - Entropy reduction in beliefs about state variables
   - Weighted lower than pragmatic value for goal-directed behavior

3. **Risk Value**: Explicit penalty for unsafe states
   - Based on proximity to obstacles and obstacle density
   - Higher weight for stronger risk aversion

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

### Planning Process

1. Generate potential waypoints around current position with adaptive radius
2. Add direct-to-target waypoints with various step sizes
3. Calculate suitability for each waypoint based on obstacle proximity and density
4. Select top waypoints by suitability
5. For each candidate, simulate the resulting state transition
6. Calculate Expected Free Energy for each action
7. Select action with minimum EFE
8. Return policy (sequence of best actions)

## Requirements

- Julia 1.7+
- Python 3.7+
- AirSim simulator
- RxInfer.jl, PyCall.jl, and other dependencies

## Setup

1. Install AirSim following the [official documentation](https://microsoft.github.io/AirSim/build_windows/)
2. Install Julia dependencies:
```julia
using Pkg
Pkg.add(["PyCall", "RxInfer", "Images", "StaticArrays", "LinearAlgebra", "JSON"])
```
3. Configure AirSim settings.json for your simulation environment

## Running the System

1. Start AirSim simulator
2. Run the Python controller:
```
python test.py
```

This will:
- Initialize the drone in AirSim
- Process sensory observations
- Call Julia for inference and planning
- Execute actions with real-time obstacle avoidance

## Preference Model

The system uses a configurable preference model to evaluate states:
- Distance preference: Favors states closer to target
- Angle preference: Favors alignment with target (with constant baseline preference)
- Suitability preference: Favors safe regions away from obstacles

Weights between these preferences can be adjusted in the `PreferenceModel` configuration.
