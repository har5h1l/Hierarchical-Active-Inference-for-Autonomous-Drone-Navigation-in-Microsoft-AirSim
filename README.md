# Autonomous Drone Navigation in AirSim With Active Inference

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

The drone's state is represented by the `DroneState` structure with five key dimensions:

- **Distance**: Distance to target (meters)
- **Azimuth**: Horizontal angle to target (radians)
- **Elevation**: Vertical angle to target (radians)
- **Suitability**: Environmental safety measure (0-1)
- **Obstacle Density**: Density of obstacles in vicinity (0-1)

States are computed from `DroneObservation` objects that contain raw sensory data including drone position, orientation, target position, and obstacle information.

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
   - Alignment with target (azimuth/elevation)
   - Environmental safety (obstacle avoidance)

2. **Epistemic Value**: Information-seeking behavior to reduce uncertainty
   - Entropy reduction in beliefs about state variables
   - Weighted lower than pragmatic value for goal-directed behavior

3. **Risk Value**: Explicit penalty for unsafe states
   - Based on proximity to obstacles and obstacle density
   - Higher weight for stronger risk aversion

### Planning Process

1. Generate potential waypoints around current position
2. Additional waypoints directly toward target are added with various step sizes
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
- Execute actions in the simulation

## Preference Model

The system uses a configurable preference model to evaluate states:
- Distance preference: Favors states closer to target
- Angle preference: Favors alignment with target
- Suitability preference: Favors safe regions away from obstacles

Weights between these preferences can be adjusted in the `PreferenceModel` configuration.
