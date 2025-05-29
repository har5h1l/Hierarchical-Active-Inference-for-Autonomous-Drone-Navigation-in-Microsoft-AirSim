# Hierarchical Active Inference for Autonomous Drone Navigation

## 🎯 Project Overview

This project implements a **hierarchical active inference framework for autonomous drone navigation** that embeds affordance theory through environmental suitability states. The system demonstrates how cognitive principles from active inference can be integrated with affordance-based environmental reasoning to enable robust, adaptive navigation in complex 3D environments using Microsoft AirSim.

### 🚁 Core Innovation

Our approach introduces a **latent environmental suitability state** inferred from multimodal sensory input (LiDAR and camera) that encodes local obstacle information and navigability. This forms an internal affordance map that parameterizes planning, filters infeasible waypoints, and guides Expected Free Energy (EFE)-based policy selection toward viable trajectories while reducing computational cost.

### 🔬 Research Contributions

**Problem Addressed**: Traditional drone navigation lacks proper integration of environmental affordances and uncertainty quantification in planning systems.

**Solution**: A two-tiered hierarchical architecture that combines:
1. **Upper Tier**: Suitability state inference and waypoint filtering using affordance constraints
2. **Lower Tier**: EFE minimization over filtered action space with pragmatic/epistemic balance

**Key Innovation**: Integration of Gibson's affordance theory with modern active inference through latent environmental states, enabling context-aware and computationally efficient navigation.

## 🏗️ System Architecture

### Technical Implementation
- **Belief Updating**: Gaussian kernel-based approach for computational efficiency in real-time scenarios
- **Multimodal Sensing**: LiDAR and depth camera data processing into latent voxel grids
- **Hierarchical Planning**: Two-stage process combining affordance filtering with EFE optimization
- **Adaptive Behavior**: Context-dependent weight adjustment and dynamic replanning

### Performance Results
- **Success Rate**: 93.3% (139/149 episodes) in complex obstacle environments
- **Planning Efficiency**: 5.43ms average per EFE calculation
- **Environment**: Microsoft AirSim obstacle-dense neighborhood with houses, trees, and tight alleys
- **Robustness**: Effective collision recovery and adaptive replanning mechanisms

## 🚀 Quick Start

### Prerequisites
- Julia 1.8+ with RxInfer.jl
- Python 3.8+ with AirSim API
- Microsoft AirSim (Neighborhood environment)

### Running the System
```bash
# 1. Start AirSim simulator
# 2. Run the hierarchical active inference navigation
julia run_inference.jl

# 3. Analyze experimental results
cd data_analysis_phase1
python analyze_single_environment.py
```

## 📁 Project Structure

```
├── README.md                          # This documentation
├── run_inference.jl                   # Main execution script
├── run_planning.jl                    # Planning component
├── actinf/                            # Active inference implementation
│   ├── src/                          # Core Julia modules
│   └── zmq_server.jl                 # Communication interface
├── airsim/                           # AirSim integration
│   ├── Sensory_Input_Processing.py   # Sensor data processing
│   └── settings.json                 # AirSim configuration
├── data_analysis_phase1/             # Analysis pipeline
│   ├── analyze_single_environment.py # Main analysis engine
│   ├── README.md                     # Analysis documentation
│   ├── data/                         # Experimental data
│   └── results/                      # Generated visualizations
├── experiment_results/               # Raw experimental data
└── interface/                        # Communication protocols
```

## ⚙️ Technical Implementation Details

### 🧮 Mathematical Framework & Constants

#### Core System Constants
```julia
# Suitability Calculation Parameters
DEFAULT_OBSTACLE_WEIGHT = 0.8        # Primary obstacle avoidance factor
DEFAULT_DENSITY_WEIGHT = 0.4         # Environmental density influence
CUTOFF_DISTANCE = 3.0                # Sigmoid transition point (meters)
STEEPNESS_DISTANCE = 4.0             # Sigmoid steepness for distance
CUTOFF_DENSITY = 0.15                # Density threshold for suitability
STEEPNESS_DENSITY = 12.0             # Sharp response to density changes

# Planning Thresholds
SUITABILITY_THRESHOLD = 0.75          # Minimum waypoint acceptance (75%)
CLOSE_TO_TARGET_THRESHOLD = 10.0     # Target approach distance (meters)
VERY_CLOSE_TO_TARGET_THRESHOLD = 5.0 # Aggressive target approach (meters)
MIN_ACCEPTABLE_SUITABILITY = 0.5     # Hard safety cutoff

# Adaptive Planning Parameters
MIN_RADIUS = 1.0                     # Minimum step size (meters)
MAX_RADIUS = 3.5                     # Maximum exploration radius (meters)
MIN_POLICY_LEN = 2                   # Minimum planning horizon
MAX_POLICY_LEN = 5                   # Maximum planning depth
MIN_WAYPOINTS = 25                   # Minimum candidate waypoints
MAX_WAYPOINTS = 100                  # Maximum waypoint generation
```

#### 🎯 Environmental Suitability Calculation

**Primary Equation**: Sigmoid-based environmental assessment
```julia
suitability = obstacle_weight * safety_factor + density_weight * density_factor

# Safety Factor (distance-based)
safety_factor = 1.0 / (1.0 + exp(-steepness_distance * (obstacle_distance - cutoff_distance)))

# Density Factor (inverted for high density penalty)
density_factor = 1.0 / (1.0 + exp(steepness_density * (obstacle_density - cutoff_density)))
```

**Hard Safety Overrides**:
- Distance < 1.2m: `safety_factor *= (distance/1.2)²`
- Density > 0.4: `density_factor *= max(0.2, 1.0 - 2*(density-0.4))`

#### 🎛️ Preference Model Configuration

**Default Weights & Parameters**:
```julia
PreferenceModel(
    distance_weight = 2.0,            # Strong distance minimization preference
    distance_scaling = 0.05,          # Gradual distance decay rate
    angle_weight = 0.5,               # Moderate directional preference
    angle_sharpness = 3.0,            # Angular precision factor
    suitability_weight = 1.0,         # Environmental safety weight
    suitability_threshold = 0.55,     # Base acceptability threshold
    max_distance = 125.0              # Normalization boundary (meters)
)
```

**Dynamic Weight Adaptation**:
- Low suitability (< 0.4): Increase safety weight by up to 2.5x, reduce distance weight by 50%
- High density environments: Boost suitability importance exponentially
- Target proximity: Reduce obstacle weight from 0.8 to 0.7-1.0 based on distance

#### 🧠 Expected Free Energy (EFE) Calculation

**Core EFE Equation**:
```julia
total_efe = pragmatic_value + epistemic_value

# Pragmatic Component (goal-seeking)
pragmatic_value = -pragmatic_weight * (preference_score + distance_bonus) * action_magnitude

# Epistemic Component (uncertainty reduction)
epistemic_value = -epistemic_weight * 0.5 * total_entropy * action_magnitude
```

**Entropy Calculation**: Multi-dimensional belief uncertainty
```julia
total_entropy = distance_entropy + azimuth_entropy + elevation_entropy + 
               suitability_entropy + density_entropy

# Per dimension: entropy = -Σ(belief[i] * log(belief[i] + 1e-10))
```

**ActionPlanner Default Weights**:
```julia
ActionPlanner(
    max_step_size = 0.5,              # Maximum action magnitude
    pragmatic_weight = 1.0,           # Goal-seeking emphasis
    epistemic_weight = 0.2,           # Exploration factor
    risk_weight = 2.0,                # Safety priority (early filtering)
    safety_distance = 1.5,            # Minimum obstacle clearance
    density_weight = 1.0              # Environmental density sensitivity
)
```

#### 📡 Gaussian Kernel Belief Updates

**Adaptive Kernel Widths**:
```julia
distance_kernel = base_kernel * (1.0 + 0.5 * distance/100.0)  # Distance-dependent
azimuth_kernel = base_kernel * 2.0                           # Wider for angles
elevation_kernel = base_kernel * 2.0                         # Angular uncertainty
```

**Bayesian Update Formula**:
```julia
# Gaussian likelihood for observation
likelihood[i] = exp(-0.5 * (distance_to_observation / kernel_width)²)

# Temporal smoothing update
belief = 0.8 * prior_belief + 0.2 * normalized_likelihood
```

**Target Preference Weighting**:
```julia
target_weight = exp(-distance_to_target / 10.0)  # Exponential decay
distance_belief *= (1.0 + target_weight * (max_distance - distance_range) / max_distance)
```

### 🎮 Adaptive Planning System

#### Dynamic Parameter Adjustment
- **Suitability-Based Radius**: `radius = MIN_RADIUS + suitability * (MAX_RADIUS - MIN_RADIUS)`
- **Policy Length**: `length = MAX_LEN - suitability * (MAX_LEN - MIN_LEN)`
- **Waypoint Count**: Inversely proportional to suitability, scaled 1.0-3.0x in dense environments

#### Hierarchical Filtering Process
1. **Generate** 25-100 candidate waypoints based on environment complexity
2. **Filter** by suitability threshold (75% default, adaptive near target)
3. **Calculate** EFE for each viable candidate
4. **Weight** final selection: 70% suitability, 30% EFE for high-quality paths
5. **Select** top K actions based on adaptive policy length

## 🔄 How the Navigation Process Works

### Step-by-Step Process Overview

The hierarchical active inference navigation system operates through a continuous cycle that integrates environmental perception, belief updating, and action selection. Here's how the drone navigates from start to target:

#### 🎯 Phase 1: Mission Initialization
1. **Target Assignment**: Drone receives target coordinates (20-100m away in experimental setup)
2. **Initial Belief Setup**: Initialize Gaussian beliefs over:
   - Distance to target (0-150m range, 15 bins)
   - Azimuth angle (-π to π, 32 bins) 
   - Elevation angle (-π/2 to π/2, 16 bins)
   - Environmental suitability (0-1, 10 bins)
   - Obstacle density (0-1, 10 bins)
3. **System Parameters**: Load adaptive planning weights and safety thresholds

#### 🔍 Phase 2: Environmental Sensing (Every Navigation Step)
1. **LiDAR Data Collection**: 360° point cloud capture (up to 3000 points)
2. **Depth Camera Processing**: RGB-D imagery for visual obstacle detection
3. **Voxel Grid Generation**: Convert sensor data into 3D occupancy grid (0.5m resolution)
4. **Obstacle Distance Calculation**: Find nearest obstacle in each direction
5. **Density Analysis**: Calculate local obstacle density in 5m radius

#### 🧠 Phase 3: Belief State Updates
1. **Sensor Fusion**: Combine LiDAR and camera data into unified environmental model
2. **Gaussian Kernel Updates**: Update beliefs using adaptive kernel widths:
   ```
   Distance kernel: base_width * (1.0 + 0.5 * distance/100.0)
   Angular kernels: base_width * 2.0 (wider uncertainty)
   ```
3. **Temporal Smoothing**: Blend new observations with prior beliefs (80% prior + 20% new)
4. **Target-Weighted Updates**: Enhance beliefs toward target direction using exponential decay

#### 🏗️ Phase 4: Hierarchical Planning (Upper Tier)
1. **Waypoint Generation**: Create 25-100 candidate waypoints in spherical coordinates
   - Radius: Adaptive based on environment complexity (1.0-3.5m)
   - Angular sampling: Uniform distribution with target bias
2. **Environmental Suitability Assessment**: For each waypoint, calculate:
   ```
   suitability = 0.8 * safety_factor + 0.4 * density_factor
   safety_factor = sigmoid(obstacle_distance - 3.0m threshold)
   density_factor = inverse_sigmoid(density - 0.15 threshold)
   ```
3. **Affordance-Based Filtering**: Remove waypoints below suitability threshold (75% default)
4. **Safety Overrides**: Apply hard constraints for proximity (< 1.2m) and high density (> 0.4)

#### ⚡ Phase 5: Action Selection (Lower Tier)
1. **EFE Calculation**: For each viable waypoint, compute Expected Free Energy:
   ```
   EFE = pragmatic_component + epistemic_component
   pragmatic = -1.0 * (preference_score + distance_bonus) * action_magnitude
   epistemic = -0.2 * entropy_reduction * action_magnitude
   ```
2. **Preference Scoring**: Weight factors (distance: 2.0, angle: 0.5, suitability: 1.0)
3. **Multi-Objective Ranking**: Combine suitability (70%) and EFE minimization (30%)
4. **Policy Selection**: Choose top K actions based on adaptive policy length (2-5 steps)

#### 🚁 Phase 6: Action Execution
1. **Waypoint Command**: Send selected waypoint to AirSim flight controller
2. **Motion Execution**: Drone moves toward waypoint with collision monitoring
3. **Progress Tracking**: Monitor distance to target and execution success
4. **Replanning Triggers**: Initiate new planning cycle if:
   - Waypoint reached (normal progression)
   - Obstacle detected during flight (emergency replanning)
   - Significant belief uncertainty increase

#### 🔄 Phase 7: Adaptive Feedback Loop
1. **Performance Assessment**: Evaluate movement success and environmental changes
2. **Parameter Adaptation**: Adjust planning parameters based on context:
   - Low suitability: Increase safety weight (up to 2.5x), reduce distance weight (50%)
   - Near target: Reduce obstacle weight, increase target preference
   - High uncertainty: Increase epistemic exploration weight
3. **Replanning Decision**: Continue cycle until target reached or mission timeout

### 🎛️ Adaptive Mechanisms in Action

**Dynamic Weight Adjustment Example**:
```
Standard weights: obstacle=0.8, distance=2.0, suitability=1.0
Dense environment: obstacle=1.0, distance=1.0, suitability=2.5
Near target: obstacle=0.7, distance=3.0, suitability=0.8
```

**Emergency Behaviors**:
- **Collision Risk**: Immediately increase safety weight, reduce planning horizon
- **Target Unreachable**: Expand search radius, increase epistemic exploration
- **High Uncertainty**: Trigger additional sensing, slower movement

### 📊 Computational Performance
- **Sensor Processing**: ~10-15ms per cycle (voxel grid generation)
- **Belief Updates**: ~2-3ms (Gaussian kernel operations)
- **EFE Calculation**: ~5.43ms average (per waypoint evaluation)
- **Total Cycle Time**: ~20-30ms end-to-end
- **Replanning Frequency**: 1.56 times per successful episode (adaptive triggers)

This hierarchical approach enables the drone to navigate complex environments by continuously balancing goal-seeking behavior (pragmatic EFE) with uncertainty reduction (epistemic EFE) while respecting environmental affordances through the suitability state framework.

## 🔬 Research Applications and Future Directions

This project demonstrates:

1. **Hierarchical Active Inference**: Integration of affordance theory with EFE-based planning
2. **Environmental Suitability Encoding**: Latent state representation of navigational affordances
3. **Computational Efficiency**: Waypoint filtering reduces policy space while maintaining performance
4. **Adaptive Behavior**: Context-dependent planning in complex obstacle environments

### Future Research Directions
- Quantitative benchmarking against traditional planning baselines
- Real-world UAV deployment and validation
- Extension to dynamic environments with moving obstacles
- Development of cognitive maps as learned spatial representations
- Investigation of full VFE-based inference over Gaussian kernels
- Expanded navigation capabilities for object recognition, search, and tracking

## 📊 Analysis and Results

Comprehensive analysis of the system's performance is available in the `data_analysis_phase1/` directory, including:

- **Publication-ready visualizations**: EFE vs distance trajectories with statistical analysis
- **Performance metrics**: 93.3% success rate across 149 episodes
- **Statistical significance testing**: Efficiency, replanning behavior, and distance improvement analysis
- **Implementation insights**: Gaussian kernel effectiveness and affordance integration analysis

For detailed analysis results and methodology, see: [`data_analysis_phase1/README.md`](data_analysis_phase1/README.md)

## 👥 Authors & Contact

**Harshil Shah**  
GitHub: [@har5h1l](https://github.com/har5h1l)  
Email: 28hshah@gmail.com

**Satyaki Maitra**  
GitHub: [@gubgubi](https://github.com/gubgubi)  
Email: satyakimaitra2010@gmail.com

### 🏛️ Institutional Affiliation
This research was conducted through the **Active Inference Institute Internship Program**, supporting the development of active inference applications in autonomous systems and cognitive robotics.

---
