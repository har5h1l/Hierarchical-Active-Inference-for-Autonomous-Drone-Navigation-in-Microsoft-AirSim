# Hierarchical Active Inference Data Analysis Pipeline

## Project Overview

This analysis pipeline supports research on **hierarchical active inference for autonomous drone navigation** that embeds affordance theory through environmental suitability states. The system demonstrates how cognitive principles from active inference can be integrated with affordance-based environmental reasoning to enable robust, adaptive navigation in complex 3D environments.

### Core Innovation
Our approach introduces a **latent environmental suitability state** inferred from multimodal sensory input (LiDAR and camera) that encodes local obstacle information and navigability. This forms an internal affordance map that parameterizes planning, filters infeasible waypoints, and guides Expected Free Energy (EFE)-based policy selection toward viable trajectories while reducing computational cost.

## Technical Implementation Analysis

**What This Analysis Pipeline Covers**:
- **EFE (Expected Free Energy)**: Real calculation with pragmatic/epistemic components for policy selection
- **Derived "VFE" Values**: Computed as negative EFE pragmatic component (`vfe = -efe_pragmatic`) for compatibility
- **Gaussian Belief Updates**: Kernel-based belief updating over navigation states and suitability
- **Affordance Integration**: Environmental suitability state effectiveness in waypoint filtering
- **Hierarchical Planning**: Two-tiered architecture combining belief-driven filtering with EFE minimization

**Research Context** (What Is NOT Traditional Active Inference):
- Uses Gaussian kernels for computational simplicity rather than full variational inference
- Suitability state provides affordance-based constraints on action space
- Hybrid approach combining symbolic planning with probabilistic inference

## 📊 Overview

The analysis system provides insights into the hierarchical active inference architecture:
- **EFE Dynamics**: Energy evolution patterns and affordance-aware planning behavior
- **Suitability State Analysis**: Environmental affordance encoding effectiveness
- **Navigation Performance**: Success rates, path efficiency, and adaptive behavioral patterns  
- **Gaussian Belief Evolution**: How beliefs update via kernel methods over navigation and suitability states
- **Environmental Adaptation**: Context-dependent weight adjustment and waypoint filtering patterns

## 🔬 Analysis Features

### 📈 Enhanced EFE vs Distance Visualization
- **Publication-ready** EFE trajectory analysis with statistical overlays
- Episode-by-episode tracking of Expected Free Energy relative to target distance
- Success/failure differentiation with trend analysis and confidence intervals
- Output formats: PNG (300 DPI) and PDF for submission

### 🎯 Hierarchical Planning Analysis
- EFE minimization patterns across episodes with affordance constraints
- Pragmatic vs epistemic value trade-offs in filtered action spaces
- Suitability state influence on waypoint selection and planning efficiency
- Adaptive weight mechanism effectiveness in complex environments

### 🔗 Affordance Integration Analysis
- Environmental suitability state correlation with navigation success
- Waypoint filtering effectiveness and computational cost reduction
- Statistical significance testing for affordance-based behavioral patterns
- Multimodal sensory integration (LiDAR + camera) performance

### 📊 Performance Analysis
- Episode success rates and outcome analysis (93.3% success rate across 149 episodes)
- EFE minimization effectiveness with affordance constraints
- Planning efficiency and computational cost (5.43ms average per EFE calculation)
- Collision avoidance and recovery behavior patterns

### 🌊 Belief State Analysis
- Gaussian belief evolution over navigation states and environmental suitability
- Entropy patterns from belief distributions during affordance-aware planning
- Uncertainty quantification through belief variance in complex environments
- Pragmatic/epistemic balance in different obstacle configurations

## 📊 Key Metrics Analyzed (Corrected)

- **EFE (Expected Free Energy)**: Real calculation with pragmatic/epistemic components
- **Derived "VFE"**: Negative pragmatic component (`-efe_pragmatic`) for analysis
- **Entropy**: Information-theoretic uncertainty from belief distributions
- **Planning Frequency**: EFE-based replanning trigger frequency
- **Distance Improvement**: Progress toward target via EFE minimization
- **Success Rate**: Task completion percentage
- **Path Efficiency**: Optimality of EFE-selected paths
- **Collision Rate**: Safety performance via suitability-based filtering
- **Belief Variance**: Uncertainty quantification in Gaussian distributions
- **Suitability State Effectiveness**: Affordance-based waypoint filtering performance
- **Environmental Adaptation**: Context-dependent planning behavior metrics

## 🚀 Quick Start

### Complete Analysis Pipeline
```bash
cd data_analysis_phase1
python analyze_single_environment.py
```

### Dependencies Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## 📁 Directory Structure

```
data_analysis_phase1/
├── analyze_single_environment.py    # Main analysis engine
├── requirements.txt                # Python dependencies  
├── README.md                       # This documentation
├── data/                           # Experimental data
│   ├── episode_summaries.csv       # Episode outcomes and metrics
│   └── metrics.csv                 # Step-by-step detailed metrics
└── results/                        # Analysis outputs
    ├── enhanced_efe_vs_distance_trajectories.png # Publication-ready EFE analysis
    ├── enhanced_efe_vs_distance_trajectories.pdf # Vector format
    ├── analysis_report.md           # Comprehensive analysis report
    ├── performance_dashboard.png    # Performance overview
    └── correlation_matrix.png       # Behavioral correlations
```

## 🔍 Understanding the Analysis Results

### Key Experimental Results (149 Episodes)
Based on experiments in Microsoft AirSim's obstacle-dense neighborhood environment:

### Performance Metrics
- **Success Rate**: 93.3% (139/149 episodes successful)
- **EFE Minimization**: Effective pragmatic/epistemic balance with affordance constraints
- **Planning Efficiency**: 5.43ms average per EFE calculation
- **Hierarchical Planning**: Successful integration of suitability-based filtering with EFE optimization

### Statistical Significance Findings
- **Efficiency Metric**: Significant difference (p=0.009572) between success/failure episodes
- **Replanning Count**: Highly significant difference (p<0.000001) - successful episodes: 1.56 ± 2.04, failed episodes: 10.60 ± 17.44
- **Distance Improvement**: Significant difference (p<0.000001) - successful: 97.7%, failed: 57.0%

### Implementation Notes
- **Belief Updating**: Uses Gaussian kernel-based approach for computational efficiency
- **Planning Effectiveness**: EFE minimization working correctly with affordance constraints
- **Stable Performance**: Consistent navigation behavior across diverse obstacle configurations

## 🎯 Research Applications and Future Directions

This analysis pipeline demonstrates:

1. **Hierarchical Active Inference**: Integration of affordance theory with EFE-based planning
2. **Environmental Suitability Encoding**: Latent state representation of navigational affordances
3. **Computational Efficiency**: Waypoint filtering reduces policy space while maintaining performance
4. **Adaptive Behavior**: Context-dependent planning in complex obstacle environments

### Research Contributions
- **Novel Integration**: Successful embedding of affordance theory into active inference through latent suitability states
- **Computational Efficiency**: Hierarchical approach reduces planning complexity while maintaining high performance  
- **Robustness**: 93.3% success rate in complex obstacle-dense environments
- **Theoretical Bridge**: Practical implementation connecting Gibson's affordance theory with modern active inference

### Future Research Directions
- Quantitative benchmarking against traditional planning baselines
- Real-world UAV deployment and validation
- Extension to dynamic environments with moving obstacles
- Development of cognitive maps as learned spatial representations
- Investigation of full VFE-based inference over Gaussian kernels
- Expanded navigation capabilities for object recognition, search, and tracking

## 📚 Technical Implementation Notes

### Hierarchical Architecture Details
The system implements a two-tiered approach:
1. **Upper Tier**: Suitability state inference and waypoint filtering using affordance constraints
2. **Lower Tier**: EFE minimization over filtered action space with pragmatic/epistemic balance

### Key Mathematical Relationships
```python
# Core implementation uses Gaussian kernel-based belief updating
# for computational efficiency in real-time navigation scenarios
```

### Experimental Validation
- **Environment**: Microsoft AirSim obstacle-dense neighborhood (houses, trees, tight alleys)
- **Episodes**: 149 total experimental runs
- **Target Sampling**: Random locations 20-100 meters away (seed 42)
- **Success Metrics**: Navigation completion, collision avoidance, planning efficiency

## ⚠️ Important Notes for Researchers

1. **Implementation Approach**: System uses Gaussian kernel-based belief updating for computational efficiency
2. **Performance Validity**: 93.3% navigation success demonstrates system effectiveness in complex environments
3. **Affordance Integration**: Latent suitability states successfully filter waypoints and guide planning

## 📞 Contact & Support

For questions about the corrected analysis interpretation:
- **Implementation Details**: Refer to `actinf/src/` code comments
- **Analysis Questions**: Check mathematical derivations in analysis scripts
- **Research Applications**: Consider EFE-based vs VFE-based approaches

---

**Project Status**: ✅ **Active Research Project**  
**Implementation**: Hierarchical Active Inference + Affordance Theory  
**Performance**: 93.3% Success Rate in Complex Environments  
**Analysis**: Publication-Ready Visualizations and Statistical Analysis

**Research Focus**: Autonomous Navigation · Active Inference · Affordance Theory · Computational Neuroscience

For technical details, refer to the main codebase in the parent directory and the generated analysis reports.
