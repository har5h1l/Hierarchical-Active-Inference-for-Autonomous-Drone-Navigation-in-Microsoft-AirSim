# Data Analysis Pipeline

This directory contains the data analysis pipeline for the hierarchical active inference drone navigation system. For complete project information, system architecture, and technical implementation details, see the main project README.

## Analysis Features

### Enhanced EFE vs Distance Visualization
- **Publication-ready** EFE trajectory analysis with statistical overlays
- Episode-by-episode tracking of Expected Free Energy relative to target distance  
- Success/failure differentiation with trend analysis and confidence intervals
- Output formats: PNG (300 DPI) and PDF for research publication

### Performance Metrics Analysis
- Episode success rates and outcome analysis
- EFE minimization effectiveness patterns
- Planning efficiency and computational cost metrics
- Collision avoidance and recovery behavior analysis

### Statistical Analysis
- Success vs failure episode comparisons
- Trend fitting and correlation analysis
- Confidence intervals and significance testing
- Behavioral pattern identification

## Data Pipeline

### Input Data
- **Episode Summaries** (`data/episode_summaries.csv`): Episode-level outcomes and performance metrics
- **Detailed Metrics** (`data/metrics.csv`): Step-by-step calculation results during navigation

### Key Metrics Analyzed
- **EFE (Expected Free Energy)**: Real calculation with pragmatic/epistemic components
- **Derived "VFE"**: Negative pragmatic component (`-efe_pragmatic`) for compatibility analysis
- **Distance to Target**: Navigation progress tracking
- **Success/Failure Outcomes**: Episode completion status
- **Planning Frequency**: EFE-based replanning patterns
- **Performance Statistics**: Efficiency, collision rates, path optimality

## Running the Analysis

### Complete Analysis Pipeline
```bash
cd data_analysis_phase1
python analyze_single_environment.py
```

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Output Files
- `enhanced_efe_vs_distance_trajectories.png` - Publication-ready EFE analysis (300 DPI)
- `enhanced_efe_vs_distance_trajectories.pdf` - Vector format for publications
- `analysis_report.md` - Detailed statistical analysis report
- `performance_dashboard.png` - Performance overview visualization
- `correlation_matrix.png` - Behavioral correlation analysis

## Directory Structure

```
data_analysis_phase1/
├── analyze_single_environment.py    # Main analysis engine with enhanced EFE visualization
├── requirements.txt                # Python dependencies  
├── README.md                       # Analysis pipeline documentation
├── data/                           # Experimental data directory
│   ├── episode_summaries.csv       # Episode-level outcomes and metrics
│   └── metrics.csv                 # Step-by-step detailed measurements
└── results/                        # Analysis outputs
    ├── enhanced_efe_vs_distance_trajectories.png # Publication-ready EFE analysis
    ├── enhanced_efe_vs_distance_trajectories.pdf # Vector format output
    ├── analysis_report.md           # Statistical analysis report
    ├── performance_dashboard.png    # Performance overview
    ├── correlation_matrix.png       # Behavioral correlation matrix
    ├── path_comparisons/            # Path visualization comparisons
    └── drone_path.png              # Sample navigation path
```

## Analysis Results Summary

**Based on 149 experimental episodes:**

### Performance Statistics
- **Success Rate**: 93.3% (139/149 episodes)
- **Average Planning Time**: 5.43ms per EFE calculation
- **Successful Episode Replanning**: 1.56 ± 2.04 times
- **Failed Episode Replanning**: 10.60 ± 17.44 times
- **Distance Improvement (Success)**: 97.7%
- **Distance Improvement (Failure)**: 57.0%

### Statistical Significance
- **Efficiency Difference**: p=0.009572 (success vs failure)
- **Replanning Frequency**: p<0.000001 (highly significant)
- **Distance Improvement**: p<0.000001 (highly significant)

## Implementation Notes

- **Data Source**: Microsoft AirSim obstacle-dense neighborhood environment
- **Episode Design**: Random targets 20-100m away, consistent seed (42)
- **Metric Calculation**: EFE values computed with pragmatic/epistemic components
- **Compatibility Mode**: "VFE" derived as negative pragmatic component for analysis
- **Visualization**: Enhanced publication-ready graphs with statistical overlays

---

For complete project information, system architecture, and technical implementation details, refer to the main project README in the parent directory.
