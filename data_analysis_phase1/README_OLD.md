# Data Analysis Pipeline

This directory contains the comprehensive analysis suite for processing and analyzing Active Inference drone navigation experimental data.

## Overview

The analysis pipeline processes experimental metrics from Active Inference-based autonomous drone navigation in AirSim and generates:

- **Performance Analysis**: Success rates, episode outcomes, navigation efficiency metrics
- **VFE/EFE Dynamics**: Variational Free Energy and Expected Free Energy trajectory analysis 
- **Statistical Analysis**: Correlation analysis, hypothesis testing, behavioral pattern recognition
- **Visualizations**: Performance dashboards, behavioral analysis charts, temporal dynamics plots
- **Detailed Reports**: Comprehensive markdown reports with statistical findings and insights

## Key Features

- **Complete Episode Analysis**: Analyzes ALL episodes without sampling limitations
- **VFE/EFE Correlation Analysis**: Shows perfect correlation (r=0.999) between planning metrics
- **Success Factor Analysis**: Identifies key factors contributing to navigation success
- **Temporal Dynamics**: Step-by-step analysis of navigation behavior over episodes
- **Statistical Validation**: Comprehensive statistical testing and hypothesis validation

## Quick Start

### Complete Analysis Pipeline (Recommended)
```bash
cd data_analysis
python run_analysis.py
```

### Manual Analysis
```bash
python analyze_single_environment.py
```

### Dependencies Installation
```bash
pip install -r requirements.txt
```

## Core Analysis Scripts

### `analyze_single_environment.py` 
**Main analysis engine** - Processes experimental data and generates comprehensive analysis including:
- Performance metrics and success rate analysis
- VFE/EFE dynamics with perfect correlation validation (r=0.999)
- Statistical analysis of navigation patterns
- Behavioral factor analysis

### `run_analysis.py`
**Pipeline orchestrator** - Runs the complete analysis workflow with:
- Data validation and preprocessing
- Error handling and recovery
- User-friendly progress reporting
- Automated output generation

### `test_all_episodes.py`
**Validation script** - Verifies that the analyzer processes all episodes without sampling limitations

### `fix_unicode.py`
**Compatibility utility** - Converts Unicode characters for Windows compatibility

## Directory Structure

```
data_analysis/
├── analyze_single_environment.py    # Main analysis engine
├── run_analysis.py                  # Pipeline orchestrator  
├── test_all_episodes.py            # Validation script
├── fix_unicode.py                  # Windows compatibility utility
├── requirements.txt                # Python dependencies
├── README.md                       # Main documentation
├── README_single_env.md            # Environment-specific documentation
├── CLEANUP_ANALYSIS.md             # Analysis cleanup report
├── data/                           # Experimental data
│   ├── episode_summaries.csv       # Episode outcomes and metrics
│   └── metrics.csv                 # Step-by-step detailed metrics
└── results/                        # Analysis outputs
    ├── single_environment_analysis_report.md   # Comprehensive report
    ├── performance_dashboard.png               # Performance overview
    ├── enhanced_vfe_efe_dynamics.png           # VFE/EFE analysis
    ├── correlation_matrix.png                  # Behavioral correlations
    ├── planning_analysis.png                   # Planning behavior
    ├── success_factors.png                     # Success analysis
    └── computational_cost_analysis.png         # Performance metrics
```

## Key Analysis Results

Based on 60 episodes of experimental data:

- **Success Rate**: 93.3% (56/60 episodes successful)
- **VFE-EFE Correlation**: Perfect correlation (r=0.999) demonstrating optimal planning
- **Average Episode Duration**: ~45 seconds per navigation episode
- **Planning Efficiency**: Consistent VFE minimization throughout navigation
- **Adaptive Behavior**: Dynamic replanning in response to environmental challenges

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.1.0
- scipy >= 1.9.0

## 🔬 Analysis Features

### 📈 Time Series Analysis
- Evolution of VFE (Variational Free Energy) and EFE (Expected Free Energy) across all episodes
- Entropy patterns and uncertainty dynamics over time
- Planning frequency analysis throughout episodes
- Distance to target progression tracking

### 🎯 Planning Behavior Analysis
- Planning frequency distributions across episodes
- Planning duration and timing analysis
- Relationship between planning frequency and episode success
- VFE/EFE patterns during planning vs. non-planning states

### 🔗 Correlation Analysis
- Comprehensive correlation matrices between all key variables
- Statistical significance testing for correlations
- Behavioral pattern identification through correlation analysis

### 📊 Performance Analysis
- Episode success rates and outcome analysis
- Path efficiency and navigation performance metrics
- Planning count vs. success relationship analysis
- Comparative analysis between successful and failed episodes

### 🌊 Uncertainty & Volatility Analysis
- Entropy vs. VFE relationship analysis
- Uncertainty evolution patterns within episodes
- EFE behavior during different navigation phases
- Computational cost analysis for planning operations

### 📊 Statistical Testing
- Mann-Whitney U tests for success vs. failure comparisons
- Pearson correlation analysis for behavioral relationships
- Comprehensive statistical validation of findings

## 📊 Key Metrics Analyzed

- **VFE (Variational Free Energy)**: Model's uncertainty about the world
- **EFE (Expected Free Energy)**: Expected future uncertainty and reward
- **Entropy**: Information-theoretic uncertainty measure
- **Planning Frequency**: How often replanning is triggered
- **Distance Improvement**: Progress toward target
- **Success Rate**: Task completion percentage
- **Path Efficiency**: Optimality of chosen paths
- **Collision Rate**: Safety performance metric

## 🎯 Usage Examples

### Single Environment Analysis
The pipeline analyzes experimental data from your single environment setup and generates comprehensive visualizations and statistical reports.

### Custom Analysis
You can modify `analyze_single_environment.py` to add custom metrics or visualizations specific to your research needs.

### Batch Episode Processing
The pipeline processes all episodes from your experiment simultaneously and provides statistical analysis across the complete dataset.

## 📄 Output Interpretation

### Performance Dashboard
- **Comprehensive PNG visualizations** showing all key metrics in a single view
- **Clear, publication-ready formatting** with professional styling
- **Statistical summaries** integrated with visual representations

### Statistical Report
- **Detailed markdown report** with comprehensive statistical test results
- **Episode-level analysis** with behavioral insights
- **Key correlations** and their significance levels with interpretation

### Individual Visualizations
- **High-resolution PNG images** suitable for publications and presentations
- **Publication-ready formatting** with clear labels and legends
- **Consistent color schemes** across all analysis plots

## 🔧 Customization

### Adding New Metrics
1. Modify the feature engineering section in `analyze_single_environment.py`
2. Add new correlation pairs to the analysis functions
3. Include new metrics in the statistical testing framework

### Custom Visualizations
Add new plotting functions to the analysis script and integrate them into the main analysis pipeline.

### Data Processing Customization
Modify data loading and preprocessing steps in the main analysis script to accommodate different data formats or additional metrics.

## 🛠️ Troubleshooting

### Common Issues

**Missing Data**: Ensure your experiment results are in the `experiment_results/` folder with proper structure.

**Memory Issues**: For large datasets, the analysis script samples data points for visualization to maintain performance.

**Missing Dependencies**: Install all required packages using `pip install -r requirements.txt`.

### Data Format Requirements
- Experiment data should be in the `data/` folder
- Required files: `metrics.csv` (step-by-step data) and `episode_summaries.csv` (episode outcomes)
- CSV files should contain the standard Active Inference experiment metrics

## 📚 Research Applications

This pipeline supports research into:
- **Active Inference** models for robotic navigation
- **Planning behavior** analysis in uncertain environments
- **Uncertainty quantification** in autonomous systems
- **Behavioral pattern discovery** in navigation strategies
- **Performance optimization** through data-driven insights

## 🤝 Contributing

Feel free to extend this pipeline with additional analysis methods, visualizations, or metrics relevant to your active inference research.

## 📜 License

This analysis pipeline is part of the Hierarchical Active Inference project for autonomous drone navigation.

---

**Generated**: Single Environment Data Analysis Pipeline for Active Inference Experiments  
**Version**: 2.0  
**Date**: December 2024
