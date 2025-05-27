# 🧠 Active Inference Single Environment Analysis Pipeline

A comprehensive analysis suite for analyzing experimental metrics from active inference-based drone navigation in a single Microsoft AirSim environment.

## 📊 Overview

This pipeline processes experimental data from your single environment drone navigation experiments and generates:
- **Performance Analysis**: Success rates, episode outcomes, comprehensive metrics  
- **VFE/EFE Dynamics**: Complete trajectory analysis showing ALL episodes over steps
- **Statistical Analysis**: Correlation analysis, hypothesis testing, behavioral patterns
- **Visualizations**: Performance dashboards, behavioral analysis, temporal dynamics
- **Detailed Reports**: Markdown reports with statistical findings and key insights

## 🚀 Quick Start

### Complete Analysis (Recommended)
```bash
python run_analysis.py
```
This runs the entire single environment analysis pipeline and generates all visualizations and reports.

### Manual Analysis
```bash
python analyze_single_environment.py
```
Run just the main analysis script directly.

## 📁 File Structure

```
data_analysis/
├── analyze_single_environment.py  # Main single environment analysis script
├── run_analysis.py               # Complete pipeline runner  
├── fix_unicode.py               # Windows compatibility utility
├── test_all_episodes.py         # Test script for validation
├── requirements.txt             # Python dependencies
├── README.md                   # This file (main documentation)
├── README_single_env.md        # Single environment specific docs
├── data/                       # Experiment data files
│   ├── episode_summaries.csv   # Episode outcomes and metrics
│   └── metrics.csv             # Step-by-step detailed metrics
└── results/                    # Analysis outputs
    ├── single_environment_analysis_report.md  # Detailed report
    ├── performance_dashboard.png              # Performance summary
    ├── enhanced_vfe_efe_dynamics.png          # VFE/EFE trajectory analysis
    ├── correlation_matrix.png                 # Behavioral correlations
    ├── planning_analysis.png                  # Planning behavior analysis
    ├── success_factors.png                    # Success vs failure analysis
    └── computational_cost_analysis.png        # Performance metrics
        ├── performance_analysis.png
        ├── uncertainty_analysis.png
        └── behavioral_clustering.png
```

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
