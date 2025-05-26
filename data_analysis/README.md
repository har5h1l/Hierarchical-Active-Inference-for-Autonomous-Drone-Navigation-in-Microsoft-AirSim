# 🧠 Active Inference Experiment Analysis Pipeline

A comprehensive, modular analysis suite for analyzing experimental metrics from active inference-based drone navigation systems in Microsoft AirSim.

## 📊 Overview

This pipeline processes experimental data from your drone navigation experiments and generates:
- **Statistical Analysis**: Correlation analysis, hypothesis testing, behavioral clustering
- **Visualizations**: Time series plots, performance analysis, uncertainty patterns
- **Interactive Dashboard**: HTML-based viewer for all results
- **Detailed Reports**: Markdown reports with key findings and statistics

## 🚀 Quick Start

### Option 1: Complete Analysis (Recommended)
```bash
python run_analysis.py
```
This runs the entire pipeline and opens an interactive dashboard in your browser.

### Option 2: Step-by-Step Analysis
```bash
# 1. Process raw experiment data
python process_data.py

# 2. Run statistical analysis and generate plots
python analyze_envs.py

# 3. Generate interactive dashboard
python generate_dashboard.py
```

## 📁 File Structure

```
data_analysis/
├── analyze_envs.py           # Main analysis script
├── process_data.py           # Data preprocessing
├── generate_dashboard.py     # HTML dashboard generator
├── run_analysis.py          # Complete pipeline runner
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Processed experiment data
└── results/                # Analysis outputs
    ├── experiment_analysis_report.md  # Detailed report
    ├── dashboard.html                 # Interactive dashboard
    └── plots_[environment]/           # Environment-specific plots
        ├── timeseries_analysis.png
        ├── planning_behavior.png
        ├── correlation_matrix.png
        ├── lagged_correlations.png
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
- Evolution of VFE (Variational Free Energy) and EFE (Expected Free Energy)
- Entropy patterns over time
- Planning frequency dynamics
- Distance to target progression

### 🎯 Planning Behavior Analysis
- Planning frequency distributions
- Planning time analysis
- Relationship between planning and performance
- VFE patterns during planning vs. non-planning states

### 🔗 Correlation Analysis
- Correlation matrices between key variables
- Lagged correlation analysis for temporal relationships
- Statistical significance testing

### 📊 Performance Analysis
- Success rates by environment and experiment
- Path efficiency metrics
- Planning count vs. success analysis
- Average VFE comparison between successful and failed episodes

### 🌊 Uncertainty & Volatility Analysis
- Entropy vs. VFE relationships
- Volatility pattern analysis
- EFE behavior during planning
- Uncertainty evolution within episodes

### 🎲 Behavioral Clustering
- PCA-based dimensionality reduction
- K-means clustering of behavioral patterns
- Cluster characteristic analysis

### 📊 Statistical Testing
- Mann-Whitney U tests for group comparisons
- Pearson correlation analysis
- Success rate statistical analysis

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

### Analyzing Specific Environments
The pipeline automatically detects different environments from your experiment data and generates separate analyses for each.

### Custom Analysis
You can modify `analyze_envs.py` to add custom metrics or visualizations specific to your research needs.

### Batch Processing
The pipeline can process multiple experiments simultaneously and compare results across different configurations.

## 📄 Output Interpretation

### Dashboard
- **Interactive HTML interface** with all plots organized by environment
- **Easy navigation** between different analysis types
- **Embedded plots** that scale with your browser window

### Statistical Report
- **Comprehensive markdown report** with statistical test results
- **Environment-specific findings** and comparisons
- **Key correlations** and their significance levels

### Individual Plots
- **High-resolution PNG images** suitable for publications
- **Publication-ready formatting** with clear labels and legends
- **Consistent color schemes** across all visualizations

## 🔧 Customization

### Adding New Metrics
1. Modify the feature engineering section in `analyze_envs.py`
2. Add new correlation pairs to the analysis
3. Include new metrics in the statistical testing

### Environment-Specific Analysis
The pipeline automatically handles multiple environments. Configure environment names in your experiment config files.

### Custom Visualizations
Add new plotting functions to the `ExperimentAnalyzer` class and call them in the `analyze_by_environment` method.

## 🛠️ Troubleshooting

### Common Issues

**Missing Data**: Ensure your experiment results are in the `experiment_results/` folder with proper structure.

**Memory Issues**: For large datasets, the analysis script samples data points for visualization to maintain performance.

**Missing Dependencies**: Install all required packages using `pip install -r requirements.txt`.

### Data Format Requirements
- Experiments should be in folders named `experiment_YYYYMMDD_HHMMSS/`
- Each experiment should contain `metrics.csv` and `config.json`
- Optional: `episode_summaries.csv` for episode-level analysis

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

**Generated**: Data Analysis Pipeline for Active Inference Experiments  
**Version**: 1.0  
**Date**: May 2025
