# Active Inference Single Environment Analysis Pipeline

A comprehensive analysis suite for analyzing experimental metrics from a single active inference-based drone navigation environment in Microsoft AirSim.

## Overview

This pipeline processes experimental data from your single environment drone navigation experiments and generates:
- **Performance Analysis**: Success rates, episode outcomes, key metrics summary
- **Behavioral Patterns**: Correlation analysis, VFE/EFE dynamics, planning behavior
- **Visual Dashboard**: Comprehensive 6-panel performance dashboard
- **Step-by-step Analysis**: Detailed temporal dynamics using metrics.csv data  
- **Success Factor Analysis**: Comparative analysis of successful vs failed episodes
- **Detailed Reports**: Markdown reports with statistical findings

## Quick Start

### Option 1: Complete Analysis (Recommended)
```bash
python run_analysis.py
```
This runs the entire pipeline, processes data, and generates all visualizations and reports.

### Option 2: Single Environment Analysis Only
```bash
python analyze_single_environment.py
```
Run just the single environment analysis without preprocessing.

## Data Requirements

The analysis expects the following data files in the `data/` folder:

1. **episode_summaries.csv**: Episode-level data with columns like:
   - `episode_id`, `status`, `steps_taken`, `final_distance`
   - `distance_improvement_percentage`, `avg_vfe`, `avg_efe`
   - `collisions`, `replanning_count`, `duration_seconds`

2. **metrics.csv**: Step-level detailed metrics with columns like:
   - `episode_id`, `step`, `vfe`, `efe`, `distance_to_target`
   - Planning and navigation metrics for temporal analysis

## File Structure

```
data_analysis/
├── analyze_single_environment.py  # Main single environment analyzer
├── analyze_envs.py                # Legacy multi-environment analyzer  
├── process_data.py                # Data preprocessing from experiment_results/
├── generate_dashboard.py          # HTML dashboard generator
├── run_analysis.py               # Complete pipeline runner
├── fix_unicode.py                # Windows Unicode compatibility fix
├── requirements.txt              # Python dependencies
├── README_single_env.md          # This file
├── data/                         # Input data files
│   ├── episode_summaries.csv     # Episode-level data
│   └── metrics.csv               # Step-level detailed metrics
└── results/                      # Analysis outputs
    ├── analysis_report.md         # Comprehensive analysis report
    ├── performance_dashboard.png   # 6-panel performance dashboard
    ├── correlation_matrix.png     # Behavioral correlation heatmap
    ├── enhanced_vfe_efe_dynamics.png  # 12-panel VFE/EFE comprehensive analysis
    ├── planning_analysis.png      # Planning behavior distributions
    ├── success_factors.png        # Success vs failure comparison
    └── dashboard.html             # Interactive dashboard (if generated)
```

## Key Analysis Features

### Performance Dashboard (6-panel visualization)
1. **Episode Outcomes**: Pie chart of success/failure rates
2. **Steps Distribution**: Histogram by episode status
3. **Distance Improvement**: Distribution of improvement percentages
4. **VFE vs EFE**: Scatter plot showing relationship for success/failure
5. **Collision Analysis**: Bar chart of collision frequencies
6. **Learning Curve**: Rolling average performance over episodes

### Behavioral Analysis
- **Correlation Matrix**: Heatmap of relationships between all numeric metrics
- **VFE/EFE Dynamics**: Temporal analysis of variational and expected free energy
- **Planning Behavior**: Distribution analysis of replanning patterns
- **Success Factors**: Comparative analysis of successful vs failed episodes

### Generated Reports
- **analysis_report.md**: Comprehensive markdown report with:
  - Episode outcome statistics
  - Key performance metrics (mean, std, median)
  - Behavioral correlations and patterns
  - Success vs failure comparative analysis
  - Planning behavior statistics

## Requirements

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

## Windows Compatibility

The `fix_unicode.py` script automatically converts Unicode emoji characters to ASCII equivalents for Windows terminal compatibility. This runs automatically as part of the pipeline.

## Key Metrics Analyzed

- **VFE (Variational Free Energy)**: Model's uncertainty about the world
- **EFE (Expected Free Energy)**: Expected future uncertainty and reward
- **Distance Improvement**: Progress toward target percentage
- **Success Rate**: Task completion percentage
- **Planning Frequency**: How often replanning is triggered
- **Collision Rate**: Safety performance metric
- **Episode Duration**: Time-based performance analysis

## Usage Examples

### Basic Analysis
```bash
# Run complete pipeline
python run_analysis.py

# View results
# Check results/ folder for generated files
# Open performance_dashboard.png for visual overview
# Read analysis_report.md for detailed findings
```

### Custom Analysis Periods
The analyzer automatically processes all available episodes. To analyze specific periods, filter your input CSV files before running the analysis.

## Output Interpretation

### Performance Dashboard
- **Success Rate**: Overall experiment effectiveness
- **VFE/EFE Patterns**: Learning and uncertainty dynamics
- **Collision Analysis**: Safety performance trends
- **Learning Curve**: Improvement over time

### Correlation Analysis
- **Strong Positive Correlations**: Metrics that increase together
- **Strong Negative Correlations**: Inverse relationships
- **Planning Dependencies**: How planning affects other metrics

### Success Factor Analysis
- **Box Plots**: Compare distributions between successful and failed episodes
- **Statistical Differences**: Quantified performance gaps
- **Key Discriminators**: Metrics most predictive of success

## Research Applications

This pipeline supports research into:
- **Single Environment Learning**: How agents improve in consistent conditions
- **Active Inference Dynamics**: VFE/EFE behavior patterns
- **Planning Strategy Analysis**: When and why replanning occurs
- **Safety Analysis**: Collision patterns and avoidance behaviors
- **Performance Optimization**: Identifying key success factors

## Troubleshooting

### Common Issues
- **Unicode Errors**: Run `python fix_unicode.py` to fix Windows compatibility
- **Missing Data**: Ensure `episode_summaries.csv` and `metrics.csv` are in `data/` folder
- **Empty Results**: Check that your CSV files have the expected column names
- **Memory Issues**: Analysis now displays ALL trajectory data; consider system resources for very large datasets

### Data Format Requirements
- CSV files should have headers matching expected column names
- `episode_id` should be consistent across both files for linking
- Numeric columns should be properly formatted (no mixed types)

---

**Single Environment Analysis Pipeline**  
**Version**: 2.0  
**Date**: May 2025  
**Optimized for**: Windows compatibility and single environment experiments
