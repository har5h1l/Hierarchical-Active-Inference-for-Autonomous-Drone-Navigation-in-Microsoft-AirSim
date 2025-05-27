# Active Inference Experiment Analysis Report

**Generated on:** 2025-05-26 21:45:47
**Total Episodes:** 149

## Episode Outcomes
- **Success:** 139 episodes (93.3%)
- **Stuck:** 9 episodes (6.0%)
- **Timeout:** 1 episodes (0.7%)

## Key Performance Metrics
- **Steps Taken:** Mean=16.28, Std=8.84, Median=16.00
- **Final Distance:** Mean=2.47, Std=7.07, Median=1.16
- **Distance Improvement Percentage:** Mean=94.93, Std=12.85, Median=97.91
- **Avg Vfe:** Mean=198.55, Std=50.52, Median=194.71
- **Avg Efe:** Mean=-206.32, Std=50.96, Median=-202.40
- **Collisions:** Mean=1.23, Std=2.86, Median=1.00
- **Replanning Count:** Mean=2.17, Std=5.25, Median=1.00
- **Duration Seconds:** Mean=46.22, Std=27.83, Median=42.37

## Key Behavioral Correlations
- **replanning_count** vs **dynamic_replanning_count**: 1.000 (positive)
- **replanning_percentage** vs **dynamic_replanning_percentage**: 1.000 (positive)
- **normalized_improvement** vs **distance_improvement_percentage**: 1.000 (positive)
- **avg_vfe** vs **avg_efe**: -1.000 (negative)
- **final_distance** vs **distance_improvement_percentage**: -0.988 (negative)
- **final_distance** vs **normalized_improvement**: -0.988 (negative)
- **episode_id** vs **timestamp_completed**: 0.974 (positive)
- **steps_taken** vs **path_length**: 0.971 (positive)
- **collisions** vs **replanning_count**: 0.963 (positive)
- **collisions** vs **dynamic_replanning_count**: 0.963 (positive)

## Enhanced VFE/EFE Dynamics Analysis

### VFE Statistics:
- **Raw VFE**: Mean=183.23, Std=114.38, Min=1.20, Max=759.19
- **Log-Normalized VFE**: Mean=4.90, Std=1.12

### EFE Statistics:
- **Raw EFE**: Mean=-190.89, Std=116.38, Min=-782.12, Max=-7.25
- **Log-Normalized EFE**: Mean=5.03, Std=0.81

### Energy Minimization Trends (Normalized):
- **VFE Trend**: -0.0008 per episode (Improving)
- **EFE Trend**: 0.0008 per episode (Improving)

## Planning Behavior Statistics
- **Replanning Count:** Mean=2.17, Median=1.00, Max=56.00
- **Dynamic Replanning Count:** Mean=2.17, Median=1.00, Max=56.00
- **Avg Planning Time Ms:** Mean=5.43, Median=5.34, Max=8.49

## Success vs Failure Analysis
- **Steps Taken:** Success=16.06, Failure=19.20 (Diff: -16.3%)
- **Avg Vfe:** Success=199.41, Failure=186.54 (Diff: +6.9%)
- **Avg Efe:** Success=-207.19, Failure=-194.23 (Diff: +6.7%)
- **Collisions:** Success=0.94, Failure=5.40 (Diff: -82.7%)
- **Replanning Count:** Success=1.56, Failure=10.60 (Diff: -85.3%)
- **Avg Planning Time Ms:** Success=5.43, Failure=5.43 (Diff: -0.0%)

## Computational Cost Analysis
*Detailed computational efficiency metrics and cost-benefit analysis.*
- **Average Planning Time**: 5.43 ± 0.53 ms
- **Average Episode Duration**: 46.22 ± 27.83 seconds
- **Average Step Time**: 3.04 seconds per step
- **Cost-Benefit Ratio**: 0.0144 (efficiency per planning ms)

## Statistical Significance Testing (ANOVA)
*Comprehensive statistical testing between successful and failed episodes.*

## VFE-EFE Correlation Investigation
*Deep analysis of the relationship between Variational Free Energy and Expected Free Energy.*
- **Primary Correlation**: r = -0.999987, p = 0.000000
- **Relationship Strength**: Perfect negative correlation
- **Theoretical Implications**:
  - Strong correlation suggests tight coupling between perception and action
  - Consistent with Active Inference principle of energy minimization
  - May indicate shared computational basis or theoretical constraint
- **VFE Range**: [14.2, 326.1]
- **EFE Range**: [-336.2, -20.5]

## Generated Visualizations
*This analysis produced the following visualization files:*
- **performance_dashboard.png**: 6-panel performance overview
- **enhanced_vfe_efe_dynamics.png**: 12-panel VFE/EFE analysis with log normalization
- **correlation_matrix.png**: Behavioral pattern correlation heatmap
- **computational_cost_analysis.png**: 12-panel computational efficiency analysis
- **anova_statistical_testing.png**: Statistical significance testing results
- **vfe_efe_correlation_investigation.png**: 12-panel correlation deep dive
- **planning_analysis.png**: Planning behavior distributions
- **success_factors.png**: Success vs failure factor analysis

## Methodology Summary
*Statistical methods and analysis approaches used:*
- **Descriptive Statistics**: Mean, standard deviation, median for all metrics
- **Correlation Analysis**: Pearson correlation for linear relationships
- **ANOVA Testing**: F-tests for group differences between success/failure
- **Effect Size Analysis**: Cohen's d for practical significance
- **Normality Testing**: D'Agostino normality tests for distribution assumptions
- **Variance Testing**: Levene's test for equal variances
- **Logarithmic Normalization**: log(|value| + 1e-6) for VFE/EFE analysis
- **Bootstrap Analysis**: Correlation stability testing
- **Temporal Analysis**: Time series trends and rolling averages

## Data Quality Summary
- **Episode Records**: 149 episodes analyzed
- **Step Records**: 2425 step-level measurements
- **Numeric Features**: 31 quantitative metrics
- **Missing Data**: No missing values detected

## Key Findings Summary
*Primary insights from the comprehensive analysis:*
- **Overall Success Rate**: 93.3%
- **Average Distance Improvement**: 94.9%
- **Planning Efficiency**: Success episodes -0.0% planning time vs failures
- **Energy Coupling**: Strong correlation indicates tight VFE-EFE relationship

---
**Analysis completed**: 2025-05-26 21:45:58
**Framework**: Single Environment Active Inference Analysis v2.0