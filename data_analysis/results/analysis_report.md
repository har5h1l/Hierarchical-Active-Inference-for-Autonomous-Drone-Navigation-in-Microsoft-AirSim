# Active Inference Experiment Analysis Report

**Generated on:** 2025-05-27 02:29:05
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