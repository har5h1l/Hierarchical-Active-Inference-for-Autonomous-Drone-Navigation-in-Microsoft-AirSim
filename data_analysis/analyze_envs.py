"""
3-ENVIRONMENT COMPARISON ANALYSIS FOR ACTIVE INFERENCE EXPERIMENTS

🧠 Purpose:
Analyze and compare 3 CSV files from 3 different experiments with episode data in 3 different environments.
Focus on comparative analysis between environments and experiment performance.

📂 Expected Input:
- 3 CSV files containing episode data from different environments
- Each CSV should have columns: episode_id, success, environment, VFE, EFE, entropy, planning metrics, etc.

🛠️ Key Features:
- Environment-specific performance analysis
- Cross-environment statistical comparisons
- Behavioral pattern identification across environments
- Success rate and failure mode analysis
- Interactive visualizations for environment comparison
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu, kruskal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import glob
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class ThreeEnvironmentAnalyzer:
    """Specialized analyzer for comparing 3 experiments across 3 different environments"""
    
    def __init__(self):
        self.env_data = {}  # Dictionary to store data for each environment
        self.combined_data = None
        self.report_lines = []
        self.env_names = []
        
    def load_three_csv_files(self, csv_file_paths, environment_names=None):
        """
        Load 3 CSV files from different experiments/environments
        
        Args:
            csv_file_paths: List of 3 CSV file paths
            environment_names: Optional list of 3 environment names (default: env1, env2, env3)
        """
        if len(csv_file_paths) != 3:
            raise ValueError("Exactly 3 CSV files must be provided")
            
        if environment_names is None:
            environment_names = ['Environment_1', 'Environment_2', 'Environment_3']
        elif len(environment_names) != 3:
            raise ValueError("Exactly 3 environment names must be provided")
            
        self.env_names = environment_names
        print(f"🔍 Loading data from 3 environments: {', '.join(environment_names)}")
        
        all_data = []
        
        for i, (csv_path, env_name) in enumerate(zip(csv_file_paths, environment_names)):
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
                
            print(f"📊 Loading {env_name} data from: {os.path.basename(csv_path)}")
            
            try:
                df = pd.read_csv(csv_path)
                df['environment'] = env_name
                df['env_id'] = i + 1
                
                # Store environment-specific data
                self.env_data[env_name] = df.copy()
                all_data.append(df)
                
                print(f"   ✅ Loaded {len(df)} episodes from {env_name}")
                
            except Exception as e:
                print(f"   ❌ Error loading {csv_path}: {e}")
                raise
        
        # Combine all data
        self.combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📈 Combined dataset: {len(self.combined_data)} total episodes across 3 environments")
        
        return self.combined_data
    
    def validate_data_structure(self):
        """Validate that the loaded data has expected columns and structure"""
        print("🔧 Validating data structure...")
        
        required_columns = ['episode_id']  # Minimum required
        recommended_columns = [
            'episode_id', 'success', 'vfe', 'efe', 'entropy', 
            'planning_events', 'replanning_count', 'episode_length',
            'final_distance_to_goal', 'planning_time_ms'
        ]
        
        if self.combined_data is None:
            raise ValueError("No data loaded. Call load_three_csv_files() first.")
        
        # Check for required columns
        missing_required = [col for col in required_columns if col not in self.combined_data.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        # Check for recommended columns
        missing_recommended = [col for col in recommended_columns if col not in self.combined_data.columns]
        if missing_recommended:
            print(f"⚠️  Missing recommended columns: {missing_recommended}")
            print("   Analysis may be limited without these columns.")
        
        # Display data summary
        print(f"✅ Data validation complete")
        print(f"   Columns available: {list(self.combined_data.columns)}")
        print(f"   Episodes per environment:")
        for env_name in self.env_names:
            count = len(self.env_data[env_name])
            print(f"     {env_name}: {count} episodes")
    
    def engineer_comparative_features(self):
        """Create features specifically for environment comparison"""
        print("🔧 Engineering comparative features...")
        
        df = self.combined_data.copy()
        
        # Basic success metrics
        if 'success' in df.columns:
            df['success_binary'] = df['success'].astype(int)
        
        # Normalize metrics by environment for comparison
        metrics_to_normalize = ['vfe', 'efe', 'entropy', 'episode_length']
        available_metrics = [col for col in metrics_to_normalize if col in df.columns]
        
        for metric in available_metrics:
            # Z-score normalization within each environment
            df[f'{metric}_normalized'] = df.groupby('environment')[metric].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            
            # Percentile rank within environment
            df[f'{metric}_percentile'] = df.groupby('environment')[metric].transform(
                lambda x: x.rank(pct=True)
            )
        
        # Planning efficiency metrics
        if 'planning_events' in df.columns and 'episode_length' in df.columns:
            df['planning_frequency'] = df['planning_events'] / (df['episode_length'] + 1)
        
        if 'planning_time_ms' in df.columns and 'planning_events' in df.columns:
            df['avg_planning_time'] = df['planning_time_ms'] / (df['planning_events'] + 1)
        
        # Volatility measures (if time-series data available)
        if 'step' in df.columns:
            window_size = 5
            for metric in ['vfe', 'efe', 'entropy']:
                if metric in df.columns:
                    df[f'{metric}_volatility'] = df.groupby(['environment', 'episode_id'])[metric].transform(
                        lambda x: x.rolling(window_size, min_periods=1).std()
                    )
        
        self.combined_data = df
        print(f"✅ Feature engineering complete. Dataset now has {len(df.columns)} columns")
    
    def analyze_environment_performance(self):
        """Analyze and compare performance across the 3 environments"""
        print("📊 Analyzing environment-specific performance...")
        
        results = {}
        
        for env_name in self.env_names:
            env_data = self.env_data[env_name]
            env_results = {
                'environment': env_name,
                'total_episodes': len(env_data),
            }
            
            # Success rate
            if 'success' in env_data.columns:
                success_rate = env_data['success'].mean()
                env_results['success_rate'] = success_rate
                env_results['successful_episodes'] = env_data['success'].sum()
                env_results['failed_episodes'] = len(env_data) - env_data['success'].sum()
            
            # Performance metrics
            numeric_columns = env_data.select_dtypes(include=[np.number]).columns
            for col in ['vfe', 'efe', 'entropy', 'episode_length', 'planning_events']:
                if col in numeric_columns:
                    env_results[f'{col}_mean'] = env_data[col].mean()
                    env_results[f'{col}_std'] = env_data[col].std()
                    env_results[f'{col}_median'] = env_data[col].median()
            
            results[env_name] = env_results
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results).T
        
        # Save results
        comparison_df.to_csv(os.path.join(RESULTS_DIR, 'environment_comparison.csv'))
        
        # Add to report
        self.report_lines.append("## Environment Performance Comparison")
        self.report_lines.append("\n")
        self.report_lines.append("| Environment | Episodes | Success Rate | Avg VFE | Avg EFE | Avg Episode Length |")
        self.report_lines.append("|-------------|----------|--------------|---------|---------|-------------------|")
        
        for env_name in self.env_names:
            r = results[env_name]
            success_rate = f"{r.get('success_rate', 0):.1%}"
            avg_vfe = f"{r.get('vfe_mean', 0):.2f}"
            avg_efe = f"{r.get('efe_mean', 0):.2f}"
            avg_length = f"{r.get('episode_length_mean', 0):.1f}"
            
            self.report_lines.append(
                f"| {env_name} | {r['total_episodes']} | {success_rate} | {avg_vfe} | {avg_efe} | {avg_length} |"
            )
        
        return results
    
    def statistical_environment_comparison(self):
        """Perform statistical tests comparing the 3 environments"""
        print("📈 Performing statistical comparisons between environments...")
        
        stats_results = {}
        
        # Metrics to compare
        metrics_to_test = ['vfe', 'efe', 'entropy', 'episode_length', 'planning_events']
        available_metrics = [col for col in metrics_to_test if col in self.combined_data.columns]
        
        for metric in available_metrics:
            print(f"   Testing {metric}...")
            
            # Get data for each environment
            env_data = []
            for env_name in self.env_names:
                data = self.env_data[env_name][metric].dropna()
                env_data.append(data)
            
            # Kruskal-Wallis test (non-parametric ANOVA alternative)
            if len(env_data) == 3 and all(len(data) > 0 for data in env_data):
                stat, p_value = kruskal(*env_data)
                
                stats_results[metric] = {
                    'test': 'Kruskal-Wallis',
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                # Pairwise comparisons (Mann-Whitney U)
                pairwise_results = {}
                for i, env1 in enumerate(self.env_names):
                    for j, env2 in enumerate(self.env_names[i+1:], i+1):
                        stat_mw, p_mw = mannwhitneyu(env_data[i], env_data[j], alternative='two-sided')
                        pairwise_results[f'{env1}_vs_{env2}'] = {
                            'statistic': stat_mw,
                            'p_value': p_mw,
                            'significant': p_mw < 0.05
                        }
                
                stats_results[metric]['pairwise'] = pairwise_results
        
        # Add to report
        self.report_lines.append("\n## Statistical Comparisons")
        self.report_lines.append("\n")
        
        for metric, result in stats_results.items():
            self.report_lines.append(f"### {metric.upper()}")
            self.report_lines.append(f"**Kruskal-Wallis Test:** H = {result['statistic']:.3f}, p = {result['p_value']:.3f}")
            
            if result['significant']:
                self.report_lines.append("*Significant difference between environments (p < 0.05)*")
                
                # Show pairwise comparisons
                self.report_lines.append("\n**Pairwise Comparisons:**")
                for pair, pair_result in result['pairwise'].items():
                    significance = "**" if pair_result['significant'] else ""
                    self.report_lines.append(
                        f"- {pair}: p = {pair_result['p_value']:.3f} {significance}"
                    )
            else:
                self.report_lines.append("*No significant difference between environments*")
          
        return stats_results
    
    def create_environment_comparison_plots(self):
        """Create comprehensive comparison plots across all 3 environments"""
        print("📊 Creating environment comparison visualizations...")
        
        # 1. Performance comparison dashboard
        self.plot_performance_comparison_dashboard()
        
        # 2. Statistical distributions comparison
        self.plot_distributions_comparison()
        
        # 3. Success factors analysis
        self.plot_success_factors_analysis()
        
        # 4. Behavioral patterns comparison
        self.plot_behavioral_patterns_comparison()
        
        # 5. Temporal dynamics comparison
        self.plot_temporal_dynamics_comparison()
    
    def plot_performance_comparison_dashboard(self):
        """Create a comprehensive performance comparison dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('3-Environment Performance Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # Success rates
        success_rates = []
        episode_counts = []
        for env_name in self.env_names:
            env_data = self.env_data[env_name]
            if 'success' in env_data.columns:
                success_rate = env_data['success'].mean()
                success_rates.append(success_rate)
            else:
                success_rates.append(0)
            episode_counts.append(len(env_data))
        
        bars = axes[0,0].bar(self.env_names, success_rates, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0,0].set_title('Success Rate by Environment')
        axes[0,0].set_ylabel('Success Rate')
        axes[0,0].set_ylim(0, 1)
        for i, (bar, rate) in enumerate(zip(bars, success_rates)):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{rate:.1%}', ha='center', va='bottom')
        
        # Episode counts
        bars = axes[0,1].bar(self.env_names, episode_counts, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0,1].set_title('Number of Episodes')
        axes[0,1].set_ylabel('Episode Count')
        for i, (bar, count) in enumerate(zip(bars, episode_counts)):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                          str(count), ha='center', va='bottom')
        
        # VFE comparison (boxplot)
        vfe_data = []
        for env_name in self.env_names:
            if 'vfe' in self.env_data[env_name].columns:
                vfe_data.append(self.env_data[env_name]['vfe'].dropna())
            else:
                vfe_data.append([])
        
        if any(len(data) > 0 for data in vfe_data):
            axes[0,2].boxplot(vfe_data, labels=self.env_names)
            axes[0,2].set_title('VFE Distribution by Environment')
            axes[0,2].set_ylabel('VFE')
        
        # Planning frequency comparison
        planning_freq = []
        for env_name in self.env_names:
            env_data = self.env_data[env_name]
            if 'planning_events' in env_data.columns and 'episode_length' in env_data.columns:
                freq = (env_data['planning_events'] / env_data['episode_length']).mean()
                planning_freq.append(freq)
            elif 'planning_frequency' in env_data.columns:
                planning_freq.append(env_data['planning_frequency'].mean())
            else:
                planning_freq.append(0)
        
        bars = axes[1,0].bar(self.env_names, planning_freq, color=['skyblue', 'lightgreen', 'salmon'])
        axes[1,0].set_title('Average Planning Frequency')
        axes[1,0].set_ylabel('Planning Frequency')
        for i, (bar, freq) in enumerate(zip(bars, planning_freq)):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                          f'{freq:.3f}', ha='center', va='bottom')
        
        # Episode length comparison
        length_data = []
        for env_name in self.env_names:
            if 'episode_length' in self.env_data[env_name].columns:
                length_data.append(self.env_data[env_name]['episode_length'].dropna())
            else:
                length_data.append([])
        
        if any(len(data) > 0 for data in length_data):
            axes[1,1].boxplot(length_data, labels=self.env_names)
            axes[1,1].set_title('Episode Length Distribution')
            axes[1,1].set_ylabel('Episode Length')
        
        # Entropy comparison
        entropy_data = []
        for env_name in self.env_names:
            if 'entropy' in self.env_data[env_name].columns:
                entropy_data.append(self.env_data[env_name]['entropy'].dropna())
            else:
                entropy_data.append([])
        
        if any(len(data) > 0 for data in entropy_data):
            axes[1,2].boxplot(entropy_data, labels=self.env_names)
            axes[1,2].set_title('Entropy Distribution by Environment')
            axes[1,2].set_ylabel('Entropy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'environment_comparison_dashboard.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_distributions_comparison(self):
        """Compare key metric distributions across environments"""
        metrics = ['vfe', 'efe', 'entropy', 'episode_length']
        available_metrics = [m for m in metrics if any(m in self.env_data[env].columns for env in self.env_names)]
        
        if not available_metrics:
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        fig.suptitle('Metric Distributions Across 3 Environments', fontsize=16, fontweight='bold')
        
        colors = ['skyblue', 'lightgreen', 'salmon']
        
        for i, metric in enumerate(available_metrics[:4]):  # Limit to 4 plots
            ax = axes[i]
            
            for j, env_name in enumerate(self.env_names):
                if metric in self.env_data[env_name].columns:
                    data = self.env_data[env_name][metric].dropna()
                    if len(data) > 0:
                        ax.hist(data, bins=20, alpha=0.6, label=env_name, 
                               color=colors[j], density=True)
            
            ax.set_title(f'{metric.upper()} Distribution')
            ax.set_xlabel(metric)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_metrics), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'metric_distributions_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_success_factors_analysis(self):
        """Analyze factors contributing to success in each environment"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Success Factors Analysis by Environment', fontsize=16, fontweight='bold')
        
        for i, env_name in enumerate(self.env_names):
            env_data = self.env_data[env_name]
            ax = axes[i]
            
            if 'success' in env_data.columns:
                successful = env_data[env_data['success'] == True]
                failed = env_data[env_data['success'] == False]
                
                # Compare a key metric (VFE if available, else episode_length)
                metric = 'vfe' if 'vfe' in env_data.columns else 'episode_length'
                if metric in env_data.columns:
                    if len(successful) > 0:
                        success_data = successful[metric].dropna()
                        if len(success_data) > 0:
                            ax.hist(success_data, bins=15, alpha=0.7, 
                                   label=f'Success (n={len(success_data)})', color='green', density=True)
                    
                    if len(failed) > 0:
                        failed_data = failed[metric].dropna()
                        if len(failed_data) > 0:
                            ax.hist(failed_data, bins=15, alpha=0.7, 
                                   label=f'Failed (n={len(failed_data)})', color='red', density=True)
                    
                    ax.set_title(f'{env_name}\n{metric.upper()} by Outcome')
                    ax.set_xlabel(metric)
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No {metric} data\navailable', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{env_name}')
            else:
                ax.text(0.5, 0.5, 'No success data\navailable', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{env_name}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'success_factors_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_behavioral_patterns_comparison(self):
        """Compare behavioral patterns across environments using radar chart"""
        # Calculate behavioral metrics for each environment
        behavioral_metrics = {}
        
        for env_name in self.env_names:
            env_data = self.env_data[env_name]
            metrics = {}
            
            # Calculate normalized metrics (0-1 scale)
            if 'vfe' in env_data.columns:
                metrics['Low VFE'] = 1 - (env_data['vfe'].mean() / env_data['vfe'].max() if env_data['vfe'].max() > 0 else 0)
            else:
                metrics['Low VFE'] = 0.5
                
            if 'planning_events' in env_data.columns and 'episode_length' in env_data.columns:
                planning_freq = (env_data['planning_events'] / env_data['episode_length']).mean()
                metrics['Planning Frequency'] = min(planning_freq * 5, 1)  # Scale to 0-1
            else:
                metrics['Planning Frequency'] = 0.5
                
            if 'success' in env_data.columns:
                metrics['Success Rate'] = env_data['success'].mean()
            else:
                metrics['Success Rate'] = 0.5
                
            if 'entropy' in env_data.columns:
                metrics['Low Uncertainty'] = 1 - (env_data['entropy'].mean() / env_data['entropy'].max() if env_data['entropy'].max() > 0 else 0)
            else:
                metrics['Low Uncertainty'] = 0.5
                
            if 'episode_length' in env_data.columns:
                max_length = max(self.env_data[e]['episode_length'].max() for e in self.env_names if 'episode_length' in self.env_data[e].columns)
                metrics['Efficiency'] = 1 - (env_data['episode_length'].mean() / max_length if max_length > 0 else 0)
            else:
                metrics['Efficiency'] = 0.5
            
            behavioral_metrics[env_name] = metrics
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Get all metric names
        all_metrics = list(behavioral_metrics[self.env_names[0]].keys())
        
        # Number of variables
        N = len(all_metrics)
        
        # Angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        colors = ['blue', 'green', 'red']
        
        for i, env_name in enumerate(self.env_names):
            values = [behavioral_metrics[env_name][metric] for metric in all_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=env_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Behavioral Patterns Comparison\n(Higher values = Better performance)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'behavioral_patterns_radar.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_dynamics_comparison(self):
        """Compare temporal dynamics across environments if step data is available"""
        # Check if any environment has step/time data
        has_temporal_data = any('step' in self.env_data[env].columns for env in self.env_names)
        
        if not has_temporal_data:
            print("   No temporal data available for dynamics comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Dynamics Comparison', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'green', 'red']
        
        # Plot VFE evolution over time
        for i, env_name in enumerate(self.env_names):
            env_data = self.env_data[env_name]
            if 'step' in env_data.columns and 'vfe' in env_data.columns:
                # Sample data for readability
                sample_data = env_data.sample(min(500, len(env_data)))
                axes[0,0].scatter(sample_data['step'], sample_data['vfe'], 
                                alpha=0.6, s=1, label=env_name, color=colors[i])
        
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('VFE')
        axes[0,0].set_title('VFE Evolution Over Time')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot entropy evolution
        for i, env_name in enumerate(self.env_names):
            env_data = self.env_data[env_name]
            if 'step' in env_data.columns and 'entropy' in env_data.columns:
                sample_data = env_data.sample(min(500, len(env_data)))
                axes[0,1].scatter(sample_data['step'], sample_data['entropy'], 
                                alpha=0.6, s=1, label=env_name, color=colors[i])
        
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Entropy')
        axes[0,1].set_title('Entropy Evolution Over Time')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot planning events over time
        for i, env_name in enumerate(self.env_names):
            env_data = self.env_data[env_name]
            if 'step' in env_data.columns:
                planning_col = None
                if 'planning_triggered' in env_data.columns:
                    planning_col = 'planning_triggered'
                elif 'planning_events' in env_data.columns:
                    planning_col = 'planning_events'
                
                if planning_col:
                    # Calculate rolling average of planning
                    rolling_planning = env_data.groupby('episode_id').apply(
                        lambda x: x[planning_col].rolling(window=10, min_periods=1).mean()
                    ).reset_index(level=0, drop=True)
                    
                    sample_indices = np.random.choice(len(env_data), min(500, len(env_data)), replace=False)
                    axes[1,0].scatter(env_data.iloc[sample_indices]['step'], 
                                    rolling_planning.iloc[sample_indices],
                                    alpha=0.6, s=1, label=env_name, color=colors[i])
        
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Planning Activity (Rolling Avg)')
        axes[1,0].set_title('Planning Activity Over Time')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot episode progress (if distance data available)
        for i, env_name in enumerate(self.env_names):
            env_data = self.env_data[env_name]
            distance_col = None
            if 'distance_to_target' in env_data.columns:
                distance_col = 'distance_to_target'
            elif 'final_distance_to_goal' in env_data.columns:
                distance_col = 'final_distance_to_goal'
            
            if 'step' in env_data.columns and distance_col:
                sample_data = env_data.sample(min(500, len(env_data)))
                axes[1,1].scatter(sample_data['step'], sample_data[distance_col], 
                                alpha=0.6, s=1, label=env_name, color=colors[i])
        
        axes[1,1].set_xlabel('Step')
        axes[1,1].set_ylabel('Distance to Target')
        axes[1,1].set_title('Progress Toward Target Over Time')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'temporal_dynamics_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive markdown report for the 3-environment analysis"""
        print("📝 Generating comprehensive 3-environment analysis report...")
        
        report_path = os.path.join(RESULTS_DIR, "three_environment_analysis_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Three-Environment Active Inference Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes and compares performance across 3 different environments ")
            f.write("for active inference-based drone navigation experiments.\n\n")
            
            # Environment overview
            f.write("## Environment Overview\n\n")
            f.write("| Environment | Episodes | Data Columns Available |\n")
            f.write("|-------------|----------|------------------------|\n")
            
            for env_name in self.env_names:
                env_data = self.env_data[env_name]
                episode_count = len(env_data)
                columns = list(env_data.columns)
                f.write(f"| {env_name} | {episode_count} | {len(columns)} columns |\n")
            
            f.write("\n")
            
            # Add all the analysis results
            for line in self.report_lines:
                f.write(line + "\n")
            
            # Add methodology section
            f.write("\n## Methodology\n\n")
            f.write("### Data Processing\n")
            f.write("1. **Data Loading**: Loaded 3 CSV files, one for each environment\n")
            f.write("2. **Feature Engineering**: Created comparative and normalized features\n")
            f.write("3. **Statistical Testing**: Applied Kruskal-Wallis and Mann-Whitney U tests\n")
            f.write("4. **Visualization**: Generated comprehensive comparison plots\n\n")
            
            f.write("### Statistical Methods\n")
            f.write("- **Kruskal-Wallis Test**: Non-parametric test for comparing 3+ groups\n")
            f.write("- **Mann-Whitney U Test**: Non-parametric test for pairwise comparisons\n")
            f.write("- **Pearson Correlation**: Linear relationship analysis\n")
            f.write("- **Normalization**: Z-score normalization within environments for fair comparison\n\n")
            
            # Files generated
            f.write("## Generated Visualizations\n\n")
            f.write("This analysis generated the following visualization files:\n\n")
            f.write("1. **`environment_comparison_dashboard.png`** - Overall performance dashboard\n")
            f.write("2. **`metric_distributions_comparison.png`** - Distribution comparisons\n")
            f.write("3. **`success_factors_analysis.png`** - Success factor analysis by environment\n")
            f.write("4. **`behavioral_patterns_radar.png`** - Behavioral patterns radar chart\n")
            f.write("5. **`temporal_dynamics_comparison.png`** - Temporal dynamics (if available)\n")
            f.write("6. **`environment_comparison.csv`** - Raw comparison statistics\n\n")
            
            # Conclusions
            f.write("## Key Insights\n\n")
            f.write("*This section summarizes the main findings from the cross-environment analysis:*\n\n")
            
            # Calculate some summary insights
            if self.combined_data is not None and 'success' in self.combined_data.columns:
                overall_success = self.combined_data['success'].mean()
                f.write(f"- **Overall Success Rate**: {overall_success:.1%} across all environments\n")
                
                env_success_rates = {}
                for env_name in self.env_names:
                    env_success = self.env_data[env_name]['success'].mean() if 'success' in self.env_data[env_name].columns else 0
                    env_success_rates[env_name] = env_success
                    f.write(f"- **{env_name} Success Rate**: {env_success:.1%}\n")
                
                # Find best and worst performing environment
                if env_success_rates:
                    best_env = max(env_success_rates, key=env_success_rates.get)
                    worst_env = min(env_success_rates, key=env_success_rates.get)
                    f.write(f"- **Best Performing Environment**: {best_env} ({env_success_rates[best_env]:.1%} success)\n")
                    f.write(f"- **Most Challenging Environment**: {worst_env} ({env_success_rates[worst_env]:.1%} success)\n")
            
            f.write("\n*For detailed statistical comparisons and significance tests, see the Statistical Comparisons section above.*\n")
        
        print(f"📄 Comprehensive report saved to: {report_path}")
    
    def run_three_environment_analysis(self, csv_file_paths, environment_names=None):
        """
        Run complete analysis for 3 environments
        
        Args:
            csv_file_paths: List of 3 CSV file paths
            environment_names: Optional list of 3 environment names
        """
        print("🚀 Starting 3-Environment Active Inference Analysis...\n")
        
        try:
            # Load and validate data
            self.load_three_csv_files(csv_file_paths, environment_names)
            self.validate_data_structure()
            
            # Process and analyze
            self.engineer_comparative_features()
            
            # Run analyses
            print("📊 Running performance analysis...")
            self.analyze_environment_performance()
            
            print("📈 Running statistical comparisons...")
            self.statistical_environment_comparison()
            
            print("🎨 Creating visualizations...")
            self.create_environment_comparison_plots()
            
            print("📝 Generating report...")
            self.generate_comprehensive_report()
            
            print(f"\n✅ 3-Environment Analysis Complete!")
            print(f"📊 Results saved in: {RESULTS_DIR}")
            print(f"📈 Analyzed {len(self.combined_data)} total episodes across 3 environments")
            print(f"📝 Comprehensive report: {os.path.join(RESULTS_DIR, 'three_environment_analysis_report.md')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False


def main():
    """
    Main function to run the 3-environment analysis
    
    Usage examples:
    1. Prepare 3 CSV files with episode data from different environments
    2. Update the csv_files list below with your file paths
    3. Optionally customize environment names
    4. Run this script
    """
    
    # CONFIGURE YOUR ANALYSIS HERE
    # ===========================
    
    # Example CSV file paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    csv_files = [
        os.path.join(DATA_DIR, "environment_1_episodes.csv"),  # Replace with actual path
        os.path.join(DATA_DIR, "environment_2_episodes.csv"),  # Replace with actual path  
        os.path.join(DATA_DIR, "environment_3_episodes.csv"),  # Replace with actual path
    ]
    
    # Optional: Custom environment names (default: Environment_1, Environment_2, Environment_3)
    env_names = [
        "Forest_Environment",     # Replace with your environment names
        "Urban_Environment", 
        "Mountain_Environment"
    ]
    
    # ===========================
    
    print("🧠 Three-Environment Active Inference Analysis Tool")
    print("=" * 60)
    print(f"Expected CSV files:")
    for i, path in enumerate(csv_files, 1):
        print(f"  {i}. {path}")
    print(f"Environment names: {env_names}")
    print("=" * 60)
    
    # Check if files exist
    missing_files = [f for f in csv_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing CSV files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 To use this tool:")
        print("1. Place your 3 CSV files in the data_analysis/data/ folder")
        print("2. Update the csv_files list in this script with correct paths")
        print("3. Ensure each CSV has episode data with columns like:")
        print("   episode_id, success, vfe, efe, entropy, planning_events, etc.")
        return
    
    # Run analysis
    analyzer = ThreeEnvironmentAnalyzer()
    success = analyzer.run_three_environment_analysis(csv_files, env_names)
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Check the results folder: {RESULTS_DIR}")
    else:
        print("\n💥 Analysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()