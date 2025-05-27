"""
SINGLE ENVIRONMENT ANALYSIS FOR ACTIVE INFERENCE EXPERIMENTS

[BRAIN] Purpose:
Analyze episode data from a single environment experiment with comprehensive metrics analysis.

[FOLDER] Expected Input:
- episode_summaries.csv: Main episode data with performance metrics
- metrics.csv: Step-by-step detailed metrics during episodes

[TOOLS] Key Features:
- Performance analysis and success rate calculations
- VFE/EFE analysis and behavioral patterns
- Planning and replanning analysis
- Collision and obstacle avoidance metrics
- Temporal dynamics and learning curves
- Step-by-step behavioral analysis
- Comprehensive visualizations and reporting
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EPISODE_FILE = "episode_summaries.csv"
METRICS_FILE = "metrics.csv"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

class SingleEnvironmentAnalyzer:
    """Analyzer for single environment Active Inference experiment data"""
    
    def __init__(self):
        self.episode_data = None
        self.metrics_data = None
        self.report_lines = []
        self.numeric_columns = []
        
    def load_data(self):
        """Load episode summaries and metrics data"""
        episode_path = os.path.join(DATA_DIR, EPISODE_FILE)
        metrics_path = os.path.join(DATA_DIR, METRICS_FILE)
        
        if not os.path.exists(episode_path):
            raise FileNotFoundError(f"Episode data not found at {episode_path}")
            
        print(f"Loading episode data from {episode_path}")
        self.episode_data = pd.read_csv(episode_path)
        print(f"OK Loaded {len(self.episode_data)} episodes")
        
        # Load metrics if available
        if os.path.exists(metrics_path):
            print(f"Loading detailed metrics from {metrics_path}")
            self.metrics_data = pd.read_csv(metrics_path)
            print(f"OK Loaded {len(self.metrics_data)} step records")
        else:
            print(f"[WARNING]  Metrics file not found at {metrics_path}, proceeding with episode data only")
            
        # Identify numeric columns for analysis
        self.numeric_columns = self.episode_data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"[CHART] Found {len(self.numeric_columns)} numeric columns for analysis")
        
    def analyze_performance_overview(self):
        """Analyze overall performance metrics"""
        self.report_lines.append("# Active Inference Experiment Analysis Report")
        self.report_lines.append(f"\n**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report_lines.append(f"**Total Episodes:** {len(self.episode_data)}")
        
        # Success rate analysis
        if 'status' in self.episode_data.columns:
            status_counts = self.episode_data['status'].value_counts()
            total_episodes = len(self.episode_data)
            
            self.report_lines.append("\n## Episode Outcomes")
            for status, count in status_counts.items():
                percentage = (count / total_episodes) * 100
                self.report_lines.append(f"- **{status.title()}:** {count} episodes ({percentage:.1f}%)")
            
        # Key metrics summary
        key_metrics = ['steps_taken', 'final_distance', 'distance_improvement_percentage', 
                      'avg_vfe', 'avg_efe', 'collisions', 'replanning_count', 'duration_seconds']
        
        available_metrics = [m for m in key_metrics if m in self.episode_data.columns]
        
        if available_metrics:
            self.report_lines.append("\n## Key Performance Metrics")
            for metric in available_metrics:
                data = self.episode_data[metric].dropna()
                if len(data) > 0:
                    mean_val = data.mean()
                    std_val = data.std()
                    median_val = data.median()
                    self.report_lines.append(f"- **{metric.replace('_', ' ').title()}:** Mean={mean_val:.2f}, Std={std_val:.2f}, Median={median_val:.2f}")
    
    def plot_performance_dashboard(self):
        """Create comprehensive performance dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Active Inference Experiment - Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Success rate pie chart
        if 'status' in self.episode_data.columns:
            status_counts = self.episode_data['status'].value_counts()
            colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
            axes[0,0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                         colors=colors[:len(status_counts)])
            axes[0,0].set_title('Episode Outcomes', fontweight='bold')
        
        # Steps taken distribution by status
        if 'steps_taken' in self.episode_data.columns and 'status' in self.episode_data.columns:
            for status in self.episode_data['status'].unique():
                subset = self.episode_data[self.episode_data['status'] == status]['steps_taken']
                axes[0,1].hist(subset, alpha=0.7, label=status, bins=15)
            axes[0,1].set_title('Steps Taken Distribution by Status', fontweight='bold')
            axes[0,1].set_xlabel('Steps')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].legend()
        
        # Distance improvement
        if 'distance_improvement_percentage' in self.episode_data.columns:
            axes[0,2].hist(self.episode_data['distance_improvement_percentage'], bins=20, 
                          alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0,2].set_title('Distance Improvement Distribution', fontweight='bold')
            axes[0,2].set_xlabel('Improvement %')
            axes[0,2].set_ylabel('Frequency')
            axes[0,2].axvline(self.episode_data['distance_improvement_percentage'].mean(), 
                             color='red', linestyle='--', label='Mean')
            axes[0,2].legend()
        
        # VFE vs EFE scatter
        if 'avg_vfe' in self.episode_data.columns and 'avg_efe' in self.episode_data.columns:
            success_mask = self.episode_data['status'] == 'success'
            axes[1,0].scatter(self.episode_data.loc[success_mask, 'avg_vfe'], 
                            self.episode_data.loc[success_mask, 'avg_efe'], 
                            alpha=0.6, label='Success', color='green', s=50)
            axes[1,0].scatter(self.episode_data.loc[~success_mask, 'avg_vfe'], 
                            self.episode_data.loc[~success_mask, 'avg_efe'], 
                            alpha=0.6, label='Failure', color='red', s=50)
            axes[1,0].set_xlabel('Average VFE')
            axes[1,0].set_ylabel('Average EFE')
            axes[1,0].set_title('VFE vs EFE Relationship', fontweight='bold')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Collision analysis
        if 'collisions' in self.episode_data.columns:
            collision_data = self.episode_data['collisions'].value_counts().sort_index()
            bars = axes[1,1].bar(collision_data.index, collision_data.values, alpha=0.7, color='orange')
            axes[1,1].set_title('Collision Distribution', fontweight='bold')
            axes[1,1].set_xlabel('Number of Collisions')
            axes[1,1].set_ylabel('Episodes')
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                             f'{int(height)}', ha='center', va='bottom')
        
        # Learning curve (rolling average performance)
        if 'episode_id' in self.episode_data.columns and 'distance_improvement_percentage' in self.episode_data.columns:
            # Sort by episode_id and calculate rolling average
            sorted_data = self.episode_data.sort_values('episode_id')
            window_size = min(10, len(sorted_data) // 5)  # Adaptive window size
            rolling_perf = sorted_data['distance_improvement_percentage'].rolling(window=window_size, min_periods=1).mean()
            axes[1,2].plot(sorted_data['episode_id'], rolling_perf, alpha=0.8, color='purple', linewidth=2)
            axes[1,2].scatter(sorted_data['episode_id'], sorted_data['distance_improvement_percentage'], 
                             alpha=0.3, color='purple', s=20)
            axes[1,2].set_title(f'Learning Curve (Rolling Avg, window={window_size})', fontweight='bold')
            axes[1,2].set_xlabel('Episode ID')
            axes[1,2].set_ylabel('Distance Improvement %')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("OK Performance dashboard saved")
    
    def analyze_behavioral_patterns(self):
        """Analyze behavioral patterns and correlations"""
        if len(self.numeric_columns) < 2:
            return
            
        # Correlation analysis
        correlation_matrix = self.episode_data[self.numeric_columns].corr()
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('Correlation Matrix of Performance Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find strongest correlations
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:  # Only significant correlations
                    correlation_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
          # Sort by absolute correlation value
        correlation_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        self.report_lines.append("\n## Key Behavioral Correlations")
        for var1, var2, corr in correlation_pairs[:10]:  # Top 10
            direction = "positive" if corr > 0 else "negative"
            self.report_lines.append(f"- **{var1}** vs **{var2}**: {corr:.3f} ({direction})")
        
        print("OK Behavioral pattern analysis completed")
    
    def analyze_vfe_efe_dynamics(self):
        """Enhanced analysis of VFE and EFE dynamics with logarithmic normalization"""
        if self.metrics_data is None:
            return
        
        # Create comprehensive dynamics analysis with multiple visualizations
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Enhanced VFE/EFE Dynamics Analysis (Lower Values = Better Performance)', 
                     fontsize=18, fontweight='bold')
        
        # Get more episodes for trajectory analysis (up to 15 episodes)
        all_episodes = self.metrics_data['episode_id'].unique()
        num_episodes = min(15, len(all_episodes))
        sample_episodes = all_episodes[:num_episodes]
        
        # Colors for different trajectories
        colors = plt.cm.tab20(np.linspace(0, 1, num_episodes))
        
        # 1. Raw VFE trajectories (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        for i, ep_id in enumerate(sample_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'vfe' in ep_data.columns and len(ep_data) > 0:
                ax1.plot(ep_data['step'], ep_data['vfe'], alpha=0.7, 
                        color=colors[i], label=f'Ep {ep_id}', linewidth=1.5)
        ax1.set_title('Raw VFE Trajectories', fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('VFE (Raw)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 2. Log-normalized VFE trajectories (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        for i, ep_id in enumerate(sample_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'vfe' in ep_data.columns and len(ep_data) > 0:
                # Apply log normalization (add small constant to avoid log(0))
                log_vfe = np.log(np.abs(ep_data['vfe']) + 1e-6)
                ax2.plot(ep_data['step'], log_vfe, alpha=0.7, 
                        color=colors[i], label=f'Ep {ep_id}', linewidth=1.5)
        ax2.set_title('Log-Normalized VFE Trajectories', fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Log(|VFE|)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Raw EFE trajectories (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        for i, ep_id in enumerate(sample_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'efe' in ep_data.columns and len(ep_data) > 0:
                ax3.plot(ep_data['step'], ep_data['efe'], alpha=0.7, 
                        color=colors[i], label=f'Ep {ep_id}', linewidth=1.5)
        ax3.set_title('Raw EFE Trajectories', fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('EFE (Raw)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Log-normalized EFE trajectories (second row left)
        ax4 = fig.add_subplot(gs[1, 0])
        for i, ep_id in enumerate(sample_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'efe' in ep_data.columns and len(ep_data) > 0:
                # Apply log normalization for EFE (handle negative values)
                log_efe = np.log(np.abs(ep_data['efe']) + 1e-6)
                ax4.plot(ep_data['step'], log_efe, alpha=0.7, 
                        color=colors[i], label=f'Ep {ep_id}', linewidth=1.5)
        ax4.set_title('Log-Normalized EFE Trajectories', fontweight='bold')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Log(|EFE|)')
        ax4.grid(True, alpha=0.3)
        
        # 5. VFE vs EFE scatter with success/failure coloring (second row middle)
        ax5 = fig.add_subplot(gs[1, 1])
        if 'vfe' in self.metrics_data.columns and 'efe' in self.metrics_data.columns:
            # Merge with episode data to get success status
            merged_data = self.metrics_data.merge(
                self.episode_data[['episode_id', 'status']], 
                on='episode_id', how='left'
            )
            
            # Sample for performance but maintain balance
            sample_size = min(1500, len(merged_data))
            sample_data = merged_data.sample(sample_size)
            
            # Plot success vs failure
            success_mask = sample_data['status'] == 'success'
            ax5.scatter(sample_data.loc[success_mask, 'vfe'], 
                       sample_data.loc[success_mask, 'efe'], 
                       alpha=0.6, c='green', s=15, label='Success')
            ax5.scatter(sample_data.loc[~success_mask, 'vfe'], 
                       sample_data.loc[~success_mask, 'efe'], 
                       alpha=0.6, c='red', s=15, label='Failure')
            ax5.set_xlabel('VFE')
            ax5.set_ylabel('EFE')
            ax5.set_title('VFE vs EFE by Episode Outcome', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
        
        # 6. Log-normalized VFE vs EFE scatter (second row right)
        ax6 = fig.add_subplot(gs[1, 2])
        if 'vfe' in self.metrics_data.columns and 'efe' in self.metrics_data.columns:
            merged_data = self.metrics_data.merge(
                self.episode_data[['episode_id', 'status']], 
                on='episode_id', how='left'
            )
            sample_data = merged_data.sample(min(1500, len(merged_data)))
            
            # Log normalize
            log_vfe = np.log(np.abs(sample_data['vfe']) + 1e-6)
            log_efe = np.log(np.abs(sample_data['efe']) + 1e-6)
            
            success_mask = sample_data['status'] == 'success'
            ax6.scatter(log_vfe[success_mask], log_efe[success_mask], 
                       alpha=0.6, c='green', s=15, label='Success')
            ax6.scatter(log_vfe[~success_mask], log_efe[~success_mask], 
                       alpha=0.6, c='red', s=15, label='Failure')
            ax6.set_xlabel('Log(|VFE|)')
            ax6.set_ylabel('Log(|EFE|)')
            ax6.set_title('Log-Normalized VFE vs EFE', fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
        
        # 7. VFE/EFE minimization trends over episodes (third row left)
        ax7 = fig.add_subplot(gs[2, 0])
        if 'vfe' in self.metrics_data.columns:
            # Calculate average VFE per episode
            episode_vfe_means = self.metrics_data.groupby('episode_id')['vfe'].mean()
            episode_ids_sorted = sorted(episode_vfe_means.index)
            vfe_means_sorted = [episode_vfe_means[ep_id] for ep_id in episode_ids_sorted]
            
            ax7.plot(episode_ids_sorted, vfe_means_sorted, 'b-', alpha=0.7, linewidth=2, label='Episode Mean VFE')
            # Add rolling average to show trend
            if len(vfe_means_sorted) > 5:
                window = min(10, len(vfe_means_sorted)//3)
                rolling_mean = pd.Series(vfe_means_sorted).rolling(window=window, min_periods=1).mean()
                ax7.plot(episode_ids_sorted, rolling_mean, 'r-', linewidth=3, 
                        label=f'Rolling Mean (window={window})')
            ax7.set_xlabel('Episode ID')
            ax7.set_ylabel('Average VFE')
            ax7.set_title('VFE Minimization Trend Across Episodes', fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
        
        # 8. EFE minimization trends over episodes (third row middle)
        ax8 = fig.add_subplot(gs[2, 1])
        if 'efe' in self.metrics_data.columns:
            episode_efe_means = self.metrics_data.groupby('episode_id')['efe'].mean()
            episode_ids_sorted = sorted(episode_efe_means.index)
            efe_means_sorted = [episode_efe_means[ep_id] for ep_id in episode_ids_sorted]
            
            ax8.plot(episode_ids_sorted, efe_means_sorted, 'purple', alpha=0.7, linewidth=2, label='Episode Mean EFE')
            if len(efe_means_sorted) > 5:
                window = min(10, len(efe_means_sorted)//3)
                rolling_mean = pd.Series(efe_means_sorted).rolling(window=window, min_periods=1).mean()
                ax8.plot(episode_ids_sorted, rolling_mean, 'orange', linewidth=3, 
                        label=f'Rolling Mean (window={window})')
            ax8.set_xlabel('Episode ID')
            ax8.set_ylabel('Average EFE')
            ax8.set_title('EFE Minimization Trend Across Episodes', fontweight='bold')
            ax8.grid(True, alpha=0.3)
            ax8.legend()
        
        # 9. Combined VFE + EFE minimization (third row right)
        ax9 = fig.add_subplot(gs[2, 2])
        if 'vfe' in self.metrics_data.columns and 'efe' in self.metrics_data.columns:
            # Calculate combined energy (sum of absolute values)
            combined_energy = self.metrics_data.groupby('episode_id').apply(
                lambda x: (np.abs(x['vfe']) + np.abs(x['efe'])).mean()
            )
            episode_ids_sorted = sorted(combined_energy.index)
            combined_sorted = [combined_energy[ep_id] for ep_id in episode_ids_sorted]
            
            ax9.plot(episode_ids_sorted, combined_sorted, 'darkgreen', alpha=0.7, linewidth=2, 
                    label='Combined |VFE| + |EFE|')
            if len(combined_sorted) > 5:
                window = min(10, len(combined_sorted)//3)
                rolling_mean = pd.Series(combined_sorted).rolling(window=window, min_periods=1).mean()
                ax9.plot(episode_ids_sorted, rolling_mean, 'red', linewidth=3, 
                        label=f'Rolling Mean (window={window})')
            ax9.set_xlabel('Episode ID')
            ax9.set_ylabel('Combined Energy')
            ax9.set_title('Total Energy Minimization Trend', fontweight='bold')
            ax9.grid(True, alpha=0.3)
            ax9.legend()
        
        # 10. VFE/EFE distribution comparison (fourth row left)
        ax10 = fig.add_subplot(gs[3, 0])
        if 'vfe' in self.metrics_data.columns and 'efe' in self.metrics_data.columns:
            ax10.hist(self.metrics_data['vfe'], bins=50, alpha=0.5, label='VFE', color='blue', density=True)
            ax10.hist(np.abs(self.metrics_data['efe']), bins=50, alpha=0.5, label='|EFE|', color='red', density=True)
            ax10.set_xlabel('Energy Value')
            ax10.set_ylabel('Density')
            ax10.set_title('VFE vs |EFE| Distribution', fontweight='bold')
            ax10.grid(True, alpha=0.3)
            ax10.legend()
            ax10.set_yscale('log')  # Log scale for better visualization
        
        # 11. Distance vs Energy relationship (fourth row middle)
        ax11 = fig.add_subplot(gs[3, 1])
        if 'distance_to_target' in self.metrics_data.columns and 'vfe' in self.metrics_data.columns:
            sample_data = self.metrics_data.sample(min(1000, len(self.metrics_data)))
            ax11.scatter(sample_data['distance_to_target'], sample_data['vfe'], 
                        alpha=0.5, s=10, color='blue', label='VFE vs Distance')
            ax11.set_xlabel('Distance to Target')
            ax11.set_ylabel('VFE')
            ax11.set_title('Distance vs VFE Relationship', fontweight='bold')
            ax11.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(sample_data['distance_to_target'], sample_data['vfe'], 1)
            p = np.poly1d(z)
            ax11.plot(sample_data['distance_to_target'], p(sample_data['distance_to_target']), 
                     "r--", alpha=0.8, linewidth=2, label='Trend Line')
            ax11.legend()
        
        # 12. Energy variance analysis (fourth row right)
        ax12 = fig.add_subplot(gs[3, 2])
        if 'vfe' in self.metrics_data.columns and 'efe' in self.metrics_data.columns:
            episode_vfe_var = self.metrics_data.groupby('episode_id')['vfe'].var()
            episode_efe_var = self.metrics_data.groupby('episode_id')['efe'].var()
            
            episode_ids = sorted(episode_vfe_var.index)
            vfe_vars = [episode_vfe_var[ep_id] for ep_id in episode_ids]
            efe_vars = [episode_efe_var[ep_id] for ep_id in episode_ids]
            
            ax12.plot(episode_ids, vfe_vars, 'b-', alpha=0.7, linewidth=2, label='VFE Variance')
            ax12.plot(episode_ids, efe_vars, 'r-', alpha=0.7, linewidth=2, label='EFE Variance')
            ax12.set_xlabel('Episode ID')
            ax12.set_ylabel('Energy Variance')
            ax12.set_title('Energy Stability Across Episodes', fontweight='bold')
            ax12.grid(True, alpha=0.3)
            ax12.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'enhanced_vfe_efe_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate detailed VFE/EFE statistics for the report
        self.report_lines.append("\n## Enhanced VFE/EFE Dynamics Analysis")
        
        if 'vfe' in self.metrics_data.columns:
            vfe_stats = self.metrics_data['vfe'].describe()
            log_vfe_stats = np.log(np.abs(self.metrics_data['vfe']) + 1e-6).describe()
            self.report_lines.append(f"\n### VFE Statistics:")
            self.report_lines.append(f"- **Raw VFE**: Mean={vfe_stats['mean']:.2f}, Std={vfe_stats['std']:.2f}, Min={vfe_stats['min']:.2f}, Max={vfe_stats['max']:.2f}")
            self.report_lines.append(f"- **Log-Normalized VFE**: Mean={log_vfe_stats['mean']:.2f}, Std={log_vfe_stats['std']:.2f}")
        
        if 'efe' in self.metrics_data.columns:
            efe_stats = self.metrics_data['efe'].describe()
            log_efe_stats = np.log(np.abs(self.metrics_data['efe']) + 1e-6).describe()
            self.report_lines.append(f"\n### EFE Statistics:")
            self.report_lines.append(f"- **Raw EFE**: Mean={efe_stats['mean']:.2f}, Std={efe_stats['std']:.2f}, Min={efe_stats['min']:.2f}, Max={efe_stats['max']:.2f}")
            self.report_lines.append(f"- **Log-Normalized EFE**: Mean={log_efe_stats['mean']:.2f}, Std={log_efe_stats['std']:.2f}")
        
        # Energy minimization trends
        if 'vfe' in self.metrics_data.columns:
            episode_vfe_means = self.metrics_data.groupby('episode_id')['vfe'].mean()
            vfe_trend = np.polyfit(range(len(episode_vfe_means)), episode_vfe_means.values, 1)[0]
            self.report_lines.append(f"\n### Energy Minimization Trends:")
            self.report_lines.append(f"- **VFE Trend**: {vfe_trend:.4f} per episode ({'Improving' if vfe_trend < 0 else 'Worsening'})")
        
        if 'efe' in self.metrics_data.columns:
            episode_efe_means = self.metrics_data.groupby('episode_id')['efe'].mean()
            efe_trend = np.polyfit(range(len(episode_efe_means)), episode_efe_means.values, 1)[0]
            self.report_lines.append(f"- **EFE Trend**: {efe_trend:.4f} per episode ({'Improving' if efe_trend > 0 else 'Worsening'})")
        
        print("OK Enhanced VFE/EFE dynamics analysis completed")
    
    def analyze_planning_behavior(self):
        """Analyze planning and replanning patterns"""
        planning_metrics = ['replanning_count', 'dynamic_replanning_count', 'avg_planning_time_ms']
        available_planning = [m for m in planning_metrics if m in self.episode_data.columns]
        
        if not available_planning:
            return
            
        fig, axes = plt.subplots(1, len(available_planning), figsize=(6*len(available_planning), 5))
        if len(available_planning) == 1:
            axes = [axes]
            
        for i, metric in enumerate(available_planning):
            # Distribution by success status
            if 'status' in self.episode_data.columns:
                for status in self.episode_data['status'].unique():
                    subset = self.episode_data[self.episode_data['status'] == status][metric]
                    axes[i].hist(subset, alpha=0.6, label=status, bins=15)
                axes[i].legend()
            else:
                axes[i].hist(self.episode_data[metric], bins=20, alpha=0.7)
            
            axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
            axes[i].set_xlabel(metric.replace("_", " ").title())
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Planning Behavior Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'planning_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Planning statistics
        self.report_lines.append("\n## Planning Behavior Statistics")
        for metric in available_planning:
            data = self.episode_data[metric].dropna()
            if len(data) > 0:
                mean_val = data.mean()
                median_val = data.median()
                max_val = data.max()
                self.report_lines.append(f"- **{metric.replace('_', ' ').title()}:** Mean={mean_val:.2f}, Median={median_val:.2f}, Max={max_val:.2f}")
        
        print("OK Planning behavior analysis completed")
    
    def analyze_success_factors(self):
        """Analyze factors that contribute to success vs failure"""
        if 'status' not in self.episode_data.columns:
            return
            
        # Compare successful vs unsuccessful episodes
        success_data = self.episode_data[self.episode_data['status'] == 'success']
        failure_data = self.episode_data[self.episode_data['status'] != 'success']
        
        if len(success_data) == 0 or len(failure_data) == 0:
            return
        
        comparison_metrics = ['steps_taken', 'avg_vfe', 'avg_efe', 'collisions', 'replanning_count', 'avg_planning_time_ms']
        available_comparison = [m for m in comparison_metrics if m in self.episode_data.columns]
        
        self.report_lines.append("\n## Success vs Failure Analysis")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_comparison[:6]):
            if i >= 6:
                break
                
            success_vals = success_data[metric].dropna()
            failure_vals = failure_data[metric].dropna()
            
            if len(success_vals) > 0 and len(failure_vals) > 0:
                # Box plot comparison
                data_to_plot = [success_vals, failure_vals]
                axes[i].boxplot(data_to_plot, labels=['Success', 'Failure'])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
                
                # Statistical comparison
                success_mean = success_vals.mean()
                failure_mean = failure_vals.mean()
                diff_percent = ((success_mean - failure_mean) / failure_mean) * 100 if failure_mean != 0 else 0
                
                self.report_lines.append(f"- **{metric.replace('_', ' ').title()}:** Success={success_mean:.2f}, Failure={failure_mean:.2f} (Diff: {diff_percent:+.1f}%)")
        
        plt.suptitle('Success vs Failure Factor Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'success_factors.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("OK Success factor analysis completed")
    
    def generate_summary_report(self):
        """Generate a comprehensive text report"""
        report_path = os.path.join(RESULTS_DIR, 'analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        
        print(f"OK Analysis report saved to {report_path}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print(">> Starting Single Environment Active Inference Analysis")
        print("=" * 60)
        
        try:
            # Load data
            self.load_data()
            
            # Run analyses
            print("\n[CHART] Analyzing performance overview...")
            self.analyze_performance_overview()
            
            print("[CHART] Creating performance dashboard...")
            self.plot_performance_dashboard()
            
            print("[CHART] Analyzing behavioral patterns...")
            self.analyze_behavioral_patterns()
            
            if self.metrics_data is not None:
                print("[CHART] Analyzing VFE/EFE dynamics...")
                self.analyze_vfe_efe_dynamics()
            
            print("[CHART] Analyzing planning behavior...")
            self.analyze_planning_behavior()
            
            print("[CHART] Analyzing success factors...")
            self.analyze_success_factors()
            
            print("[REPORT] Generating summary report...")
            self.generate_summary_report()
            
            print("\n[SUCCESS] Analysis completed successfully!")
            print(f"[FOLDER] Results saved in: {RESULTS_DIR}")
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function"""
    analyzer = SingleEnvironmentAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
