"""
SINGLE ENVIRONMENT ANALYSIS FOR ACTIVE INFERENCE EXPERIMENTS

[BRAIN] Purpose:
Analyze episode data from a single environment experiment with comprehensive metrics analysis.
Displays ALL trajectory data over steps for complete test analysis (no sampling).

⚠️  IMPORTANT NOTE ABOUT VFE VALUES:
The "VFE" values in this analysis are DERIVED from EFE pragmatic components, NOT true 
Variational Free Energy calculations. The system uses Gaussian kernel belief updating.
VFE = -EFE_pragmatic_component (derived metric for analysis purposes only).

[FOLDER] Expected Input:
- episode_summaries.csv: Main episode data with performance metrics
- metrics.csv: Step-by-step detailed metrics during episodes

[TOOLS] Key Features:
- Performance analysis and success rate calculations
- Derived VFE/EFE analysis showing ALL trajectories over steps
- Planning and replanning analysis
- Collision and obstacle avoidance metrics
- Temporal dynamics and learning curves
- Complete step-by-step behavioral analysis for all episodes
- Comprehensive visualizations and reporting

[METRICS] VFE-EFE Relationship:
The perfect correlation between VFE and EFE (r ≈ -0.999987) is a mathematical artifact
because VFE = -EFE_pragmatic. This is NOT evidence of deep Active Inference coupling.
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
        """Create comprehensive performance dashboard with both raw and log-normalized Derived VFE vs EFE analysis"""
        # Expand to 3x3 grid to accommodate log-normalized Derived VFE vs EFE
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('Single Environment Performance Dashboard (Enhanced Derived VFE/EFE Analysis)', fontsize=16, fontweight='bold')
        
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
        
        # Raw Derived VFE vs EFE scatter
        if 'avg_vfe' in self.episode_data.columns and 'avg_efe' in self.episode_data.columns:
            success_mask = self.episode_data['status'] == 'success'
            axes[1,0].scatter(self.episode_data.loc[success_mask, 'avg_vfe'], 
                            self.episode_data.loc[success_mask, 'avg_efe'], 
                            alpha=0.6, label='Success', color='green', s=50)
            axes[1,0].scatter(self.episode_data.loc[~success_mask, 'avg_vfe'], 
                            self.episode_data.loc[~success_mask, 'avg_efe'], 
                            alpha=0.6, label='Failure', color='red', s=50)
            axes[1,0].set_xlabel('Average Derived VFE (Raw)')
            axes[1,0].set_ylabel('Average EFE (Raw)')
            axes[1,0].set_title('Raw Derived VFE vs EFE Relationship', fontweight='bold')
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
        
        # NEW: Log-normalized VFE vs EFE scatter (row 3, col 1)
        if 'avg_vfe' in self.episode_data.columns and 'avg_efe' in self.episode_data.columns:
            success_mask = self.episode_data['status'] == 'success'
            
            # Apply log normalization
            log_vfe = np.log(np.abs(self.episode_data['avg_vfe']) + 1e-6)
            log_efe = np.log(np.abs(self.episode_data['avg_efe']) + 1e-6)
            
            axes[2,0].scatter(log_vfe[success_mask], log_efe[success_mask], 
                            alpha=0.6, label='Success', color='green', s=50)
            axes[2,0].scatter(log_vfe[~success_mask], log_efe[~success_mask], 
                            alpha=0.6, label='Failure', color='red', s=50)
            axes[2,0].set_xlabel('Log(|Average VFE|)')
            axes[2,0].set_ylabel('Log(|Average EFE|)')
            axes[2,0].set_title('Log-Normalized VFE vs EFE Relationship', fontweight='bold')
            axes[2,0].legend()
            axes[2,0].grid(True, alpha=0.3)
        
        # NEW: VFE/EFE Energy Distribution Comparison (row 3, col 2)
        if 'avg_vfe' in self.episode_data.columns and 'avg_efe' in self.episode_data.columns:
            axes[2,1].hist(self.episode_data['avg_vfe'], bins=20, alpha=0.6, 
                          label='VFE', color='blue', density=True)
            axes[2,1].hist(np.abs(self.episode_data['avg_efe']), bins=20, alpha=0.6, 
                          label='|EFE|', color='red', density=True)
            axes[2,1].set_xlabel('Energy Value')
            axes[2,1].set_ylabel('Density')
            axes[2,1].set_title('VFE vs |EFE| Distribution Comparison', fontweight='bold')
            axes[2,1].legend()
            axes[2,1].grid(True, alpha=0.3)
        
        # NEW: Combined Energy Analysis (row 3, col 3)
        if 'avg_vfe' in self.episode_data.columns and 'avg_efe' in self.episode_data.columns:
            # Calculate combined energy
            combined_energy = np.abs(self.episode_data['avg_vfe']) + np.abs(self.episode_data['avg_efe'])
            success_mask = self.episode_data['status'] == 'success'
            
            axes[2,2].hist(combined_energy[success_mask], bins=15, alpha=0.6, 
                          label='Success', color='green', density=True)
            axes[2,2].hist(combined_energy[~success_mask], bins=15, alpha=0.6, 
                          label='Failure', color='red', density=True)
            axes[2,2].set_xlabel('Combined Energy (|VFE| + |EFE|)')
            axes[2,2].set_ylabel('Density')
            axes[2,2].set_title('Combined Energy by Episode Outcome', fontweight='bold')
            axes[2,2].legend()
            axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("OK Enhanced performance dashboard with VFE/EFE analysis saved")
    
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
        fig.suptitle('Single Environment VFE/EFE Dynamics - ALL Trajectories Over Steps (Test Analysis)', 
                     fontsize=18, fontweight='bold')
        
        # Get ALL episodes for trajectory analysis (show complete dataset)
        all_episodes = sorted(self.metrics_data['episode_id'].unique())
        
        # Merge with episode status for grouping
        episode_status = self.episode_data.set_index('episode_id')['status'].to_dict()
        success_episodes = [ep for ep in all_episodes if episode_status.get(ep) == 'success']
        failure_episodes = [ep for ep in all_episodes if episode_status.get(ep) != 'success']
        
        # Use ALL episodes (no sampling - show complete dataset)
        num_success = len(success_episodes)
        num_failure = len(failure_episodes)
        sample_success = success_episodes  # All successful episodes
        sample_failure = failure_episodes  # All failed episodes
        
        # Create sophisticated color schemes for ALL episodes
        if num_success > 0:
            success_colors = plt.cm.Greens(np.linspace(0.3, 0.9, num_success))
        else:
            success_colors = []
        
        if num_failure > 0:
            failure_colors = plt.cm.Reds(np.linspace(0.3, 0.9, num_failure))
        else:
            failure_colors = []
        
        # 1. Raw VFE trajectories with grouped styling (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Plot successful episodes with green tones
        for i, ep_id in enumerate(sample_success):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'vfe' in ep_data.columns and len(ep_data) > 0:
                ax1.plot(ep_data['step'], ep_data['vfe'], alpha=0.6, 
                        color=success_colors[i], linewidth=1.2)
        
        # Plot failed episodes with red tones
        for i, ep_id in enumerate(sample_failure):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'vfe' in ep_data.columns and len(ep_data) > 0:
                ax1.plot(ep_data['step'], ep_data['vfe'], alpha=0.6, 
                        color=failure_colors[i], linewidth=1.2, linestyle='--')
        
        # Add representative average trajectories
        if sample_success:
            success_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in sample_success])
            for step in range(max_steps):
                step_values = []
                for ep_id in sample_success:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step and 'vfe' in ep_data.columns:
                        step_values.append(ep_data.iloc[step]['vfe'])
                if step_values:
                    success_avg.append(np.mean(step_values))
            if success_avg:
                ax1.plot(range(len(success_avg)), success_avg, color='darkgreen', 
                        linewidth=3, label=f'Success Avg (n={num_success})', alpha=0.9)
        
        if sample_failure:
            failure_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in sample_failure])
            for step in range(max_steps):
                step_values = []
                for ep_id in sample_failure:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step and 'vfe' in ep_data.columns:
                        step_values.append(ep_data.iloc[step]['vfe'])
                if step_values:
                    failure_avg.append(np.mean(step_values))
            if failure_avg:
                ax1.plot(range(len(failure_avg)), failure_avg, color='darkred', 
                        linewidth=3, label=f'Failure Avg (n={num_failure})', alpha=0.9, linestyle='-.')
        
        ax1.set_title(f'Raw VFE Trajectories - ALL Episodes (Success: {num_success}, Failure: {num_failure})', fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('VFE (Raw)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 2. Log-normalized VFE trajectories with grouped styling (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Plot successful episodes
        for i, ep_id in enumerate(sample_success):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'vfe' in ep_data.columns and len(ep_data) > 0:
                log_vfe = np.log(np.abs(ep_data['vfe']) + 1e-6)
                ax2.plot(ep_data['step'], log_vfe, alpha=0.6, 
                        color=success_colors[i], linewidth=1.2)
        
        # Plot failed episodes
        for i, ep_id in enumerate(sample_failure):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'vfe' in ep_data.columns and len(ep_data) > 0:
                log_vfe = np.log(np.abs(ep_data['vfe']) + 1e-6)
                ax2.plot(ep_data['step'], log_vfe, alpha=0.6, 
                        color=failure_colors[i], linewidth=1.2, linestyle='--')
        
        # Add average trajectories for log-normalized VFE
        if sample_success:
            success_log_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in sample_success])
            for step in range(max_steps):
                step_values = []
                for ep_id in sample_success:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step and 'vfe' in ep_data.columns:
                        step_values.append(np.log(np.abs(ep_data.iloc[step]['vfe']) + 1e-6))
                if step_values:
                    success_log_avg.append(np.mean(step_values))
            if success_log_avg:
                ax2.plot(range(len(success_log_avg)), success_log_avg, color='darkgreen', 
                        linewidth=3, label=f'Success Avg (n={num_success})', alpha=0.9)
        
        if sample_failure:
            failure_log_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in sample_failure])
            for step in range(max_steps):
                step_values = []
                for ep_id in sample_failure:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step and 'vfe' in ep_data.columns:
                        step_values.append(np.log(np.abs(ep_data.iloc[step]['vfe']) + 1e-6))
                if step_values:
                    failure_log_avg.append(np.mean(step_values))
            if failure_log_avg:
                ax2.plot(range(len(failure_log_avg)), failure_log_avg, color='darkred', 
                        linewidth=3, label=f'Failure Avg (n={num_failure})', alpha=0.9, linestyle='-.')
        
        ax2.set_title(f'Log-Normalized VFE Trajectories - ALL Episodes (Success: {num_success}, Failure: {num_failure})', fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Log(|VFE|)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # 3. Raw EFE trajectories with grouped styling (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Plot successful episodes
        for i, ep_id in enumerate(sample_success):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'efe' in ep_data.columns and len(ep_data) > 0:
                ax3.plot(ep_data['step'], ep_data['efe'], alpha=0.6, 
                        color=success_colors[i], linewidth=1.2)
        
        # Plot failed episodes
        for i, ep_id in enumerate(sample_failure):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'efe' in ep_data.columns and len(ep_data) > 0:
                ax3.plot(ep_data['step'], ep_data['efe'], alpha=0.6, 
                        color=failure_colors[i], linewidth=1.2, linestyle='--')
        
        # Add average trajectories for EFE
        if sample_success:
            success_efe_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in sample_success])
            for step in range(max_steps):
                step_values = []
                for ep_id in sample_success:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step and 'efe' in ep_data.columns:
                        step_values.append(ep_data.iloc[step]['efe'])
                if step_values:
                    success_efe_avg.append(np.mean(step_values))
            if success_efe_avg:
                ax3.plot(range(len(success_efe_avg)), success_efe_avg, color='darkgreen', 
                        linewidth=3, label=f'Success Avg (n={num_success})', alpha=0.9)
        
        if sample_failure:
            failure_efe_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in sample_failure])
            for step in range(max_steps):
                step_values = []
                for ep_id in sample_failure:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step and 'efe' in ep_data.columns:
                        step_values.append(ep_data.iloc[step]['efe'])
                if step_values:
                    failure_efe_avg.append(np.mean(step_values))
            if failure_efe_avg:
                ax3.plot(range(len(failure_efe_avg)), failure_efe_avg, color='darkred', 
                        linewidth=3, label=f'Failure Avg (n={num_failure})', alpha=0.9, linestyle='-.')
        
        ax3.set_title(f'Raw EFE Trajectories - ALL Episodes (Success: {num_success}, Failure: {num_failure})', fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('EFE (Raw)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        # 4. Log-normalized EFE trajectories with grouped styling (second row left)
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Plot successful episodes
        for i, ep_id in enumerate(sample_success):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'efe' in ep_data.columns and len(ep_data) > 0:
                log_efe = np.log(np.abs(ep_data['efe']) + 1e-6)
                ax4.plot(ep_data['step'], log_efe, alpha=0.6, 
                        color=success_colors[i], linewidth=1.2)
        
        # Plot failed episodes
        for i, ep_id in enumerate(sample_failure):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'efe' in ep_data.columns and len(ep_data) > 0:
                log_efe = np.log(np.abs(ep_data['efe']) + 1e-6)
                ax4.plot(ep_data['step'], log_efe, alpha=0.6, 
                        color=failure_colors[i], linewidth=1.2, linestyle='--')
        
        # Add average trajectories for log-normalized EFE
        if sample_success:
            success_log_efe_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in sample_success])
            for step in range(max_steps):
                step_values = []
                for ep_id in success_episodes:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step and 'efe' in ep_data.columns:
                        step_values.append(np.log(np.abs(ep_data.iloc[step]['efe']) + 1e-6))
                if step_values:
                    success_log_efe_avg.append(np.mean(step_values))
            if success_log_efe_avg:
                ax4.plot(range(len(success_log_efe_avg)), success_log_efe_avg, color='darkgreen', 
                        linewidth=3, label=f'Success Avg (n={num_success})', alpha=0.9)
        
        if sample_failure:
            failure_log_efe_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in sample_failure])
            for step in range(max_steps):
                step_values = []
                for ep_id in failure_episodes:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step and 'efe' in ep_data.columns:
                        step_values.append(np.log(np.abs(ep_data.iloc[step]['efe']) + 1e-6))
                if step_values:
                    failure_log_efe_avg.append(np.mean(step_values))
            if failure_log_efe_avg:
                ax4.plot(range(len(failure_log_efe_avg)), failure_log_efe_avg, color='darkred', 
                        linewidth=3, label=f'Failure Avg (n={num_failure})', alpha=0.9, linestyle='-.')
        
        ax4.set_title(f'Log-Normalized EFE Trajectories - ALL Episodes (Success: {num_success}, Failure: {num_failure})', fontweight='bold')
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
            
            # Use ALL data points (no sampling for complete analysis)
            sample_data = merged_data
            
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
            ax5.set_title('VFE vs EFE - ALL Data Points by Outcome', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
        
        # 6. Log-normalized VFE vs EFE scatter (second row right)
        ax6 = fig.add_subplot(gs[1, 2])
        if 'vfe' in self.metrics_data.columns and 'efe' in self.metrics_data.columns:
            merged_data = self.metrics_data.merge(
                self.episode_data[['episode_id', 'status']], 
                on='episode_id', how='left'
            )
            sample_data = merged_data  # Use ALL data points
            
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
            ax6.set_title('Log-Normalized VFE vs EFE - ALL Data Points', fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
        
        # 7. VFE/EFE minimization trends over episodes (third row left) - with normalization
        ax7 = fig.add_subplot(gs[2, 0])
        if 'vfe' in self.metrics_data.columns:
            # Calculate average VFE per episode
            episode_vfe_means = self.metrics_data.groupby('episode_id')['vfe'].mean()
            episode_ids_sorted = sorted(episode_vfe_means.index)
            vfe_means_sorted = [episode_vfe_means[ep_id] for ep_id in episode_ids_sorted]
            
            # Normalize VFE values for trend visualization
            vfe_min, vfe_max = min(vfe_means_sorted), max(vfe_means_sorted)
            vfe_normalized = [(val - vfe_min) / (vfe_max - vfe_min + 1e-6) for val in vfe_means_sorted]
            
            ax7.plot(episode_ids_sorted, vfe_normalized, 'b-', alpha=0.7, linewidth=2, label='Normalized VFE')
            # Add rolling average to show trend
            if len(vfe_normalized) > 5:
                window = min(10, len(vfe_normalized)//3)
                rolling_mean = pd.Series(vfe_normalized).rolling(window=window, min_periods=1).mean()
                ax7.plot(episode_ids_sorted, rolling_mean, 'r-', linewidth=3, 
                        label=f'Rolling Mean (window={window})')
            ax7.set_xlabel('Episode ID')
            ax7.set_ylabel('Normalized VFE (0-1 scale)')
            ax7.set_title('Normalized VFE Minimization Trend Across Episodes', fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
        
        # 8. EFE minimization trends over episodes (third row middle)
        ax8 = fig.add_subplot(gs[2, 1])
        if 'efe' in self.metrics_data.columns:
            episode_efe_means = self.metrics_data.groupby('episode_id')['efe'].mean()
            episode_ids_sorted = sorted(episode_efe_means.index)
            efe_means_sorted = [episode_efe_means[ep_id] for ep_id in episode_ids_sorted]
            
            # Normalize EFE values for better trend visualization
            efe_min = min(efe_means_sorted)
            efe_max = max(efe_means_sorted)
            efe_normalized = [(val - efe_min) / (efe_max - efe_min + 1e-6) for val in efe_means_sorted]
            
            ax8.plot(episode_ids_sorted, efe_normalized, 'purple', alpha=0.7, linewidth=2, label='Episode Mean EFE (Normalized)')
            if len(efe_normalized) > 5:
                window = min(10, len(efe_normalized)//3)
                rolling_mean = pd.Series(efe_normalized).rolling(window=window, min_periods=1).mean()
                ax8.plot(episode_ids_sorted, rolling_mean, 'orange', linewidth=3, 
                        label=f'Rolling Mean (window={window})')
            ax8.set_xlabel('Episode ID')
            ax8.set_ylabel('Normalized EFE (0-1 scale)')
            ax8.set_title('Normalized EFE Minimization Trend Across Episodes', fontweight='bold')
            ax8.grid(True, alpha=0.3)
            ax8.legend()
        
        # 9. Combined VFE + EFE minimization (third row right)
        ax9 = fig.add_subplot(gs[2, 2])
        if 'vfe' in self.metrics_data.columns and 'efe' in self.metrics_data.columns:
            # Calculate combined energy using normalized values
            episode_vfe_means = self.metrics_data.groupby('episode_id')['vfe'].mean()
            episode_efe_means = self.metrics_data.groupby('episode_id')['efe'].mean()
            
            # Normalize both VFE and EFE values (0-1 scale)
            vfe_normalized = (episode_vfe_means.values - episode_vfe_means.min()) / (episode_vfe_means.max() - episode_vfe_means.min() + 1e-6)
            efe_normalized = (episode_efe_means.values - episode_efe_means.min()) / (episode_efe_means.max() - episode_efe_means.min() + 1e-6)
            
            # Combined normalized energy (sum of normalized values)
            combined_energy = pd.Series(vfe_normalized + efe_normalized, index=episode_vfe_means.index)
            episode_ids_sorted = sorted(combined_energy.index)
            combined_sorted = [combined_energy[ep_id] for ep_id in episode_ids_sorted]
            
            ax9.plot(episode_ids_sorted, combined_sorted, 'darkgreen', alpha=0.7, linewidth=2, 
                    label='Combined Normalized VFE + EFE')
            if len(combined_sorted) > 5:
                window = min(10, len(combined_sorted)//3)
                rolling_mean = pd.Series(combined_sorted).rolling(window=window, min_periods=1).mean()
                ax9.plot(episode_ids_sorted, rolling_mean, 'red', linewidth=3, 
                        label=f'Rolling Mean (window={window})')
            ax9.set_xlabel('Episode ID')
            ax9.set_ylabel('Combined Normalized Energy (0-2 scale)')
            ax9.set_title('Normalized Total Energy Minimization Trend', fontweight='bold')
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
        
        # Energy minimization trends - using normalized values
        if 'vfe' in self.metrics_data.columns:
            episode_vfe_means = self.metrics_data.groupby('episode_id')['vfe'].mean()
            # Normalize VFE values (0-1 scale) for trend analysis
            vfe_normalized = (episode_vfe_means.values - episode_vfe_means.min()) / (episode_vfe_means.max() - episode_vfe_means.min() + 1e-6)
            vfe_trend = np.polyfit(range(len(vfe_normalized)), vfe_normalized, 1)[0]
            self.report_lines.append(f"\n### Energy Minimization Trends (Normalized):")
            self.report_lines.append(f"- **VFE Trend**: {vfe_trend:.4f} per episode ({'Improving' if vfe_trend < 0 else 'Worsening'})")
        
        if 'efe' in self.metrics_data.columns:
            episode_efe_means = self.metrics_data.groupby('episode_id')['efe'].mean()
            # Normalize EFE values (0-1 scale) for trend analysis
            efe_normalized = (episode_efe_means.values - episode_efe_means.min()) / (episode_efe_means.max() - episode_efe_means.min() + 1e-6)
            efe_trend = np.polyfit(range(len(efe_normalized)), efe_normalized, 1)[0]
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
    
    def analyze_computational_cost(self):
        """Analyze computational cost and efficiency metrics"""
        if self.episode_data is None:
            print("[WARNING] No episode data available for computational cost analysis")
            return
            
        # Import required modules for ANOVA
        from scipy.stats import f_oneway
        
        print(">> Analyzing computational cost and efficiency...")
        
        # Create comprehensive computational cost analysis
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Computational Cost and Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Planning Time Distribution
        if 'avg_planning_time_ms' in self.episode_data.columns:
            axes[0,0].hist(self.episode_data['avg_planning_time_ms'].dropna(), 
                          bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,0].set_title('Planning Time Distribution')
            axes[0,0].set_xlabel('Average Planning Time (ms)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].axvline(self.episode_data['avg_planning_time_ms'].mean(), 
                             color='red', linestyle='--', label=f'Mean: {self.episode_data["avg_planning_time_ms"].mean():.2f}ms')
            axes[0,0].legend()
        
        # 2. Episode Duration vs Performance
        if 'duration_seconds' in self.episode_data.columns and 'efficiency_metric' in self.episode_data.columns:
            success_mask = self.episode_data['status'] == 'success'
            axes[0,1].scatter(self.episode_data.loc[success_mask, 'duration_seconds'], 
                            self.episode_data.loc[success_mask, 'efficiency_metric'],
                            alpha=0.6, label='Success', color='green')
            axes[0,1].scatter(self.episode_data.loc[~success_mask, 'duration_seconds'], 
                            self.episode_data.loc[~success_mask, 'efficiency_metric'],
                            alpha=0.6, label='Failure', color='red', marker='x')
            axes[0,1].set_title('Duration vs Efficiency')
            axes[0,1].set_xlabel('Duration (seconds)')
            axes[0,1].set_ylabel('Efficiency Metric')
            axes[0,1].legend()
        
        # 3. Steps vs Planning Time
        if 'steps_taken' in self.episode_data.columns and 'avg_planning_time_ms' in self.episode_data.columns:
            axes[0,2].scatter(self.episode_data['steps_taken'], 
                            self.episode_data['avg_planning_time_ms'], alpha=0.6)
            axes[0,2].set_title('Steps vs Planning Time')
            axes[0,2].set_xlabel('Steps Taken')
            axes[0,2].set_ylabel('Avg Planning Time (ms)')
            
            # Add correlation
            corr, p_val = pearsonr(self.episode_data['steps_taken'].dropna(), 
                                  self.episode_data['avg_planning_time_ms'].dropna())
            axes[0,2].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                          transform=axes[0,2].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # 4. Replanning Cost Analysis
        if 'replanning_count' in self.episode_data.columns:
            replanning_costs = self.episode_data.groupby('replanning_count').agg({
                'avg_planning_time_ms': 'mean',
                'duration_seconds': 'mean',
                'efficiency_metric': 'mean'
            }).reset_index()
            
            axes[0,3].bar(replanning_costs['replanning_count'], 
                         replanning_costs['avg_planning_time_ms'], alpha=0.7)
            axes[0,3].set_title('Replanning vs Planning Time')
            axes[0,3].set_xlabel('Replanning Count')
            axes[0,3].set_ylabel('Avg Planning Time (ms)')
        
        # 5. Computational Cost by Success/Failure
        cost_metrics = ['avg_planning_time_ms', 'duration_seconds', 'steps_taken', 'replanning_count']
        available_metrics = [m for m in cost_metrics if m in self.episode_data.columns]
        
        if available_metrics:
            success_data = self.episode_data[self.episode_data['status'] == 'success'][available_metrics]
            failure_data = self.episode_data[self.episode_data['status'] != 'success'][available_metrics]
            
            for i, metric in enumerate(available_metrics[:4]):
                row = 1 + i // 2
                col = i % 2
                if row < 3:
                    axes[row, col].boxplot([success_data[metric].dropna(), failure_data[metric].dropna()], 
                                          labels=['Success', 'Failure'])
                    axes[row, col].set_title(f'{metric.replace("_", " ").title()} by Outcome')
                    axes[row, col].set_ylabel(metric.replace("_", " ").title())
        
        # 6. Efficiency Trends Over Time
        if 'episode_id' in self.episode_data.columns and 'avg_step_time' in self.episode_data.columns:
            # Rolling average efficiency
            window_size = min(10, len(self.episode_data) // 5)
            if window_size > 1:
                rolling_efficiency = self.episode_data['avg_step_time'].rolling(window=window_size).mean()
                axes[2,0].plot(self.episode_data['episode_id'], rolling_efficiency, 
                              linewidth=2, label=f'Rolling Average (window={window_size})')
                axes[2,0].scatter(self.episode_data['episode_id'], self.episode_data['avg_step_time'], 
                                 alpha=0.3, s=20)
                axes[2,0].set_title('Step Time Efficiency Over Episodes')
                axes[2,0].set_xlabel('Episode ID')
                axes[2,0].set_ylabel('Average Step Time (s)')
                axes[2,0].legend()
        
        # 7. Planning Efficiency vs Success Rate
        if len(available_metrics) >= 2:
            # Bin episodes by planning time and calculate success rate
            planning_bins = pd.qcut(self.episode_data['avg_planning_time_ms'].dropna(), 
                                   q=5, duplicates='drop')
            binned_data = self.episode_data.groupby(planning_bins).agg({
                'status': lambda x: (x == 'success').mean(),
                'efficiency_metric': 'mean'
            }).reset_index()
            
            if len(binned_data) > 1:
                axes[2,1].plot(range(len(binned_data)), binned_data['status'], 
                              marker='o', linewidth=2, label='Success Rate')
                axes[2,1].set_title('Success Rate vs Planning Time Bins')
                axes[2,1].set_xlabel('Planning Time Quintile (Low → High)')
                axes[2,1].set_ylabel('Success Rate')
                axes[2,1].set_ylim(0, 1)
                axes[2,1].legend()
        
        # 8. Resource Utilization Heatmap
        if self.metrics_data is not None and 'planning_time_ms' in self.metrics_data.columns:
            # Aggregate step-level planning times by episode
            step_planning = self.metrics_data.groupby('episode_id')['planning_time_ms'].agg(['mean', 'std', 'max']).reset_index()
            step_planning = step_planning.merge(self.episode_data[['episode_id', 'status']], on='episode_id')
            
            # Create correlation matrix for computational metrics
            computational_cols = ['mean', 'std', 'max']
            if len(step_planning) > 10:
                corr_matrix = step_planning[computational_cols].corr()
                im = axes[2,2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[2,2].set_xticks(range(len(computational_cols)))
                axes[2,2].set_yticks(range(len(computational_cols)))
                axes[2,2].set_xticklabels(['Mean Time', 'Std Time', 'Max Time'])
                axes[2,2].set_yticklabels(['Mean Time', 'Std Time', 'Max Time'])
                axes[2,2].set_title('Planning Time Metrics Correlation')
                
                # Add correlation values
                for i in range(len(computational_cols)):
                    for j in range(len(computational_cols)):
                        axes[2,2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                      ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        # 9. Cost-Benefit Analysis
        if 'efficiency_metric' in self.episode_data.columns and 'avg_planning_time_ms' in self.episode_data.columns:
            # Calculate cost-benefit ratio (efficiency per unit planning time)
            cost_benefit = self.episode_data['efficiency_metric'] / (self.episode_data['avg_planning_time_ms'] + 1e-6)
            
            axes[2,3].hist(cost_benefit.dropna(), bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[2,3].set_title('Cost-Benefit Ratio Distribution')
            axes[2,3].set_xlabel('Efficiency per Planning Time Unit')
            axes[2,3].set_ylabel('Frequency')
            axes[2,3].axvline(cost_benefit.mean(), color='red', linestyle='--', 
                             label=f'Mean: {cost_benefit.mean():.4f}')
            axes[2,3].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'computational_cost_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate computational cost statistics
        cost_stats = {
            'avg_planning_time_mean': self.episode_data['avg_planning_time_ms'].mean() if 'avg_planning_time_ms' in self.episode_data.columns else None,
            'avg_planning_time_std': self.episode_data['avg_planning_time_ms'].std() if 'avg_planning_time_ms' in self.episode_data.columns else None,
            'duration_mean': self.episode_data['duration_seconds'].mean() if 'duration_seconds' in self.episode_data.columns else None,
            'duration_std': self.episode_data['duration_seconds'].std() if 'duration_seconds' in self.episode_data.columns else None,
            'steps_mean': self.episode_data['steps_taken'].mean() if 'steps_taken' in self.episode_data.columns else None,
            'steps_std': self.episode_data['steps_taken'].std() if 'steps_taken' in self.episode_data.columns else None,
        }
        
        print(f"[METRICS] Average planning time: {cost_stats['avg_planning_time_mean']:.2f} ± {cost_stats['avg_planning_time_std']:.2f} ms")
        print(f"[METRICS] Average episode duration: {cost_stats['duration_mean']:.2f} ± {cost_stats['duration_std']:.2f} seconds")
        print(f"[METRICS] Average steps taken: {cost_stats['steps_mean']:.2f} ± {cost_stats['steps_std']:.2f}")
        
        return cost_stats

    def perform_anova_testing(self):
        """Perform ANOVA testing for statistical significance between groups"""
        if self.episode_data is None:
            print("[WARNING] No episode data available for ANOVA testing")
            return
            
        from scipy.stats import f_oneway, ttest_ind, chi2_contingency
        from scipy.stats import levene, normaltest
        
        print(">> Performing ANOVA and statistical significance testing...")
        
        # Define groups for analysis
        success_group = self.episode_data[self.episode_data['status'] == 'success']
        failure_group = self.episode_data[self.episode_data['status'] != 'success']
        
        # Test metrics for group differences
        test_metrics = [
            'steps_taken', 'duration_seconds', 'avg_planning_time_ms', 'efficiency_metric',
            'avg_vfe', 'avg_efe', 'replanning_count', 'distance_improvement_percentage'
        ]
        
        anova_results = {}
        
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTING RESULTS")
        print("="*60)
        
        for metric in test_metrics:
            if metric in self.episode_data.columns:
                success_values = success_group[metric].dropna()
                failure_values = failure_group[metric].dropna()
                
                if len(success_values) > 2 and len(failure_values) > 2:
                    # Perform t-test
                    t_stat, t_p_value = ttest_ind(success_values, failure_values)
                    
                    # Perform F-test (ANOVA)
                    f_stat, f_p_value = f_oneway(success_values, failure_values)
                    
                    # Test for equal variances (Levene's test)
                    levene_stat, levene_p = levene(success_values, failure_values)
                    
                    # Test for normality
                    success_normal_stat, success_normal_p = normaltest(success_values)
                    failure_normal_stat, failure_normal_p = normaltest(failure_values)
                    
                    anova_results[metric] = {
                        'success_mean': success_values.mean(),
                        'success_std': success_values.std(),
                        'failure_mean': failure_values.mean(),
                        'failure_std': failure_values.std(),
                        't_statistic': t_stat,
                        't_p_value': t_p_value,
                        'f_statistic': f_stat,
                        'f_p_value': f_p_value,
                        'levene_statistic': levene_stat,
                        'levene_p_value': levene_p,
                        'success_normality_p': success_normal_p,
                        'failure_normality_p': failure_normal_p,
                        'significant': f_p_value < 0.05
                    }
                    
                    # Print results
                    print(f"\n{metric.upper().replace('_', ' ')}:")
                    print(f"  Success: {success_values.mean():.3f} ± {success_values.std():.3f} (n={len(success_values)})")
                    print(f"  Failure: {failure_values.mean():.3f} ± {failure_values.std():.3f} (n={len(failure_values)})")
                    print(f"  t-test: t={t_stat:.3f}, p={t_p_value:.6f}")
                    print(f"  ANOVA: F={f_stat:.3f}, p={f_p_value:.6f}")
                    print(f"  Equal variances (Levene): p={levene_p:.6f}")
                    print(f"  Normality - Success: p={success_normal_p:.6f}, Failure: p={failure_normal_p:.6f}")
                    print(f"  {'*** SIGNIFICANT ***' if f_p_value < 0.05 else 'Not significant'}")
        
        # Multi-factor ANOVA for complex interactions
        print(f"\n{'='*60}")
        print("MULTI-FACTOR ANALYSIS")
        print("="*60)
        
        # Group by multiple factors if available
        if 'replanning_count' in self.episode_data.columns:
            # Create replanning groups
            self.episode_data['replanning_group'] = pd.cut(self.episode_data['replanning_count'], 
                                                          bins=3, labels=['Low', 'Medium', 'High'])
            
            # Test efficiency across replanning groups
            if 'efficiency_metric' in self.episode_data.columns:
                low_replan = self.episode_data[self.episode_data['replanning_group'] == 'Low']['efficiency_metric'].dropna()
                med_replan = self.episode_data[self.episode_data['replanning_group'] == 'Medium']['efficiency_metric'].dropna()
                high_replan = self.episode_data[self.episode_data['replanning_group'] == 'High']['efficiency_metric'].dropna()
                
                if len(low_replan) > 2 and len(med_replan) > 2 and len(high_replan) > 2:
                    f_stat, f_p = f_oneway(low_replan, med_replan, high_replan)
                    print(f"\nEfficiency by Replanning Groups:")
                    print(f"  Low: {low_replan.mean():.3f} ± {low_replan.std():.3f}")
                    print(f"  Medium: {med_replan.mean():.3f} ± {med_replan.std():.3f}")
                    print(f"  High: {high_replan.mean():.3f} ± {high_replan.std():.3f}")
                    print(f"  ANOVA: F={f_stat:.3f}, p={f_p:.6f}")
                    print(f"  {'*** SIGNIFICANT ***' if f_p < 0.05 else 'Not significant'}")
        
        # Create visualization of statistical results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Significance Testing Results', fontsize=16, fontweight='bold')
        
        # 1. P-value comparison
        significant_metrics = [k for k, v in anova_results.items() if v['significant']]
        non_significant_metrics = [k for k, v in anova_results.items() if not v['significant']]
        
        p_values = [anova_results[k]['f_p_value'] for k in anova_results.keys()]
        metric_names = list(anova_results.keys())
        
        colors = ['red' if anova_results[k]['significant'] else 'blue' for k in metric_names]
        
        axes[0,0].barh(range(len(metric_names)), p_values, color=colors, alpha=0.7)
        axes[0,0].set_yticks(range(len(metric_names)))
        axes[0,0].set_yticklabels([m.replace('_', ' ').title() for m in metric_names])
        axes[0,0].axvline(0.05, color='black', linestyle='--', label='α = 0.05')
        axes[0,0].set_xlabel('P-value')
        axes[0,0].set_title('ANOVA P-values by Metric')
        axes[0,0].legend()
        axes[0,0].set_xscale('log')
        
        # 2. Effect sizes (mean differences)
        effect_sizes = []
        for metric in anova_results.keys():
            success_mean = anova_results[metric]['success_mean']
            failure_mean = anova_results[metric]['failure_mean']
            pooled_std = np.sqrt((anova_results[metric]['success_std']**2 + anova_results[metric]['failure_std']**2) / 2)
            effect_size = abs(success_mean - failure_mean) / (pooled_std + 1e-6)
            effect_sizes.append(effect_size)
        
        axes[0,1].barh(range(len(metric_names)), effect_sizes, 
                      color=['red' if anova_results[k]['significant'] else 'blue' for k in metric_names], alpha=0.7)
        axes[0,1].set_yticks(range(len(metric_names)))
        axes[0,1].set_yticklabels([m.replace('_', ' ').title() for m in metric_names])
        axes[0,1].set_xlabel('Cohen\'s d (Effect Size)')
        axes[0,1].set_title('Effect Sizes by Metric')
        
        # 3. Success vs Failure means comparison
        success_means = [anova_results[k]['success_mean'] for k in metric_names]
        failure_means = [anova_results[k]['failure_mean'] for k in metric_names]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[1,0].bar(x - width/2, success_means, width, label='Success', alpha=0.7, color='green')
        axes[1,0].bar(x + width/2, failure_means, width, label='Failure', alpha=0.7, color='red')
        axes[1,0].set_xlabel('Metrics')
        axes[1,0].set_ylabel('Mean Value')
        axes[1,0].set_title('Mean Values: Success vs Failure')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels([m.replace('_', ' ')[:8] for m in metric_names], rotation=45)
        axes[1,0].legend()
        
        # 4. Normality test results
        normality_results = []
        for metric in metric_names:
            success_normal = anova_results[metric]['success_normality_p']
            failure_normal = anova_results[metric]['failure_normality_p']
            normality_results.append([success_normal, failure_normal])
        
        normality_array = np.array(normality_results)
        im = axes[1,1].imshow(normality_array.T, cmap='RdYlBu', vmin=0, vmax=0.1)
        axes[1,1].set_yticks([0, 1])
        axes[1,1].set_yticklabels(['Success', 'Failure'])
        axes[1,1].set_xticks(range(len(metric_names)))
        axes[1,1].set_xticklabels([m.replace('_', ' ')[:8] for m in metric_names], rotation=45)
        axes[1,1].set_title('Normality Test P-values')
        plt.colorbar(im, ax=axes[1,1], label='P-value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'anova_statistical_testing.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(significant_metrics)} out of {len(anova_results)} metrics show significant differences")
        print(f"Significant metrics: {', '.join(significant_metrics)}")
        print("="*60)
        
        return anova_results

    def investigate_vfe_efe_correlation(self):
        """Deep dive investigation into the strong VFE-EFE correlation"""
        if self.episode_data is None or 'avg_vfe' not in self.episode_data.columns or 'avg_efe' not in self.episode_data.columns:
            print("[WARNING] No VFE/EFE data available for correlation investigation")
            return
            
        print(">> Investigating VFE-EFE correlation in depth...")
        
        # Calculate correlations
        vfe_efe_corr, vfe_efe_p = pearsonr(self.episode_data['avg_vfe'].dropna(), 
                                          self.episode_data['avg_efe'].dropna())
        
        print(f"\n[CORRELATION] VFE-EFE Pearson correlation: r = {vfe_efe_corr:.6f}, p = {vfe_efe_p:.6f}")
        
        # Create comprehensive correlation investigation
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('VFE-EFE Correlation Investigation', fontsize=16, fontweight='bold')
        
        # 1. Basic scatter plot with regression line
        axes[0,0].scatter(self.episode_data['avg_vfe'], self.episode_data['avg_efe'], 
                         alpha=0.6, s=50)
        
        # Add regression line
        z = np.polyfit(self.episode_data['avg_vfe'].dropna(), self.episode_data['avg_efe'].dropna(), 1)
        p = np.poly1d(z)
        axes[0,0].plot(self.episode_data['avg_vfe'], p(self.episode_data['avg_vfe']), 
                      "r--", alpha=0.8, linewidth=2)
        axes[0,0].set_xlabel('Average VFE')
        axes[0,0].set_ylabel('Average EFE')
        axes[0,0].set_title(f'VFE vs EFE (r = {vfe_efe_corr:.6f})')
        axes[0,0].text(0.05, 0.95, f'y = {z[0]:.3f}x + {z[1]:.3f}', 
                      transform=axes[0,0].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # 2. Residuals analysis
        predicted_efe = p(self.episode_data['avg_vfe'])
        residuals = self.episode_data['avg_efe'] - predicted_efe
        
        axes[0,1].scatter(predicted_efe, residuals, alpha=0.6)
        axes[0,1].axhline(y=0, color='red', linestyle='--')
        axes[0,1].set_xlabel('Predicted EFE')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals vs Predicted EFE')
        
        # 3. VFE-EFE ratio analysis
        vfe_efe_ratio = self.episode_data['avg_efe'] / (self.episode_data['avg_vfe'] + 1e-6)
        axes[0,2].hist(vfe_efe_ratio.dropna(), bins=20, alpha=0.7, edgecolor='black')
        axes[0,2].set_xlabel('EFE/VFE Ratio')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('EFE/VFE Ratio Distribution')
        axes[0,2].axvline(vfe_efe_ratio.mean(), color='red', linestyle='--', 
                         label=f'Mean: {vfe_efe_ratio.mean():.3f}')
        axes[0,2].legend()
        
        # 4. VFE-EFE gap analysis
        if 'efe_vs_vfe_gap' in self.episode_data.columns:
            gap_data = self.episode_data['efe_vs_vfe_gap'].dropna()
            axes[0,3].hist(gap_data, bins=20, alpha=0.7, edgecolor='black')
            axes[0,3].set_xlabel('EFE-VFE Gap')
            axes[0,3].set_ylabel('Frequency')
            axes[0,3].set_title('EFE-VFE Gap Distribution')
        else:
            # Calculate gap manually
            vfe_efe_gap = self.episode_data['avg_efe'] - self.episode_data['avg_vfe']
            axes[0,3].hist(vfe_efe_gap.dropna(), bins=20, alpha=0.7, edgecolor='black')
            axes[0,3].set_xlabel('EFE-VFE Gap')
            axes[0,3].set_ylabel('Frequency')
            axes[0,3].set_title('EFE-VFE Gap Distribution')
        
        # 5. Temporal correlation analysis
        if 'episode_id' in self.episode_data.columns:
            # Rolling correlation
            window_size = min(20, len(self.episode_data) // 3)
            if window_size > 5:
                rolling_corr = []
                for i in range(window_size, len(self.episode_data)):
                    window_data = self.episode_data.iloc[i-window_size:i]
                    if len(window_data) > 3:
                        corr, _ = pearsonr(window_data['avg_vfe'].dropna(), window_data['avg_efe'].dropna())
                        rolling_corr.append(corr)
                
                axes[1,0].plot(range(window_size, window_size + len(rolling_corr)), rolling_corr, 
                              linewidth=2, label=f'Rolling correlation (window={window_size})')
                axes[1,0].axhline(y=vfe_efe_corr, color='red', linestyle='--', label='Overall correlation')
                axes[1,0].set_xlabel('Episode ID')
                axes[1,0].set_ylabel('Correlation Coefficient')
                axes[1,0].set_title('Temporal VFE-EFE Correlation')
                axes[1,0].legend()
                axes[1,0].set_ylim(-1.1, 1.1)
        
        # 6. Success/Failure impact on correlation
        success_mask = self.episode_data['status'] == 'success'
        if success_mask.sum() > 3 and (~success_mask).sum() > 3:
            success_corr, _ = pearsonr(self.episode_data.loc[success_mask, 'avg_vfe'].dropna(), 
                                     self.episode_data.loc[success_mask, 'avg_efe'].dropna())
            failure_corr, _ = pearsonr(self.episode_data.loc[~success_mask, 'avg_vfe'].dropna(), 
                                     self.episode_data.loc[~success_mask, 'avg_efe'].dropna())
            
            axes[1,1].bar(['Success', 'Failure', 'Overall'], 
                         [success_corr, failure_corr, vfe_efe_corr], 
                         color=['green', 'red', 'blue'], alpha=0.7)
            axes[1,1].set_ylabel('Correlation Coefficient')
            axes[1,1].set_title('VFE-EFE Correlation by Outcome')
            axes[1,1].set_ylim(-1.1, 1.1)
            
            # Add correlation values on bars
            for i, v in enumerate([success_corr, failure_corr, vfe_efe_corr]):
                axes[1,1].text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.3f}', ha='center', va='bottom')
        
        # 7. Step-level correlation analysis if available
        if self.metrics_data is not None and 'vfe' in self.metrics_data.columns and 'efe' in self.metrics_data.columns:
            step_corr, step_p = pearsonr(self.metrics_data['vfe'].dropna(), self.metrics_data['efe'].dropna())
            
            # Sample data for visualization (to avoid overcrowding)
            sample_size = min(1000, len(self.metrics_data))
            sample_data = self.metrics_data.sample(n=sample_size)
            
            axes[1,2].scatter(sample_data['vfe'], sample_data['efe'], alpha=0.3, s=10)
            axes[1,2].set_xlabel('Step-level VFE')
            axes[1,2].set_ylabel('Step-level EFE')
            axes[1,2].set_title(f'Step-level VFE-EFE (r = {step_corr:.6f})')
            
            print(f"[CORRELATION] Step-level VFE-EFE correlation: r = {step_corr:.6f}, p = {step_p:.6f}")
        
        # 8. Theoretical analysis plot
        axes[1,3].text(0.1, 0.9, 'THEORETICAL ANALYSIS', fontweight='bold', fontsize=12, transform=axes[1,3].transAxes)
        
        theoretical_text = f"""
Strong Correlation Analysis:
• Correlation: r = {vfe_efe_corr:.6f}
• P-value: {vfe_efe_p:.6f}

Possible Explanations:
1. Theoretical coupling:
   EFE = E[VFE] - E[log π]
   
2. Active inference principle:
   Action minimizes both VFE 
   and EFE simultaneously
   
3. Implementation coupling:
   Shared computational basis
   
4. Environment constraints:
   Limited action space creates
   natural VFE-EFE relationship
        """
        
        axes[1,3].text(0.05, 0.75, theoretical_text, fontsize=10, 
                      transform=axes[1,3].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        axes[1,3].set_xlim(0, 1)
        axes[1,3].set_ylim(0, 1)
        axes[1,3].axis('off')
        
        # 9. Energy minimization trajectory
        axes[2,0].plot(self.episode_data['episode_id'], self.episode_data['avg_vfe'], 
                      label='VFE', linewidth=2, alpha=0.8)
        axes[2,0].plot(self.episode_data['episode_id'], self.episode_data['avg_efe'], 
                      label='EFE', linewidth=2, alpha=0.8)
        axes[2,0].set_xlabel('Episode ID')
        axes[2,0].set_ylabel('Energy Value')
        axes[2,0].set_title('Energy Minimization Trajectories')
        axes[2,0].legend()
        
        # 10. Correlation stability analysis
        if len(self.episode_data) > 20:
            bootstrap_correlations = []
            n_bootstrap = 100
            np.random.seed(42)
            
            for _ in range(n_bootstrap):
                sample_indices = np.random.choice(len(self.episode_data), size=len(self.episode_data), replace=True)
                sample_data = self.episode_data.iloc[sample_indices]
                if len(sample_data) > 3:
                    boot_corr, _ = pearsonr(sample_data['avg_vfe'].dropna(), sample_data['avg_efe'].dropna())
                    bootstrap_correlations.append(boot_corr)
            
            axes[2,1].hist(bootstrap_correlations, bins=20, alpha=0.7, edgecolor='black')
            axes[2,1].axvline(vfe_efe_corr, color='red', linestyle='--', linewidth=2, label='Observed')
            axes[2,1].set_xlabel('Bootstrap Correlation')
            axes[2,1].set_ylabel('Frequency')
            axes[2,1].set_title('Bootstrap Correlation Distribution')
            axes[2,1].legend()
            
            print(f"[BOOTSTRAP] Correlation 95% CI: [{np.percentile(bootstrap_correlations, 2.5):.3f}, {np.percentile(bootstrap_correlations, 97.5):.3f}]")
        
        # 11. Causal direction analysis
        if len(self.episode_data) > 10:
            # Granger causality approximation using lagged correlations
            lag_correlations_vfe_to_efe = []
            lag_correlations_efe_to_vfe = []
            
            for lag in range(1, min(10, len(self.episode_data)//3)):
                if lag < len(self.episode_data):
                    vfe_lagged = self.episode_data['avg_vfe'].iloc[:-lag]
                    efe_current = self.episode_data['avg_efe'].iloc[lag:]
                    
                    efe_lagged = self.episode_data['avg_efe'].iloc[:-lag]
                    vfe_current = self.episode_data['avg_vfe'].iloc[lag:]
                    
                    if len(vfe_lagged) > 3:
                        corr_vfe_to_efe, _ = pearsonr(vfe_lagged, efe_current)
                        corr_efe_to_vfe, _ = pearsonr(efe_lagged, vfe_current)
                        
                        lag_correlations_vfe_to_efe.append(abs(corr_vfe_to_efe))
                        lag_correlations_efe_to_vfe.append(abs(corr_efe_to_vfe))
            
            if lag_correlations_vfe_to_efe:            lags = range(1, len(lag_correlations_vfe_to_efe) + 1)
            axes[2,2].plot(lags, lag_correlations_vfe_to_efe, 'o-', label='VFE → EFE', linewidth=2)
            axes[2,2].plot(lags, lag_correlations_efe_to_vfe, 's-', label='EFE → VFE', linewidth=2)
            axes[2,2].set_xlabel('Lag (episodes)')
            axes[2,2].set_ylabel('|Correlation|')
            axes[2,2].set_title('Lagged Cross-Correlations')
            axes[2,2].legend()
        
        # 12. Summary statistics
        summary_text = f"""
CORRELATION INVESTIGATION SUMMARY:

Primary Correlation:
• Pearson r = {vfe_efe_corr:.6f}
• P-value = {vfe_efe_p:.6f}
• Relationship: {'Perfect' if abs(vfe_efe_corr) > 0.99 else 'Very Strong' if abs(vfe_efe_corr) > 0.9 else 'Strong' if abs(vfe_efe_corr) > 0.7 else 'Moderate'}

Key Findings:
• VFE Range: [{self.episode_data['avg_vfe'].min():.1f}, {self.episode_data['avg_vfe'].max():.1f}]
• EFE Range: [{self.episode_data['avg_efe'].min():.1f}, {self.episode_data['avg_efe'].max():.1f}]
• Mean Ratio: {vfe_efe_ratio.mean():.3f}

Implications:
• Strong theoretical coupling
• Consistent energy minimization
• Predictable EFE from VFE        """
        
        axes[2,3].text(0.05, 0.95, summary_text, fontsize=10,
                      transform=axes[2,3].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        axes[2,3].set_xlim(0, 1)
        axes[2,3].set_ylim(0, 1)
        axes[2,3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'vfe_efe_correlation_investigation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return correlation analysis results
        correlation_results = {
            'primary_correlation': vfe_efe_corr,
            'primary_p_value': vfe_efe_p,
            'vfe_range': [self.episode_data['avg_vfe'].min(), self.episode_data['avg_vfe'].max()],
            'efe_range': [self.episode_data['avg_efe'].min(), self.episode_data['avg_efe'].max()],
            'mean_ratio': vfe_efe_ratio.mean(),
            'ratio_std': vfe_efe_ratio.std()
        }
        
        print(f"\n[ANALYSIS] Perfect correlation detected: r = {vfe_efe_corr:.6f}")
        print(f"[ANALYSIS] This suggests strong theoretical/computational coupling between VFE and EFE")
        
        return correlation_results
    
    def generate_summary_report(self):
        """Generate a comprehensive text report"""
        
        # Add sections for new analyses
        self.report_lines.append("\n## Computational Cost Analysis")
        self.report_lines.append("*Detailed computational efficiency metrics and cost-benefit analysis.*")
        
        # Add computational cost metrics
        if 'avg_planning_time_ms' in self.episode_data.columns:
            avg_planning = self.episode_data['avg_planning_time_ms'].mean()
            std_planning = self.episode_data['avg_planning_time_ms'].std()
            self.report_lines.append(f"- **Average Planning Time**: {avg_planning:.2f} ± {std_planning:.2f} ms")
        
        if 'duration_seconds' in self.episode_data.columns:
            avg_duration = self.episode_data['duration_seconds'].mean()
            std_duration = self.episode_data['duration_seconds'].std()
            self.report_lines.append(f"- **Average Episode Duration**: {avg_duration:.2f} ± {std_duration:.2f} seconds")
        
        if 'avg_step_time' in self.episode_data.columns:
            avg_step_time = self.episode_data['avg_step_time'].mean()
            self.report_lines.append(f"- **Average Step Time**: {avg_step_time:.2f} seconds per step")
        
        # Add efficiency analysis
        if 'efficiency_metric' in self.episode_data.columns and 'avg_planning_time_ms' in self.episode_data.columns:
            cost_benefit = self.episode_data['efficiency_metric'] / (self.episode_data['avg_planning_time_ms'] + 1e-6)
            self.report_lines.append(f"- **Cost-Benefit Ratio**: {cost_benefit.mean():.4f} (efficiency per planning ms)")
        
        self.report_lines.append("\n## Statistical Significance Testing (ANOVA)")
        self.report_lines.append("*Comprehensive statistical testing between successful and failed episodes.*")
        
        # Add placeholder for ANOVA results (will be populated by perform_anova_testing)
        if hasattr(self, 'anova_results'):
            significant_count = sum(1 for v in self.anova_results.values() if v.get('significant', False))
            total_tests = len(self.anova_results)
            self.report_lines.append(f"- **Significant Differences Found**: {significant_count} out of {total_tests} metrics tested")
            
            # List significant metrics
            significant_metrics = [k for k, v in self.anova_results.items() if v.get('significant', False)]
            if significant_metrics:
                self.report_lines.append(f"- **Significant Metrics**: {', '.join(significant_metrics)}")
            
            # Add effect sizes for significant metrics
            for metric in significant_metrics[:5]:  # Top 5 significant metrics
                if metric in self.anova_results:
                    result = self.anova_results[metric]
                    success_mean = result.get('success_mean', 0)
                    failure_mean = result.get('failure_mean', 0)
                    p_value = result.get('f_p_value', 1)
                    self.report_lines.append(f"  - **{metric.replace('_', ' ').title()}**: Success={success_mean:.3f}, Failure={failure_mean:.3f}, p={p_value:.6f}")
        
        self.report_lines.append("\n## VFE-EFE Correlation Investigation")
        self.report_lines.append("*Deep analysis of the relationship between Variational Free Energy and Expected Free Energy.*")
        
        # Add VFE-EFE correlation details
        if 'avg_vfe' in self.episode_data.columns and 'avg_efe' in self.episode_data.columns:
            from scipy.stats import pearsonr
            vfe_efe_corr, vfe_efe_p = pearsonr(self.episode_data['avg_vfe'].dropna(), 
                                              self.episode_data['avg_efe'].dropna())
            
            self.report_lines.append(f"- **Primary Correlation**: r = {vfe_efe_corr:.6f}, p = {vfe_efe_p:.6f}")
            
            # Correlation strength interpretation
            if abs(vfe_efe_corr) > 0.99:
                strength = "Perfect"
            elif abs(vfe_efe_corr) > 0.9:
                strength = "Very Strong"
            elif abs(vfe_efe_corr) > 0.7:
                strength = "Strong"
            elif abs(vfe_efe_corr) > 0.5:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "negative" if vfe_efe_corr < 0 else "positive"
            self.report_lines.append(f"- **Relationship Strength**: {strength} {direction} correlation")
            
            # Theoretical implications
            self.report_lines.append("- **Theoretical Implications**:")
            self.report_lines.append("  - Strong correlation suggests tight coupling between perception and action")
            self.report_lines.append("  - Consistent with Active Inference principle of energy minimization")
            self.report_lines.append("  - May indicate shared computational basis or theoretical constraint")
            
            # VFE and EFE ranges
            vfe_range = [self.episode_data['avg_vfe'].min(), self.episode_data['avg_vfe'].max()]
            efe_range = [self.episode_data['avg_efe'].min(), self.episode_data['avg_efe'].max()]
            self.report_lines.append(f"- **VFE Range**: [{vfe_range[0]:.1f}, {vfe_range[1]:.1f}]")
            self.report_lines.append(f"- **EFE Range**: [{efe_range[0]:.1f}, {efe_range[1]:.1f}]")
        
        self.report_lines.append("\n## Generated Visualizations")
        self.report_lines.append("*This analysis produced the following visualization files:*")
        self.report_lines.append("- **performance_dashboard.png**: 6-panel performance overview")
        self.report_lines.append("- **enhanced_vfe_efe_dynamics.png**: 12-panel VFE/EFE analysis with log normalization")
        self.report_lines.append("- **correlation_matrix.png**: Behavioral pattern correlation heatmap")
        self.report_lines.append("- **computational_cost_analysis.png**: 12-panel computational efficiency analysis")
        self.report_lines.append("- **anova_statistical_testing.png**: Statistical significance testing results")
        self.report_lines.append("- **vfe_efe_correlation_investigation.png**: 12-panel correlation deep dive")
        self.report_lines.append("- **planning_analysis.png**: Planning behavior distributions")
        self.report_lines.append("- **log_efe_trajectories.png**: Log-normalized EFE trajectories")
        self.report_lines.append("- **log_vfe_trajectories.png**: Log-normalized VFE trajectories")
        self.report_lines.append("- **log_vfe_efe_scatter.png**: Log VFE vs EFE scatter plot")
        self.report_lines.append("- **planning_time_distribution.png**: Planning time distribution")
        self.report_lines.append("- **planning_time_by_outcome.png**: Planning time by episode outcome")
        
        self.report_lines.append("\n## Methodology Summary")
        self.report_lines.append("*Statistical methods and analysis approaches used:*")
        self.report_lines.append("- **Descriptive Statistics**: Mean, standard deviation, median for all metrics")
        self.report_lines.append("- **Correlation Analysis**: Pearson correlation for linear relationships")
        self.report_lines.append("- **ANOVA Testing**: F-tests for group differences between success/failure")
        self.report_lines.append("- **Effect Size Analysis**: Cohen's d for practical significance")
        self.report_lines.append("- **Normality Testing**: D'Agostino normality tests for distribution assumptions")
        self.report_lines.append("- **Variance Testing**: Levene's test for equal variances")
        self.report_lines.append("- **Logarithmic Normalization**: log(|value| + 1e-6) for VFE/EFE analysis")
        self.report_lines.append("- **Bootstrap Analysis**: Correlation stability testing")
        self.report_lines.append("- **Temporal Analysis**: Time series trends and rolling averages")
        
        # Data quality summary
        self.report_lines.append("\n## Data Quality Summary")
        episode_count = len(self.episode_data) if self.episode_data is not None else 0
        metrics_count = len(self.metrics_data) if self.metrics_data is not None else 0
        
        self.report_lines.append(f"- **Episode Records**: {episode_count} episodes analyzed")
        self.report_lines.append(f"- **Step Records**: {metrics_count} step-level measurements")
        self.report_lines.append(f"- **Numeric Features**: {len(self.numeric_columns)} quantitative metrics")
        
        if self.episode_data is not None:
            missing_data_summary = []
            for col in self.episode_data.columns:
                missing_count = self.episode_data[col].isnull().sum()
                if missing_count > 0:
                    missing_pct = (missing_count / len(self.episode_data)) * 100
                    missing_data_summary.append(f"{col}: {missing_pct:.1f}%")
            
            if missing_data_summary:
                self.report_lines.append(f"- **Missing Data**: {', '.join(missing_data_summary[:5])}")
            else:
                self.report_lines.append("- **Missing Data**: No missing values detected")
        
        self.report_lines.append("\n## Key Findings Summary")
        self.report_lines.append("*Primary insights from the comprehensive analysis:*")
        
        # Success rate summary
        if 'status' in self.episode_data.columns:
            success_rate = (self.episode_data['status'] == 'success').mean()
            self.report_lines.append(f"- **Overall Success Rate**: {success_rate:.1%}")
        
        # Performance trend
        if 'distance_improvement_percentage' in self.episode_data.columns:
            avg_improvement = self.episode_data['distance_improvement_percentage'].mean()
            self.report_lines.append(f"- **Average Distance Improvement**: {avg_improvement:.1f}%")
        
        # Computational efficiency
        if 'avg_planning_time_ms' in self.episode_data.columns and 'status' in self.episode_data.columns:
            success_planning = self.episode_data[self.episode_data['status'] == 'success']['avg_planning_time_ms'].mean()
            failure_planning = self.episode_data[self.episode_data['status'] != 'success']['avg_planning_time_ms'].mean()
            if not np.isnan(success_planning) and not np.isnan(failure_planning):
                planning_diff = ((success_planning - failure_planning) / failure_planning) * 100
                self.report_lines.append(f"- **Planning Efficiency**: Success episodes {planning_diff:+.1f}% planning time vs failures")
        
        # VFE-EFE relationship strength
        if 'avg_vfe' in self.episode_data.columns and 'avg_efe' in self.episode_data.columns:
            self.report_lines.append(f"- **Energy Coupling**: Strong correlation indicates tight VFE-EFE relationship")
        
        self.report_lines.append(f"\n---")
        self.report_lines.append(f"**Analysis completed**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report_lines.append(f"**Framework**: Single Environment Active Inference Analysis v2.0")
        
        # Save the comprehensive report
        report_path = os.path.join(RESULTS_DIR, 'single_environment_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        
        print(f"OK Comprehensive analysis report saved to {report_path}")
    
    def generate_paper_ready_visualizations(self):
        """Generate individual PNG files for paper submission"""
        print("Generating paper-ready visualizations...")
        
        # 1. Log EFE trajectories
        self.create_log_efe_trajectories()
        
        # 2. Log VFE trajectories  
        self.create_log_vfe_trajectories()
        
        # 3. Log EFE vs Distance trajectories
        self.create_log_efe_vs_distance_trajectories()
        
        # 4. Log EFE vs log VFE scatter plot
        self.create_log_vfe_efe_scatter()
        
        # 5. Planning time distribution
        self.create_planning_time_distribution()
        
        # 6. Average planning time ms distribution by outcome
        self.create_planning_time_by_outcome()
        
        print("Paper-ready visualizations completed!")
    
    def create_log_efe_trajectories(self):
        """Create standalone log EFE trajectories visualization"""
        if self.metrics_data is None or 'efe' not in self.metrics_data.columns:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Get episode data for classification
        episode_status = self.episode_data.set_index('episode_id')['status'].to_dict()
        all_episodes = sorted(self.metrics_data['episode_id'].unique())
        success_episodes = [ep for ep in all_episodes if episode_status.get(ep) == 'success']
        failure_episodes = [ep for ep in all_episodes if episode_status.get(ep) != 'success']
        
        # Create color schemes
        success_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(success_episodes))) if success_episodes else []
        failure_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(failure_episodes))) if failure_episodes else []
        
        # Plot successful episodes
        for i, ep_id in enumerate(success_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if len(ep_data) > 0:
                log_efe = np.log(np.abs(ep_data['efe']) + 1e-6)
                plt.plot(ep_data['step'], log_efe, alpha=0.6, 
                        color=success_colors[i], linewidth=1.2, label='Success' if i == 0 else "")
        
        # Plot failed episodes with dashed lines
        for i, ep_id in enumerate(failure_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if len(ep_data) > 0:
                log_efe = np.log(np.abs(ep_data['efe']) + 1e-6)
                plt.plot(ep_data['step'], log_efe, alpha=0.6, 
                        color=failure_colors[i], linewidth=1.2, linestyle='--', 
                        label='Failure' if i == 0 else "")
        
        # Add average trajectories
        if success_episodes:
            success_log_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in success_episodes])
            for step in range(max_steps):
                step_values = []
                for ep_id in success_episodes:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step:
                        step_values.append(np.log(np.abs(ep_data.iloc[step]['efe']) + 1e-6))
                if step_values:
                    success_log_avg.append(np.mean(step_values))
            if success_log_avg:
                plt.plot(range(len(success_log_avg)), success_log_avg, color='darkgreen', 
                        linewidth=3, label=f'Success Average (n={len(success_episodes)})', alpha=0.9)
        
        if failure_episodes:
            failure_log_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in failure_episodes])
            for step in range(max_steps):
                step_values = []
                for ep_id in failure_episodes:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step:
                        step_values.append(np.log(np.abs(ep_data.iloc[step]['efe']) + 1e-6))
                if step_values:
                    failure_log_avg.append(np.mean(step_values))
            if failure_log_avg:
                plt.plot(range(len(failure_log_avg)), failure_log_avg, color='darkred', 
                        linewidth=3, label=f'Failure Average (n={len(failure_episodes)})', alpha=0.9, linestyle='-.')
        
        plt.title('Log-Normalized Expected Free Energy (EFE) Trajectories', fontsize=16, fontweight='bold')
        plt.xlabel('Step', fontsize=14)
        plt.ylabel('log(|EFE| + 1e-6)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'log_efe_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created log_efe_trajectories.png")
    
    def create_log_vfe_trajectories(self):
        """Create standalone log VFE trajectories visualization"""
        if self.metrics_data is None or 'vfe' not in self.metrics_data.columns:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Get episode data for classification
        episode_status = self.episode_data.set_index('episode_id')['status'].to_dict()
        all_episodes = sorted(self.metrics_data['episode_id'].unique())
        success_episodes = [ep for ep in all_episodes if episode_status.get(ep) == 'success']
        failure_episodes = [ep for ep in all_episodes if episode_status.get(ep) != 'success']
        
        # Create color schemes
        success_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(success_episodes))) if success_episodes else []
        failure_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(failure_episodes))) if failure_episodes else []
        
        # Plot successful episodes
        for i, ep_id in enumerate(success_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if len(ep_data) > 0:
                log_vfe = np.log(np.abs(ep_data['vfe']) + 1e-6)
                plt.plot(ep_data['step'], log_vfe, alpha=0.6, 
                        color=success_colors[i], linewidth=1.2, label='Success' if i == 0 else "")
        
        # Plot failed episodes with dashed lines
        for i, ep_id in enumerate(failure_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if len(ep_data) > 0:
                log_vfe = np.log(np.abs(ep_data['vfe']) + 1e-6)
                plt.plot(ep_data['step'], log_vfe, alpha=0.6, 
                        color=failure_colors[i], linewidth=1.2, linestyle='--', 
                        label='Failure' if i == 0 else "")
        
        # Add average trajectories for log-normalized VFE
        if success_episodes:
            success_log_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in success_episodes])
            for step in range(max_steps):
                step_values = []
                for ep_id in success_episodes:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step:
                        step_values.append(np.log(np.abs(ep_data.iloc[step]['vfe']) + 1e-6))
                if step_values:
                    success_log_avg.append(np.mean(step_values))
            if success_log_avg:
                plt.plot(range(len(success_log_avg)), success_log_avg, color='darkgreen', 
                        linewidth=3, label=f'Success Average (n={len(success_episodes)})', alpha=0.9)
        
        if failure_episodes:
            failure_log_avg = []
            max_steps = max([len(self.metrics_data[self.metrics_data['episode_id'] == ep]) 
                           for ep in failure_episodes])
            for step in range(max_steps):
                step_values = []
                for ep_id in failure_episodes:
                    ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                    if len(ep_data) > step:
                        step_values.append(np.log(np.abs(ep_data.iloc[step]['vfe']) + 1e-6))
                if step_values:
                    failure_log_avg.append(np.mean(step_values))
            if failure_log_avg:
                plt.plot(range(len(failure_log_avg)), failure_log_avg, color='darkred', 
                        linewidth=3, label=f'Failure Average (n={len(failure_episodes)})', alpha=0.9, linestyle='-.')
        
        plt.title('Log-Normalized Variational Free Energy (VFE) Trajectories', fontsize=16, fontweight='bold')
        plt.xlabel('Step', fontsize=14)
        plt.ylabel('log(|VFE| + 1e-6)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'log_vfe_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created log_vfe_trajectories.png")
    
    def create_log_efe_vs_distance_trajectories(self):
        """Create log EFE vs Distance trajectories for each episode"""
        if (self.metrics_data is None or 
            'efe' not in self.metrics_data.columns or 
            'distance_to_target' not in self.metrics_data.columns):
            return
            
        plt.figure(figsize=(14, 10))
        
        # Get episode data for classification
        episode_status = self.episode_data.set_index('episode_id')['status'].to_dict()
        all_episodes = sorted(self.metrics_data['episode_id'].unique())
        success_episodes = [ep for ep in all_episodes if episode_status.get(ep) == 'success']
        failure_episodes = [ep for ep in all_episodes if episode_status.get(ep) != 'success']
        
        # Create color schemes
        success_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(success_episodes))) if success_episodes else []
        failure_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(failure_episodes))) if failure_episodes else []
        
        # Plot successful episodes
        for i, ep_id in enumerate(success_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id].copy()
            if len(ep_data) > 0:
                # Sort by distance for proper trajectory
                ep_data = ep_data.sort_values('distance_to_target', ascending=False)
                log_efe = np.log(np.abs(ep_data['efe']) + 1e-6)
                plt.plot(ep_data['distance_to_target'], log_efe, alpha=0.6, 
                        color=success_colors[i], linewidth=1.5, marker='o', markersize=2,
                        label='Success' if i == 0 else "")
        
        # Plot failed episodes with different markers
        for i, ep_id in enumerate(failure_episodes):
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id].copy()
            if len(ep_data) > 0:
                # Sort by distance for proper trajectory  
                ep_data = ep_data.sort_values('distance_to_target', ascending=False)
                log_efe = np.log(np.abs(ep_data['efe']) + 1e-6)
                plt.plot(ep_data['distance_to_target'], log_efe, alpha=0.6, 
                        color=failure_colors[i], linewidth=1.5, linestyle='--', 
                        marker='x', markersize=3, label='Failure' if i == 0 else "")
        
        # Add trend analysis
        if len(self.metrics_data) > 10:
            # Overall trend line
            valid_data = self.metrics_data.dropna(subset=['efe', 'distance_to_target'])
            if len(valid_data) > 0:
                log_efe_all = np.log(np.abs(valid_data['efe']) + 1e-6)
                z = np.polyfit(valid_data['distance_to_target'], log_efe_all, 2)  # Quadratic fit
                p = np.poly1d(z)
                
                x_trend = np.linspace(valid_data['distance_to_target'].min(), 
                                    valid_data['distance_to_target'].max(), 100)
                plt.plot(x_trend, p(x_trend), color='black', linewidth=3, 
                        alpha=0.8, label='Overall Trend (Quadratic)', linestyle='-.')
        
        plt.title('Log EFE vs Distance to Target Trajectories\n(Phase 1 Gaussian Kernel Analysis)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Distance to Target (m)', fontsize=14)
        plt.ylabel('log(|EFE| + 1e-6)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # Add statistics text box
        if len(self.metrics_data) > 0:
            stats_text = f"""
            Episodes: {len(all_episodes)}
            Success: {len(success_episodes)} ({len(success_episodes)/len(all_episodes)*100:.1f}%)
            Failure: {len(failure_episodes)} ({len(failure_episodes)/len(all_episodes)*100:.1f}%)
            """
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'log_efe_vs_distance_trajectories.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created log_efe_vs_distance_trajectories.png")
    
    def create_log_vfe_efe_scatter(self):
        """Create standalone log VFE vs log EFE scatter plot"""
        if (self.metrics_data is None or 
            'vfe' not in self.metrics_data.columns or 
            'efe' not in self.metrics_data.columns):
            return
            
        plt.figure(figsize=(10, 8))
        
        # Merge with episode data to get success status
        merged_data = self.metrics_data.merge(
            self.episode_data[['episode_id', 'status']], 
            on='episode_id', how='left'
        )
        
        # Log normalize
        log_vfe = np.log(np.abs(merged_data['vfe']) + 1e-6)
        log_efe = np.log(np.abs(merged_data['efe']) + 1e-6)
        
        # Create scatter plot with success/failure coloring
        success_mask = merged_data['status'] == 'success'
        plt.scatter(log_vfe[success_mask], log_efe[success_mask], 
                   alpha=0.6, c='green', s=20, label=f'Success (n={success_mask.sum()})')
        plt.scatter(log_vfe[~success_mask], log_efe[~success_mask], 
                   alpha=0.6, c='red', s=20, label=f'Failure (n={(~success_mask).sum()})')
        
        # Add trend line for all data
        z = np.polyfit(log_vfe.dropna(), log_efe.dropna(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(log_vfe.min(), log_vfe.max(), 100)
        plt.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2, 
                label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
        
        # Calculate and display correlation
        corr = np.corrcoef(log_vfe.dropna(), log_efe.dropna())[0, 1]
        
        # Move correlation text to the bottom right to avoid legend overlap
        plt.text(0.70, 0.05, f'Correlation: r = {corr:.4f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.title('Log-Normalized VFE vs EFE Relationship', fontsize=16, fontweight='bold')
        plt.xlabel('log(|VFE| + 1e-6)', fontsize=14)
        plt.ylabel('log(|EFE| + 1e-6)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Position the legend at the top left to avoid overlap with correlation text
        plt.legend(loc='upper left', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'log_vfe_efe_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created log_vfe_efe_scatter.png")
    
    def create_planning_time_distribution(self):
        """Create standalone planning time distribution visualization"""
        if self.episode_data is None or 'avg_planning_time_ms' not in self.episode_data.columns:
            return
            
        plt.figure(figsize=(10, 6))
        
        planning_times = self.episode_data['avg_planning_time_ms'].dropna()
        
        # Create histogram
        plt.hist(planning_times, bins=30, alpha=0.7, color='skyblue', 
                edgecolor='black', density=False)
        
        # Add statistics
        mean_time = planning_times.mean()
        median_time = planning_times.median()
        std_time = planning_times.std()
        
        plt.axvline(mean_time, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_time:.2f} ms')
        plt.axvline(median_time, color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {median_time:.2f} ms')
        
        # Add text box with statistics
        stats_text = f'n = {len(planning_times)}\nMean = {mean_time:.2f} ms\nMedian = {median_time:.2f} ms\nStd = {std_time:.2f} ms'
        plt.text(0.7, 0.8, stats_text, transform=plt.gca().transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.title('Planning Time Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Average Planning Time (ms)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'planning_time_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created planning_time_distribution.png")
    
    def create_planning_time_by_outcome(self):
        """Create standalone planning time distribution by episode outcome"""
        if (self.episode_data is None or 
            'avg_planning_time_ms' not in self.episode_data.columns or
            'status' not in self.episode_data.columns):
            return
            
        plt.figure(figsize=(12, 8))
        
        # Get data by outcome
        success_times = self.episode_data[self.episode_data['status'] == 'success']['avg_planning_time_ms'].dropna()
        stuck_times = self.episode_data[self.episode_data['status'] == 'stuck']['avg_planning_time_ms'].dropna()
        timeout_times = self.episode_data[self.episode_data['status'] == 'timeout']['avg_planning_time_ms'].dropna()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Planning Time Distribution by Episode Outcome', fontsize=16, fontweight='bold')
        
        # Overall distribution
        all_times = self.episode_data['avg_planning_time_ms'].dropna()
        axes[0,0].hist(all_times, bins=25, alpha=0.7, color='lightgray', 
                      edgecolor='black', label=f'All Episodes (n={len(all_times)})')
        axes[0,0].axvline(all_times.mean(), color='black', linestyle='--', linewidth=2, 
                         label=f'Mean: {all_times.mean():.2f} ms')
        axes[0,0].set_title('Overall Distribution')
        axes[0,0].set_xlabel('Average Planning Time (ms)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Success distribution
        if len(success_times) > 0:
            axes[0,1].hist(success_times, bins=20, alpha=0.7, color='green', 
                          edgecolor='black', label=f'Success (n={len(success_times)})')
            axes[0,1].axvline(success_times.mean(), color='darkgreen', linestyle='--', linewidth=2, 
                             label=f'Mean: {success_times.mean():.2f} ms')
            axes[0,1].set_title('Success Episodes')
            axes[0,1].set_xlabel('Average Planning Time (ms)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()
        
        # Stuck distribution
        if len(stuck_times) > 0:
            axes[1,0].hist(stuck_times, bins=15, alpha=0.7, color='orange', 
                          edgecolor='black', label=f'Stuck (n={len(stuck_times)})')
            axes[1,0].axvline(stuck_times.mean(), color='darkorange', linestyle='--', linewidth=2, 
                             label=f'Mean: {stuck_times.mean():.2f} ms')
            axes[1,0].set_title('Stuck Episodes')
            axes[1,0].set_xlabel('Average Planning Time (ms)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend()
        
        # Timeout distribution
        if len(timeout_times) > 0:
            axes[1,1].hist(timeout_times, bins=15, alpha=0.7, color='red', 
                          edgecolor='black', label=f'Timeout (n={len(timeout_times)})')
            axes[1,1].axvline(timeout_times.mean(), color='darkred', linestyle='--', linewidth=2, 
                             label=f'Mean: {timeout_times.mean():.2f} ms')
            axes[1,1].set_title('Timeout Episodes')
            axes[1,1].set_xlabel('Average Planning Time (ms)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].legend()
        
        # Add summary statistics text
        summary_text = f"""Summary Statistics:
        Success: Mean={success_times.mean():.1f}ms (n={len(success_times)})
        Stuck: Mean={stuck_times.mean():.1f}ms (n={len(stuck_times)}) 
        Timeout: Mean={timeout_times.mean():.1f}ms (n={len(timeout_times)})"""
        
        if len(timeout_times) == 0:
            axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes, fontsize=12,
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'planning_time_by_outcome.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created planning_time_by_outcome.png")

if __name__ == "__main__":
    """Main execution block for single environment analysis"""
    print("="*80)
    print("SINGLE ENVIRONMENT ACTIVE INFERENCE ANALYSIS")
    print("="*80)
    
    try:
        # Initialize analyzer
        analyzer = SingleEnvironmentAnalyzer()
        
        # Load data
        analyzer.load_data()
        
        # Perform comprehensive analysis
        print("\n[ANALYSIS] Running comprehensive analysis pipeline...")
        
        # Core analysis methods
        analyzer.analyze_performance_overview()
        analyzer.plot_performance_dashboard()
        analyzer.analyze_behavioral_patterns()
        analyzer.analyze_vfe_efe_dynamics()
        analyzer.analyze_planning_behavior()
        analyzer.analyze_success_factors()
        analyzer.analyze_computational_cost()
        analyzer.perform_anova_testing()
        analyzer.investigate_vfe_efe_correlation()
        
        # Generate paper-ready visualizations
        print("\n[PAPER-READY] Generating individual PNG files for paper submission...")
        analyzer.generate_paper_ready_visualizations()
        
        # Generate final report
        analyzer.generate_summary_report()
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE - All visualizations and reports generated")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
