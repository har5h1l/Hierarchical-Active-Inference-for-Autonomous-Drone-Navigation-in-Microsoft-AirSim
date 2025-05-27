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
        """Detailed analysis of VFE and EFE dynamics"""
        if self.metrics_data is None:
            return
            
        # VFE/EFE over time analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('VFE/EFE Dynamics Analysis', fontsize=16, fontweight='bold')
        
        # Sample a few episodes for detailed trajectory analysis
        sample_episodes = self.metrics_data['episode_id'].unique()[:5]
        
        # VFE trajectories
        for ep_id in sample_episodes:
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'vfe' in ep_data.columns:
                axes[0,0].plot(ep_data['step'], ep_data['vfe'], alpha=0.7, label=f'Episode {ep_id}')
        axes[0,0].set_title('VFE Trajectories (Sample Episodes)')
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('VFE')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # EFE trajectories
        for ep_id in sample_episodes:
            ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
            if 'efe' in ep_data.columns:
                axes[0,1].plot(ep_data['step'], ep_data['efe'], alpha=0.7, label=f'Episode {ep_id}')
        axes[0,1].set_title('EFE Trajectories (Sample Episodes)')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('EFE')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # VFE vs EFE scatter (all steps)
        if 'vfe' in self.metrics_data.columns and 'efe' in self.metrics_data.columns:
            sample_data = self.metrics_data.sample(min(1000, len(self.metrics_data)))  # Sample for performance
            axes[1,0].scatter(sample_data['vfe'], sample_data['efe'], alpha=0.5, s=20)
            axes[1,0].set_xlabel('VFE')
            axes[1,0].set_ylabel('EFE')
            axes[1,0].set_title('VFE vs EFE (All Steps)')
            axes[1,0].grid(True, alpha=0.3)
        
        # Distance to target over time
        if 'distance_to_target' in self.metrics_data.columns:
            for ep_id in sample_episodes:
                ep_data = self.metrics_data[self.metrics_data['episode_id'] == ep_id]
                axes[1,1].plot(ep_data['step'], ep_data['distance_to_target'], alpha=0.7, label=f'Episode {ep_id}')
            axes[1,1].set_title('Distance to Target Over Time')
            axes[1,1].set_xlabel('Step')
            axes[1,1].set_ylabel('Distance to Target')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'vfe_efe_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("OK VFE/EFE dynamics analysis completed")
    
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
