#!/usr/bin/env python3
"""
EFE Components Analysis - Pragmatic vs Epistemic Components vs Distance

Creates publication-ready visualization showing the exploration-exploitation trade-off
in Active Inference drone navigation through EFE component analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def create_efe_components_plot():
    """Create publication-ready EFE components vs distance visualization"""
    
    # Load data
    try:
        metrics = pd.read_csv('data/metrics.csv')
        episodes = pd.read_csv('data/episode_summaries.csv')
    except FileNotFoundError:
        print("Error: Data files not found. Run this from data_analysis_phase1 directory.")
        return
    
    # Merge with episode status for success/failure analysis
    merged_data = metrics.merge(episodes[['episode_id', 'status']], on='episode_id', how='left')
    
    # Filter valid data
    valid_data = merged_data.dropna(subset=['distance_to_target', 'efe_pragmatic', 'efe_epistemic', 'efe'])
    
    if len(valid_data) == 0:
        print("Error: No valid EFE component data found.")
        return
    
    # Set up publication-ready styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Create the main figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Expected Free Energy Components: Exploration-Exploitation Trade-off\nAutonomous Drone Navigation with Active Inference', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Define colors for components
    pragmatic_color = '#2E86AB'  # Blue
    epistemic_color = '#A23B72'  # Purple/Pink
    total_efe_color = '#F18F01'  # Orange
    
    # Plot 1: EFE Components vs Distance (Main Analysis)
    # Sample data for better visualization if dataset is large
    if len(valid_data) > 2000:
        plot_data = valid_data.sample(n=2000, random_state=42)
    else:
        plot_data = valid_data
    
    # Sort by distance for better trend visualization
    plot_data = plot_data.sort_values('distance_to_target')
    
    # Plot pragmatic component (goal-directed behavior)
    ax1.scatter(plot_data['distance_to_target'], plot_data['efe_pragmatic'], 
               alpha=0.6, s=25, color=pragmatic_color, label='Pragmatic Component (Goal-directed)',
               edgecolors='white', linewidth=0.5)
    
    # Plot epistemic component (exploration/uncertainty)
    ax1.scatter(plot_data['distance_to_target'], plot_data['efe_epistemic'], 
               alpha=0.6, s=25, color=epistemic_color, label='Epistemic Component (Exploration)',
               edgecolors='white', linewidth=0.5)
    
    # Plot total EFE for reference
    ax1.scatter(plot_data['distance_to_target'], plot_data['efe'], 
               alpha=0.4, s=15, color=total_efe_color, label='Total EFE (Pragmatic + Epistemic)',
               edgecolors='none')
    
    # Add trend lines for better understanding
    distance_bins = np.linspace(plot_data['distance_to_target'].min(), 
                               plot_data['distance_to_target'].max(), 20)
    
    # Calculate binned averages for trend lines
    pragmatic_means = []
    epistemic_means = []
    total_efe_means = []
    bin_centers = []
    
    for i in range(len(distance_bins) - 1):
        bin_mask = ((plot_data['distance_to_target'] >= distance_bins[i]) & 
                   (plot_data['distance_to_target'] < distance_bins[i + 1]))
        if bin_mask.sum() > 5:  # Only include bins with sufficient data
            pragmatic_means.append(plot_data.loc[bin_mask, 'efe_pragmatic'].mean())
            epistemic_means.append(plot_data.loc[bin_mask, 'efe_epistemic'].mean())
            total_efe_means.append(plot_data.loc[bin_mask, 'efe'].mean())
            bin_centers.append((distance_bins[i] + distance_bins[i + 1]) / 2)
    
    # Plot trend lines
    if len(bin_centers) > 3:
        ax1.plot(bin_centers, pragmatic_means, color=pragmatic_color, linewidth=3, 
                alpha=0.8, label='Pragmatic Trend')
        ax1.plot(bin_centers, epistemic_means, color=epistemic_color, linewidth=3, 
                alpha=0.8, label='Epistemic Trend')
        ax1.plot(bin_centers, total_efe_means, color=total_efe_color, linewidth=3, 
                alpha=0.8, label='Total EFE Trend')
    
    ax1.set_xlabel('Distance to Target (meters)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('EFE Component Value', fontsize=14, fontweight='bold')
    ax1.set_title('EFE Components vs Distance to Target', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # Add explanation text box
    explanation_text = """Key Insights:
• Pragmatic: Goal-directed behavior
• Epistemic: Uncertainty/exploration
• Distance patterns reveal trade-offs"""
    
    ax1.text(0.02, 0.98, explanation_text, transform=ax1.transAxes,
            verticalalignment='top', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Success vs Failure Component Analysis
    success_data = valid_data[valid_data['status'] == 'success']
    failure_data = valid_data[valid_data['status'] != 'success']
    
    # Create violin plots for component comparison
    components_data = []
    
    # Prepare data for violin plots
    if len(success_data) > 0:
        components_data.extend([
            {'Component': 'Pragmatic', 'Value': val, 'Outcome': 'Success'} 
            for val in success_data['efe_pragmatic'].values
        ])
        components_data.extend([
            {'Component': 'Epistemic', 'Value': val, 'Outcome': 'Success'} 
            for val in success_data['efe_epistemic'].values
        ])
    
    if len(failure_data) > 0:
        components_data.extend([
            {'Component': 'Pragmatic', 'Value': val, 'Outcome': 'Failure'} 
            for val in failure_data['efe_pragmatic'].values
        ])
        components_data.extend([
            {'Component': 'Epistemic', 'Value': val, 'Outcome': 'Failure'} 
            for val in failure_data['efe_epistemic'].values
        ])
    
    if components_data:
        comp_df = pd.DataFrame(components_data)
        
        # Create violin plot
        sns.violinplot(data=comp_df, x='Component', y='Value', hue='Outcome', 
                      ax=ax2, palette=['green', 'red'], alpha=0.7)
        
        ax2.set_xlabel('EFE Component Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Component Value', fontsize=14, fontweight='bold')
        ax2.set_title('EFE Components: Success vs Failure Episodes', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(title='Episode Outcome', fontsize=11)
        
        # Add statistical significance testing
        if len(success_data) > 0 and len(failure_data) > 0:
            # Test pragmatic component differences
            prag_stat, prag_p = stats.mannwhitneyu(
                success_data['efe_pragmatic'], failure_data['efe_pragmatic'], 
                alternative='two-sided'
            )
            
            # Test epistemic component differences
            epist_stat, epist_p = stats.mannwhitneyu(
                success_data['efe_epistemic'], failure_data['efe_epistemic'], 
                alternative='two-sided'
            )
            
            # Add significance indicators
            sig_text = f"""Statistical Significance:
Pragmatic: p = {prag_p:.4f}
Epistemic: p = {epist_p:.4f}
α = 0.05 threshold"""
            
            ax2.text(0.98, 0.98, sig_text, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Add dataset information
    dataset_info = f"""Dataset: {len(valid_data):,} data points
Success Rate: {len(success_data)/len(valid_data)*100:.1f}%
Episodes Analyzed: {valid_data['episode_id'].nunique()}"""
    
    fig.text(0.02, 0.02, dataset_info, fontsize=10, alpha=0.7, style='italic')
    
    # Add methodology note
    methodology = "Active Inference: EFE = Pragmatic (goal-directed) + Epistemic (exploration/uncertainty reduction)"
    fig.text(0.98, 0.02, methodology, fontsize=10, alpha=0.7, style='italic', 
             horizontalalignment='right')
    
    plt.tight_layout()
    
    # Save in multiple formats for publication
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = 'efe_components_exploration_exploitation'
    
    # High-resolution PNG for presentations
    plt.savefig(os.path.join(output_dir, f'{base_filename}.png'), 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # PDF for publications
    plt.savefig(os.path.join(output_dir, f'{base_filename}.pdf'), 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # SVG for web/presentations
    plt.savefig(os.path.join(output_dir, f'{base_filename}.svg'), 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"✓ Created publication-ready EFE components analysis:")
    print(f"  - {base_filename}.png (300 DPI)")
    print(f"  - {base_filename}.pdf (vector)")
    print(f"  - {base_filename}.svg (vector)")
    print(f"\nKey findings:")
    print(f"  - Total data points analyzed: {len(valid_data):,}")
    print(f"  - Success episodes: {len(success_data):,} ({len(success_data)/len(valid_data)*100:.1f}%)")
    print(f"  - Failure episodes: {len(failure_data):,} ({len(failure_data)/len(valid_data)*100:.1f}%)")
    
    # Calculate and display component statistics
    if len(valid_data) > 0:
        print(f"\nEFE Component Statistics:")
        print(f"  Pragmatic Component - Mean: {valid_data['efe_pragmatic'].mean():.2f}, Std: {valid_data['efe_pragmatic'].std():.2f}")
        print(f"  Epistemic Component - Mean: {valid_data['efe_epistemic'].mean():.2f}, Std: {valid_data['efe_epistemic'].std():.2f}")
        
        # Correlation with distance
        prag_corr = valid_data['distance_to_target'].corr(valid_data['efe_pragmatic'])
        epist_corr = valid_data['distance_to_target'].corr(valid_data['efe_epistemic'])
        print(f"\nCorrelation with Distance:")
        print(f"  Pragmatic-Distance: r = {prag_corr:.3f}")
        print(f"  Epistemic-Distance: r = {epist_corr:.3f}")

if __name__ == "__main__":
    create_efe_components_plot()
