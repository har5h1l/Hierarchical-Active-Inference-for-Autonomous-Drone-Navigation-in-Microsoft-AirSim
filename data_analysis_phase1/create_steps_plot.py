import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
metrics = pd.read_csv('data/metrics.csv')
episodes = pd.read_csv('data/episode_summaries.csv')

print("Available columns in metrics:")
for col in metrics.columns:
    print(f"  - {col}")

# Check for step column
if 'step' in metrics.columns:
    print(f"\n✓ Step column found!")
    print(f"Step range: {metrics['step'].min()} to {metrics['step'].max()}")
    
    # Filter valid data
    valid_data = metrics.dropna(subset=['step', 'efe_pragmatic', 'efe_epistemic'])
    print(f"Valid step data points: {len(valid_data)}")
    
    if len(valid_data) > 0:
        # Create the missing third plot
        plt.figure(figsize=(14, 10))
        
        if len(valid_data) > 2000:
            plot_data = valid_data.sample(n=2000, random_state=42)
        else:
            plot_data = valid_data
        
        plot_data = plot_data.sort_values('step')
        
        plt.scatter(plot_data['step'], plot_data['efe_pragmatic'],
                   alpha=0.6, s=30, color='#2E86AB', label='Pragmatic Component (Goal-directed)',
                   edgecolors='white', linewidth=0.5)
        
        plt.scatter(plot_data['step'], plot_data['efe_epistemic'],
                   alpha=0.6, s=30, color='#A23B72', label='Epistemic Component (Exploration)',
                   edgecolors='white', linewidth=0.5)
        
        # Add trend lines
        step_bins = np.linspace(plot_data['step'].min(), plot_data['step'].max(), 20)
        
        pragmatic_means = []
        epistemic_means = []
        bin_centers = []
        
        for i in range(len(step_bins) - 1):
            bin_mask = ((plot_data['step'] >= step_bins[i]) &
                       (plot_data['step'] < step_bins[i + 1]))
            if bin_mask.sum() > 5:
                pragmatic_means.append(plot_data.loc[bin_mask, 'efe_pragmatic'].mean())
                epistemic_means.append(plot_data.loc[bin_mask, 'efe_epistemic'].mean())
                bin_centers.append((step_bins[i] + step_bins[i + 1]) / 2)
        
        if len(bin_centers) > 3:
            plt.plot(bin_centers, pragmatic_means, color='#2E86AB', linewidth=4,
                    alpha=0.9, label='Pragmatic Trend', zorder=10)
            plt.plot(bin_centers, epistemic_means, color='#A23B72', linewidth=4,
                    alpha=0.9, label='Epistemic Trend', zorder=10)
        
        plt.xlabel('Episode Step', fontsize=16, fontweight='bold')
        plt.ylabel('EFE Component Value', fontsize=16, fontweight='bold')
        plt.title('EFE Components vs Episode Steps\nTemporal Exploration-Exploitation Dynamics',
                 fontsize=18, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14, loc='best', framealpha=0.9)
        
        # Add correlations
        prag_corr = plot_data['step'].corr(plot_data['efe_pragmatic'])
        epist_corr = plot_data['step'].corr(plot_data['efe_epistemic'])
        
        corr_text = f"""Correlations with Steps:
Pragmatic: r = {prag_corr:.3f}
Epistemic: r = {epist_corr:.3f}
Data Points: {len(plot_data):,}"""
        
        plt.text(0.02, 0.98, corr_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        explanation = """Temporal Dynamics:
• Shows how components evolve over time
• Pragmatic: Goal pursuit intensity
• Epistemic: Learning/exploration over steps"""
        plt.text(0.98, 0.02, explanation, transform=plt.gca().transAxes,
                verticalalignment='bottom', horizontalalignment='right', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('results/pragmatic_epistemic_vs_steps_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Created pragmatic_epistemic_vs_steps_combined.png")
    else:
        print("❌ No valid step data available")
else:
    print("❌ No 'step' column found in metrics data")
    print("\nAlternative: Using row index as step proxy...")
    
    # Use row index as step proxy
    valid_data = metrics.dropna(subset=['efe_pragmatic', 'efe_epistemic'])
    valid_data = valid_data.reset_index(drop=True)
    valid_data['step_proxy'] = valid_data.index
    
    plt.figure(figsize=(14, 10))
    
    if len(valid_data) > 2000:
        plot_data = valid_data.sample(n=2000, random_state=42)
    else:
        plot_data = valid_data
    
    plot_data = plot_data.sort_values('step_proxy')
    
    plt.scatter(plot_data['step_proxy'], plot_data['efe_pragmatic'],
               alpha=0.6, s=30, color='#2E86AB', label='Pragmatic Component (Goal-directed)',
               edgecolors='white', linewidth=0.5)
    
    plt.scatter(plot_data['step_proxy'], plot_data['efe_epistemic'],
               alpha=0.6, s=30, color='#A23B72', label='Epistemic Component (Exploration)',
               edgecolors='white', linewidth=0.5)
    
    plt.xlabel('Data Point Sequence (Proxy for Steps)', fontsize=16, fontweight='bold')
    plt.ylabel('EFE Component Value', fontsize=16, fontweight='bold')
    plt.title('EFE Components vs Data Sequence\nTemporal Exploration-Exploitation Dynamics',
             fontsize=18, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc='best', framealpha=0.9)
    
    # Add correlations
    prag_corr = plot_data['step_proxy'].corr(plot_data['efe_pragmatic'])
    epist_corr = plot_data['step_proxy'].corr(plot_data['efe_epistemic'])
    
    corr_text = f"""Correlations with Sequence:
Pragmatic: r = {prag_corr:.3f}
Epistemic: r = {epist_corr:.3f}
Data Points: {len(plot_data):,}"""
    
    plt.text(0.02, 0.98, corr_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    explanation = """Temporal Dynamics:
• Sequence-based analysis
• Pragmatic: Goal pursuit patterns
• Epistemic: Exploration patterns"""
    plt.text(0.98, 0.02, explanation, transform=plt.gca().transAxes,
            verticalalignment='bottom', horizontalalignment='right', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/pragmatic_epistemic_vs_steps_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created pragmatic_epistemic_vs_steps_combined.png (using sequence proxy)")
