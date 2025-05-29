#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Load data
print("Loading data...")
metrics = pd.read_csv('data/metrics.csv')
episodes = pd.read_csv('data/episode_summaries.csv')

# Merge with episode status
print("Merging data...")
valid_data = metrics.dropna(subset=['efe_pragmatic', 'efe_epistemic'])
valid_data = valid_data.merge(episodes[['episode_id', 'status']], on='episode_id', how='left')

print(f"Total valid data points: {len(valid_data)}")

# Separate success and failure data
success_data = valid_data[valid_data['status'] == 'success']
failure_data = valid_data[valid_data['status'] != 'success']

print(f"Success episodes: {len(success_data)}")
print(f"Failure episodes: {len(failure_data)}")

if len(success_data) > 0 and len(failure_data) > 0:
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Prepare data for box plots
    components_data = []
    
    components_data.extend([
        {'Component': 'Pragmatic', 'Value': val, 'Outcome': 'Success'}
        for val in success_data['efe_pragmatic'].values
    ])
    components_data.extend([
        {'Component': 'Epistemic', 'Value': val, 'Outcome': 'Success'}
        for val in success_data['efe_epistemic'].values
    ])
    components_data.extend([
        {'Component': 'Pragmatic', 'Value': val, 'Outcome': 'Failure'}
        for val in failure_data['efe_pragmatic'].values
    ])
    components_data.extend([
        {'Component': 'Epistemic', 'Value': val, 'Outcome': 'Failure'}
        for val in failure_data['efe_epistemic'].values
    ])
    
    comp_df = pd.DataFrame(components_data)
    
    # Create box plot
    sns.boxplot(data=comp_df, x='Component', y='Value', hue='Outcome',
                palette=['#2E86AB', '#E63946'], alpha=0.8)
    
    plt.xlabel('EFE Component Type', fontsize=16, fontweight='bold')
    plt.ylabel('Component Value', fontsize=16, fontweight='bold')
    plt.title('EFE Components: Success vs Failure Episodes\nActive Inference Behavioral Analysis',
             fontsize=18, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Episode Outcome', fontsize=14, title_fontsize=14)
    
    # Statistical testing
    prag_stat, prag_p = stats.mannwhitneyu(
        success_data['efe_pragmatic'], failure_data['efe_pragmatic'],
        alternative='two-sided'
    )
    
    epist_stat, epist_p = stats.mannwhitneyu(
        success_data['efe_epistemic'], failure_data['efe_epistemic'],
        alternative='two-sided'
    )
    
    # Add significance indicators
    sig_text = f"""Statistical Significance (Mann-Whitney U):
Pragmatic: p = {prag_p:.4f} {'*' if prag_p < 0.05 else ''}
Epistemic: p = {epist_p:.4f} {'*' if epist_p < 0.05 else ''}
* p < 0.05 (significant)"""
    
    plt.text(0.98, 0.98, sig_text, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Dataset info
    dataset_info = f"""Dataset Summary:
Success Episodes: {len(success_data):,} ({len(success_data)/len(valid_data)*100:.1f}%)
Failure Episodes: {len(failure_data):,} ({len(failure_data)/len(valid_data)*100:.1f}%)
Total Data Points: {len(valid_data):,}"""
    
    plt.text(0.02, 0.02, dataset_info, transform=plt.gca().transAxes,
            verticalalignment='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = 'results/efe_components_success_failure.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Created {output_path}")
else:
    print("⚠️  Insufficient data for success/failure analysis")
