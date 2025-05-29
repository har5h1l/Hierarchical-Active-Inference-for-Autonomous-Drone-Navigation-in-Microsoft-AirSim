import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Loading data...")
metrics = pd.read_csv('data/metrics.csv')
episodes = pd.read_csv('data/episode_summaries.csv')

# Filter valid data and merge
valid_data = metrics.dropna(subset=['distance_to_target', 'efe', 'efe_pragmatic', 'efe_epistemic', 'step'])
valid_data = valid_data.merge(episodes[['episode_id', 'status']], on='episode_id', how='left')

print(f"Valid data points: {len(valid_data)}")

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# 1. EFE vs Distance (standalone)
print("Creating EFE vs Distance standalone...")
plt.figure(figsize=(14, 10))

if len(valid_data) > 2000:
    plot_data = valid_data.sample(n=2000, random_state=42)
else:
    plot_data = valid_data

plot_data = plot_data.sort_values('distance_to_target')

success_data = plot_data[plot_data['status'] == 'success']
failure_data = plot_data[plot_data['status'] != 'success']

if len(success_data) > 0:
    plt.scatter(success_data['distance_to_target'], success_data['efe'],
               alpha=0.6, s=30, color='#2E86AB', label='Successful Episodes',
               edgecolors='white', linewidth=0.5)

if len(failure_data) > 0:
    plt.scatter(failure_data['distance_to_target'], failure_data['efe'],
               alpha=0.6, s=30, color='#E63946', label='Failed Episodes',
               edgecolors='white', linewidth=0.5)

# Add trend line
distance_bins = np.linspace(plot_data['distance_to_target'].min(),
                           plot_data['distance_to_target'].max(), 20)

efe_means = []
bin_centers = []

for i in range(len(distance_bins) - 1):
    bin_mask = ((plot_data['distance_to_target'] >= distance_bins[i]) &
               (plot_data['distance_to_target'] < distance_bins[i + 1]))
    if bin_mask.sum() > 5:
        efe_means.append(plot_data.loc[bin_mask, 'efe'].mean())
        bin_centers.append((distance_bins[i] + distance_bins[i + 1]) / 2)

if len(bin_centers) > 3:
    plt.plot(bin_centers, efe_means, color='#F18F01', linewidth=4,
            alpha=0.9, label='Trend Line', zorder=10)

plt.xlabel('Distance to Target (meters)', fontsize=16, fontweight='bold')
plt.ylabel('Expected Free Energy (EFE)', fontsize=16, fontweight='bold')
plt.title('Expected Free Energy vs Distance to Target\nActive Inference Navigation Behavior',
         fontsize=18, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14, loc='best', framealpha=0.9)

# Removed explanatory text boxes for cleaner publication-ready visualization

plt.tight_layout()
plt.savefig('results/efe_vs_distance_standalone.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created efe_vs_distance_standalone.png")

# 2. Pragmatic + Epistemic vs Distance (combined)
print("Creating Pragmatic + Epistemic vs Distance combined...")
plt.figure(figsize=(14, 10))

plt.scatter(plot_data['distance_to_target'], plot_data['efe_pragmatic'],
           alpha=0.6, s=30, color='#2E86AB', label='Pragmatic Component (Goal-directed)',
           edgecolors='white', linewidth=0.5)

plt.scatter(plot_data['distance_to_target'], plot_data['efe_epistemic'],
           alpha=0.6, s=30, color='#A23B72', label='Epistemic Component (Exploration)',
           edgecolors='white', linewidth=0.5)

# Add trend lines
pragmatic_means = []
epistemic_means = []
bin_centers = []

for i in range(len(distance_bins) - 1):
    bin_mask = ((plot_data['distance_to_target'] >= distance_bins[i]) &
               (plot_data['distance_to_target'] < distance_bins[i + 1]))
    if bin_mask.sum() > 5:
        pragmatic_means.append(plot_data.loc[bin_mask, 'efe_pragmatic'].mean())
        epistemic_means.append(plot_data.loc[bin_mask, 'efe_epistemic'].mean())
        bin_centers.append((distance_bins[i] + distance_bins[i + 1]) / 2)

if len(bin_centers) > 3:
    plt.plot(bin_centers, pragmatic_means, color='#2E86AB', linewidth=4,
            alpha=0.9, label='Pragmatic Trend', zorder=10)
    plt.plot(bin_centers, epistemic_means, color='#A23B72', linewidth=4,
            alpha=0.9, label='Epistemic Trend', zorder=10)

plt.xlabel('Distance to Target (meters)', fontsize=16, fontweight='bold')
plt.ylabel('EFE Component Value', fontsize=16, fontweight='bold')
plt.title('EFE Components vs Distance to Target\nExploration-Exploitation Trade-off Analysis',
         fontsize=18, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14, loc='best', framealpha=0.9)

# Removed correlation text box for cleaner publication-ready visualization

plt.tight_layout()
plt.savefig('results/pragmatic_epistemic_vs_distance_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created pragmatic_epistemic_vs_distance_combined.png")

# 3. Pragmatic + Epistemic vs Steps (combined)
print("Creating Pragmatic + Epistemic vs Steps combined...")
plt.figure(figsize=(14, 10))

plot_data_steps = plot_data.sort_values('step')

plt.scatter(plot_data_steps['step'], plot_data_steps['efe_pragmatic'],
           alpha=0.6, s=30, color='#2E86AB', label='Pragmatic Component (Goal-directed)',
           edgecolors='white', linewidth=0.5)

plt.scatter(plot_data_steps['step'], plot_data_steps['efe_epistemic'],
           alpha=0.6, s=30, color='#A23B72', label='Epistemic Component (Exploration)',
           edgecolors='white', linewidth=0.5)

# Add trend lines for steps
step_bins = np.linspace(plot_data_steps['step'].min(), plot_data_steps['step'].max(), 20)

pragmatic_means_steps = []
epistemic_means_steps = []
bin_centers_steps = []

for i in range(len(step_bins) - 1):
    bin_mask = ((plot_data_steps['step'] >= step_bins[i]) &
               (plot_data_steps['step'] < step_bins[i + 1]))
    if bin_mask.sum() > 5:
        pragmatic_means_steps.append(plot_data_steps.loc[bin_mask, 'efe_pragmatic'].mean())
        epistemic_means_steps.append(plot_data_steps.loc[bin_mask, 'efe_epistemic'].mean())
        bin_centers_steps.append((step_bins[i] + step_bins[i + 1]) / 2)

if len(bin_centers_steps) > 3:
    plt.plot(bin_centers_steps, pragmatic_means_steps, color='#2E86AB', linewidth=4,
            alpha=0.9, label='Pragmatic Trend', zorder=10)
    plt.plot(bin_centers_steps, epistemic_means_steps, color='#A23B72', linewidth=4,
            alpha=0.9, label='Epistemic Trend', zorder=10)

plt.xlabel('Episode Step', fontsize=16, fontweight='bold')
plt.ylabel('EFE Component Value', fontsize=16, fontweight='bold')
plt.title('EFE Components vs Episode Steps\nTemporal Exploration-Exploitation Dynamics',
         fontsize=18, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14, loc='best', framealpha=0.9)

# Removed correlation and explanation text boxes for cleaner publication-ready visualization

plt.tight_layout()
plt.savefig('results/pragmatic_epistemic_vs_steps_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created pragmatic_epistemic_vs_steps_combined.png")

print("\n🎉 All three requested visualizations created successfully!")

# 4. Log-normalized EFE vs Distance (standalone)
print("Creating Log-normalized EFE vs Distance standalone...")
plt.figure(figsize=(14, 10))

# Create log-normalized EFE values (use absolute value since EFE can be negative)
success_data_log = plot_data[plot_data['status'] == 'success']
failure_data_log = plot_data[plot_data['status'] != 'success']

if len(success_data_log) > 0:
    plt.scatter(success_data_log['distance_to_target'], np.log(np.abs(success_data_log['efe']) + 1e-10),
               alpha=0.6, s=30, color='#2E86AB', label='Successful Episodes',
               edgecolors='white', linewidth=0.5)

if len(failure_data_log) > 0:
    plt.scatter(failure_data_log['distance_to_target'], np.log(np.abs(failure_data_log['efe']) + 1e-10),
               alpha=0.6, s=30, color='#E63946', label='Failed Episodes',
               edgecolors='white', linewidth=0.5)

# Add trend line for log-normalized data
log_efe_means = []
bin_centers_log = []

for i in range(len(distance_bins) - 1):
    bin_mask = ((plot_data['distance_to_target'] >= distance_bins[i]) &
               (plot_data['distance_to_target'] < distance_bins[i + 1]))
    if bin_mask.sum() > 5:
        log_efe_means.append(np.log(np.abs(plot_data.loc[bin_mask, 'efe'].mean()) + 1e-10))
        bin_centers_log.append((distance_bins[i] + distance_bins[i + 1]) / 2)

if len(bin_centers_log) > 3:
    plt.plot(bin_centers_log, log_efe_means, color='#F18F01', linewidth=4,
            alpha=0.9, label='Trend Line', zorder=10)

plt.xlabel('Distance to Target (meters)', fontsize=16, fontweight='bold')
plt.ylabel('Log |Expected Free Energy|', fontsize=16, fontweight='bold')
plt.title('Log-normalized |Expected Free Energy| vs Distance to Target\nActive Inference Navigation Behavior',
         fontsize=18, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14, loc='best', framealpha=0.9)

plt.tight_layout()
plt.savefig('results/log_efe_vs_distance_standalone.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created log_efe_vs_distance_standalone.png")

# 5. Log-normalized Pragmatic + Epistemic vs Distance (combined)
print("Creating Log-normalized Pragmatic + Epistemic vs Distance combined...")
plt.figure(figsize=(14, 10))

# Create log-normalized component values (use absolute value since components can be negative)
plt.scatter(plot_data['distance_to_target'], np.log(np.abs(plot_data['efe_pragmatic']) + 1e-10),
           alpha=0.6, s=30, color='#2E86AB', label='Log |Pragmatic Component| (Goal-directed)',
           edgecolors='white', linewidth=0.5)

plt.scatter(plot_data['distance_to_target'], np.log(np.abs(plot_data['efe_epistemic']) + 1e-10),
           alpha=0.6, s=30, color='#A23B72', label='Log |Epistemic Component| (Exploration)',
           edgecolors='white', linewidth=0.5)

# Add trend lines for log-normalized components
log_pragmatic_means = []
log_epistemic_means = []
bin_centers_components = []

for i in range(len(distance_bins) - 1):
    bin_mask = ((plot_data['distance_to_target'] >= distance_bins[i]) &
               (plot_data['distance_to_target'] < distance_bins[i + 1]))
    if bin_mask.sum() > 5:
        log_pragmatic_means.append(np.log(np.abs(plot_data.loc[bin_mask, 'efe_pragmatic'].mean()) + 1e-10))
        log_epistemic_means.append(np.log(np.abs(plot_data.loc[bin_mask, 'efe_epistemic'].mean()) + 1e-10))
        bin_centers_components.append((distance_bins[i] + distance_bins[i + 1]) / 2)

if len(bin_centers_components) > 3:
    plt.plot(bin_centers_components, log_pragmatic_means, color='#2E86AB', linewidth=4,
            alpha=0.9, label='Log |Pragmatic| Trend', zorder=10)
    plt.plot(bin_centers_components, log_epistemic_means, color='#A23B72', linewidth=4,
            alpha=0.9, label='Log |Epistemic| Trend', zorder=10)

plt.xlabel('Distance to Target (meters)', fontsize=16, fontweight='bold')
plt.ylabel('Log |EFE Component Value|', fontsize=16, fontweight='bold')
plt.title('Log-normalized |EFE Components| vs Distance to Target\nExploration-Exploitation Trade-off Analysis',
         fontsize=18, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14, loc='best', framealpha=0.9)

plt.tight_layout()
plt.savefig('results/log_pragmatic_epistemic_vs_distance_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created log_pragmatic_epistemic_vs_distance_combined.png")

print("\n🎉 All five visualizations created successfully (including log-normalized versions)!")
