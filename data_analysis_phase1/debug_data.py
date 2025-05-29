import pandas as pd

# Load data
metrics = pd.read_csv('data/metrics.csv')
episodes = pd.read_csv('data/episode_summaries.csv')

print("=== COLUMN ANALYSIS ===")
print("Metrics columns:", metrics.columns.tolist())
print("\nEFE-related columns:")
for col in metrics.columns:
    if 'efe' in col.lower():
        print(f"  - {col}")

print("\n=== DATA SAMPLES ===")
print("Metrics sample:")
print(metrics.head())

print("\nEpisodes sample:")
print(episodes.head())

print("\n=== STATUS DISTRIBUTION ===")
print(episodes['status'].value_counts())

# Check if we have the EFE components
if 'efe_pragmatic' in metrics.columns and 'efe_epistemic' in metrics.columns:
    print("\n✅ EFE components found!")
    
    # Quick merge test
    merged = metrics.merge(episodes[['episode_id', 'status']], on='episode_id', how='left')
    success_count = len(merged[merged['status'] == 'success'])
    failure_count = len(merged[merged['status'] != 'success'])
    
    print(f"Success data points: {success_count}")
    print(f"Failure data points: {failure_count}")
    
    if success_count > 0 and failure_count > 0:
        print("✅ Ready for success/failure analysis!")
    else:
        print("❌ Not enough data for success/failure analysis")
else:
    print("❌ EFE components not found in metrics")
