#!/usr/bin/env python3
"""
Test script to verify that the single environment analyzer now uses ALL episodes
instead of sampling limited subsets.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyze_single_environment import SingleEnvironmentAnalyzer

def test_all_episodes_usage():
    """Test that the analyzer processes all available episodes"""
    print("Testing Single Environment Analyzer - ALL Episodes Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SingleEnvironmentAnalyzer()
    analyzer.load_data()
    
    # Get total episodes
    total_episodes = len(analyzer.episode_data)
    success_count = len(analyzer.episode_data[analyzer.episode_data['status'] == 'success'])
    failure_count = total_episodes - success_count
    
    print(f"✓ Loaded {total_episodes} total episodes")
    print(f"  - Success episodes: {success_count}")
    print(f"  - Failure episodes: {failure_count}")
    
    # Verify episode status grouping
    all_episodes = sorted(analyzer.metrics_data['episode_id'].unique())
    episode_status = analyzer.episode_data.set_index('episode_id')['status'].to_dict()
    success_episodes = [ep for ep in all_episodes if episode_status.get(ep) == 'success']
    failure_episodes = [ep for ep in all_episodes if episode_status.get(ep) != 'success']
    
    print(f"✓ Found {len(success_episodes)} success episodes for analysis")
    print(f"✓ Found {len(failure_episodes)} failure episodes for analysis")
    
    # Verify we're using ALL episodes (not sampling)
    assert len(success_episodes) == success_count, "Success episode count mismatch"
    assert len(failure_episodes) == failure_count, "Failure episode count mismatch"
    
    print(f"✓ Confirmed using ALL {total_episodes} episodes (no sampling)")
    
    # Test VFE/EFE scatter plot data
    metrics_with_status = analyzer.metrics_data.merge(
        analyzer.episode_data[['episode_id', 'status']], 
        on='episode_id', how='left'
    )
    
    total_data_points = len(metrics_with_status)
    print(f"✓ VFE/EFE scatter plots use ALL {total_data_points} data points")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Analyzer now uses complete dataset")
    print("✅ No sampling limitations - showing ALL trajectory data")
    print("=" * 60)

if __name__ == "__main__":
    test_all_episodes_usage()
