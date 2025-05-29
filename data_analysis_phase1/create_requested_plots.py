#!/usr/bin/env python3
"""
Generate the three specific EFE visualizations requested:
1. EFE vs Distance (standalone)
2. Pragmatic + Epistemic vs Distance (combined)
3. Pragmatic + Epistemic vs Steps (combined)
"""

import sys
import os

try:
    from analyze_single_environment import SingleEnvironmentAnalyzer
    print('✓ Module imported successfully')
    
    analyzer = SingleEnvironmentAnalyzer()
    print('✓ Analyzer created')
    
    analyzer.load_data()
    print('✓ Data loaded successfully')
    
    print('\n=== Creating Requested Visualizations ===')
    
    print('1. Creating EFE vs Distance standalone plot...')
    analyzer.create_efe_vs_distance_standalone()
    
    print('2. Creating Pragmatic + Epistemic vs Distance combined plot...')
    analyzer.create_pragmatic_epistemic_vs_distance_combined()
    
    print('3. Creating Pragmatic + Epistemic vs Steps combined plot...')
    analyzer.create_pragmatic_epistemic_vs_steps_combined()
    
    print('\n✅ All three requested visualizations created successfully!')
    
    # List the created files
    results_dir = 'results'
    if os.path.exists(results_dir):
        print(f'\nNew files created:')
        new_files = [
            'efe_vs_distance_standalone.png',
            'pragmatic_epistemic_vs_distance_combined.png', 
            'pragmatic_epistemic_vs_steps_combined.png'
        ]
        for f in new_files:
            file_path = os.path.join(results_dir, f)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f'  ✓ {f} ({size:,} bytes)')
            else:
                print(f'  ❌ {f} (not found)')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
