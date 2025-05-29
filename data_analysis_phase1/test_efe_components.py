#!/usr/bin/env python3
"""
Test script to generate EFE component visualizations
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
    
    print('Creating EFE pragmatic vs distance...')
    analyzer.create_efe_pragmatic_vs_distance()
    
    print('Creating EFE epistemic vs distance...')
    analyzer.create_efe_epistemic_vs_distance()
    
    print('Creating EFE components success vs failure...')
    analyzer.create_efe_components_success_failure()
    
    print('✓ All EFE component visualizations created!')
    
    # List the results directory to verify files were created
    results_dir = 'results'
    if os.path.exists(results_dir):
        print(f'\nFiles in {results_dir}:')
        files = os.listdir(results_dir)
        efe_files = [f for f in files if 'efe_' in f and f.endswith('.png')]
        for f in efe_files:
            print(f'  - {f}')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
