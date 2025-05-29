#!/usr/bin/env python3
from analyze_single_environment import SingleEnvironmentAnalyzer

analyzer = SingleEnvironmentAnalyzer()
analyzer.load_data()
analyzer.create_efe_components_success_failure()
print("✓ Created efe_components_success_failure.png")
