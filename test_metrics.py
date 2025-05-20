#!/usr/bin/env python3
# test_metrics.py - Script to validate that the advanced metrics are being correctly tracked

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_csv_data(experiment_dir):
    """Load and validate CSV data from a given experiment directory"""
    metrics_file = os.path.join(experiment_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file not found at {metrics_file}")
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def validate_metrics_columns(df):
    """Check that all required metrics columns exist in the dataframe"""
    required_metrics = [
        "vfe", "efe", "delta_vfe", "efe_vs_vfe_gap", 
        "efe_pragmatic", "efe_epistemic", 
        "suitability_std", "replanning_triggered_reason",
        "action_heading_angle_rad", "action_heading_angle_deg"
    ]
    
    missing = [col for col in required_metrics if col not in df.columns]
    present = [col for col in required_metrics if col in df.columns]
    
    print("\n=== METRICS VALIDATION REPORT ===")
    print(f"Total metrics columns: {len(df.columns)}")
    print(f"Required metrics present: {len(present)}/{len(required_metrics)}")
    
    if missing:
        print(f"\n⚠️ MISSING METRICS: {missing}")
    else:
        print("\n✓ All required metrics columns are present!")
    
    return not missing, present, missing

def analyze_metrics_data(df, present_metrics):
    """Basic analysis of the metrics data to validate values"""
    print("\n=== METRICS DATA ANALYSIS ===")
    
    for metric in present_metrics:
        if metric == "replanning_triggered_reason":
            # For categorical data, show distribution
            values = df[metric].fillna("none").value_counts()
            print(f"\n{metric} distribution:")
            for val, count in values.items():
                print(f"  - {val}: {count}")
        else:
            # For numerical data, show statistics
            data = df[metric].dropna()
            if len(data) > 0:
                print(f"\n{metric}:")
                print(f"  - min: {data.min():.6f}")
                print(f"  - max: {data.max():.6f}")
                print(f"  - mean: {data.mean():.6f}")
                print(f"  - non-null values: {len(data)}/{len(df)}")
            else:
                print(f"\n{metric}: No valid data")

def visualize_metrics(df, experiment_dir):
    """Create visualizations for key metrics"""
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # Only create visualizations if we have the main metrics
    if all(m in df.columns for m in ["vfe", "efe", "delta_vfe", "efe_vs_vfe_gap"]):
        # Create output directory for visualizations
        viz_dir = os.path.join(experiment_dir, "metric_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot 1: VFE vs EFE over time
        plt.figure(figsize=(10, 6))
        plt.plot(df["step"], df["vfe"], label="VFE", marker="o", markersize=3)
        plt.plot(df["step"], df["efe"], label="EFE", marker="x", markersize=3)
        plt.title("VFE vs EFE Over Time")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "vfe_efe_over_time.png"))
        
        # Plot 2: Delta VFE over time
        plt.figure(figsize=(10, 6))
        plt.plot(df["step"], df["delta_vfe"], label="Delta VFE", marker="o", markersize=3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title("Change in VFE Over Time (Delta VFE)")
        plt.xlabel("Step")
        plt.ylabel("Delta VFE")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "delta_vfe_over_time.png"))
        
        # Plot 3: EFE Components (Pragmatic vs Epistemic)
        if all(m in df.columns for m in ["efe_pragmatic", "efe_epistemic"]):
            plt.figure(figsize=(10, 6))
            plt.plot(df["step"], df["efe_pragmatic"], label="Pragmatic", marker="o", markersize=3)
            plt.plot(df["step"], df["efe_epistemic"], label="Epistemic", marker="x", markersize=3)
            plt.title("EFE Components Over Time")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "efe_components_over_time.png"))
        
        print(f"✓ Visualizations saved to {viz_dir}")
    else:
        print("⚠️ Cannot generate visualizations: missing essential metrics")

def find_most_recent_experiment():
    """Find the most recent experiment directory"""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results")
    
    if not os.path.exists(base_dir):
        print(f"Error: Experiment directory not found at {base_dir}")
        return None
    
    # Get all experiment directories
    experiment_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("experiment_")]
    
    if not experiment_dirs:
        print("Error: No experiment directories found")
        return None
    
    # Sort by modification time (most recent first)
    experiment_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # Check if metrics.csv exists
    for exp_dir in experiment_dirs:
        metrics_file = os.path.join(exp_dir, "metrics.csv")
        if os.path.exists(metrics_file):
            return exp_dir
    
    print("Warning: No experiment directories with metrics.csv found")
    return experiment_dirs[0]  # Return most recent anyway

def main():
    # Find most recent experiment
    experiment_dir = find_most_recent_experiment()
    if not experiment_dir:
        return
    
    print(f"Analyzing metrics from: {os.path.basename(experiment_dir)}")
    
    # Load CSV data
    df = load_csv_data(experiment_dir)
    if df is None:
        return
    
    # Validate metrics columns
    success, present_metrics, missing_metrics = validate_metrics_columns(df)
    
    # Analyze metrics data
    if present_metrics:
        analyze_metrics_data(df, present_metrics)
    
    # Generate visualizations
    visualize_metrics(df, experiment_dir)
    
    print("\n=== CONCLUSION ===")
    if success:
        print("✅ All required metrics are present in the CSV file.")
        print("✅ The system appears to be correctly tracking all advanced planning and inference metrics.")
    else:
        print("⚠️ Some required metrics are missing from the CSV file.")
        print(f"Missing metrics: {missing_metrics}")

if __name__ == "__main__":
    main()
