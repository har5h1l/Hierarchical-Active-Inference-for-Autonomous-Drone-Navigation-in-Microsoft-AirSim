"""
SINGLE ENVIRONMENT ANALYSIS RUNNER

This script runs the complete analysis pipeline for a single Active Inference environment:
1. Loads episode summaries and detailed metrics data
2. Runs comprehensive statistical analysis
3. Generates visualizations and behavioral pattern analysis
4. Creates performance dashboard and summary report
"""

import os
import subprocess
import sys
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n[PROCESSING] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"OK {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run the complete single environment analysis pipeline"""
    print(">> Starting Single Environment Active Inference Analysis Pipeline")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if data files exist
    data_dir = os.path.join(script_dir, "data")
    episode_file = os.path.join(data_dir, "episode_summaries.csv")
    metrics_file = os.path.join(data_dir, "metrics.csv")
    
    if not os.path.exists(episode_file):
        print(f"[ERROR] Episode summaries file not found: {episode_file}")
        return
    
    if not os.path.exists(metrics_file):
        print(f"[ERROR] Metrics file not found: {metrics_file}")
        return
    
    print(f"[CHART] Found data files:")
    print(f"   Episode summaries: {episode_file}")
    print(f"   Detailed metrics: {metrics_file}")
    
    # Step 1: Process experiment data (if script exists)
    process_script = os.path.join(script_dir, "process_data.py")
    if os.path.exists(process_script):
        success = run_command(f"python \"{process_script}\"", "Processing experiment data")
        if not success:
            print("[WARNING]  Data processing failed, but continuing with existing data...")
    
    # Step 2: Run single environment analysis
    analyze_script = os.path.join(script_dir, "analyze_single_environment.py")
    if not os.path.exists(analyze_script):
        print(f"[ERROR] Single environment analyzer not found: {analyze_script}")
        return
    
    success = run_command(f"python \"{analyze_script}\"", "Running single environment analysis")
    
    if not success:
        print("[ERROR] Single environment analysis failed. Exiting.")
        return    
    # Step 3: Generate interactive dashboard (if script exists)
    dashboard_script = os.path.join(script_dir, "generate_dashboard.py")
    if os.path.exists(dashboard_script):
        run_command(f"python \"{dashboard_script}\"", "Generating interactive dashboard")
    
    # Step 4: Summary
    results_dir = os.path.join(script_dir, "results")
    dashboard_path = os.path.join(results_dir, "single_environment_dashboard.png")
    report_path = os.path.join(results_dir, "single_environment_analysis_report.md")
    
    print("\n" + "="*70)
    print("[SUCCESS] SINGLE ENVIRONMENT ANALYSIS COMPLETE!")
    print("="*70)
    
    if os.path.exists(dashboard_path):
        print(f"[CHART] Performance Dashboard: {dashboard_path}")
        print("   [TIP] Visual summary of experiment performance")
    
    if os.path.exists(report_path):
        print(f"[FILE] Detailed Report: {report_path}")
        print("   [READ] Contains statistical analysis and key findings")
    
    print(f"\n[FOLDER] All results saved in: {results_dir}")
    print(f"[TIME] Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show available files in results directory
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f))]
        if result_files:
            print(f"\n[LIST] Generated files:")
            for file in sorted(result_files):
                print(f"   - {file}")
    
    # Offer to open dashboard image if it exists
    try:
        if os.path.exists(dashboard_path):
            response = input(f"\n[IMAGE]  Would you like to open the dashboard image? (y/n): ")
            if response.lower() in ['y', 'yes']:
                import subprocess
                import platform
                if platform.system() == 'Windows':
                    subprocess.run(['start', dashboard_path], shell=True)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', dashboard_path])
                else:  # Linux
                    subprocess.run(['xdg-open', dashboard_path])
                print(">> Dashboard opened!")
    except Exception as e:
        print(f"[WARNING]  Could not open dashboard automatically: {e}")

if __name__ == "__main__":
    main()
