"""
COMPREHENSIVE ANALYSIS RUNNER

This script runs the complete analysis pipeline:
1. Processes raw experiment data
2. Runs statistical analysis
3. Generates visualizations
4. Creates an interactive dashboard
"""

import os
import subprocess
import sys
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run the complete analysis pipeline"""
    print("🚀 Starting Comprehensive Active Inference Analysis Pipeline")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Process experiment data
    process_script = os.path.join(script_dir, "process_data.py")
    if os.path.exists(process_script):
        success = run_command(f"python \"{process_script}\"", "Processing experiment data")
        if not success:
            print("⚠️  Data processing failed, but continuing with existing data...")
    
    # Step 2: Run main analysis
    analyze_script = os.path.join(script_dir, "analyze_envs.py")
    success = run_command(f"python \"{analyze_script}\"", "Running statistical analysis and generating plots")
    
    if not success:
        print("❌ Main analysis failed. Exiting.")
        return
    
    # Step 3: Generate interactive dashboard
    dashboard_script = os.path.join(script_dir, "generate_dashboard.py")
    if os.path.exists(dashboard_script):
        run_command(f"python \"{dashboard_script}\"", "Generating interactive dashboard")
    
    # Step 4: Summary
    results_dir = os.path.join(script_dir, "results")
    dashboard_path = os.path.join(results_dir, "dashboard.html")
    report_path = os.path.join(results_dir, "experiment_analysis_report.md")
    
    print("\n" + "="*70)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*70)
    
    if os.path.exists(dashboard_path):
        print(f"📊 Interactive Dashboard: {dashboard_path}")
        print("   💡 Open this file in your browser to explore the results")
    
    if os.path.exists(report_path):
        print(f"📄 Detailed Report: {report_path}")
        print("   📖 Contains statistical analysis and key findings")
    
    print(f"\n📁 All results saved in: {results_dir}")
    print(f"⏰ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Offer to open dashboard
    try:
        import webbrowser
        if os.path.exists(dashboard_path):
            response = input("\n🌐 Would you like to open the dashboard in your browser? (y/n): ")
            if response.lower() in ['y', 'yes']:
                webbrowser.open(f"file:///{dashboard_path}")
                print("🚀 Dashboard opened in your default browser!")
    except:
        pass

if __name__ == "__main__":
    main()
