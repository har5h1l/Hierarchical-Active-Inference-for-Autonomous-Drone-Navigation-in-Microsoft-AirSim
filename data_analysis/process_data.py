"""
EXPERIMENT DATA PROCESSOR AND CONVERTER

This script processes raw experiment data and converts it into standardized formats
for easier analysis by the main analyze_envs.py script.
"""

import os
import pandas as pd
import json
import glob
from datetime import datetime

class ExperimentProcessor:
    """Process and standardize experiment data"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, "experiment_results")
        self.data_dir = os.path.join(base_dir, "data_analysis", "data")
        
        os.makedirs(self.data_dir, exist_ok=True)
    
    def process_all_experiments(self):
        """Process all experiments and create environment-specific datasets"""
        print("🔄 Processing experiment data...")
        
        # Discover experiments
        experiment_folders = glob.glob(os.path.join(self.experiment_dir, "experiment_*"))
        
        env_data = {}  # Group by environment
        
        for folder in experiment_folders:
            try:
                exp_name = os.path.basename(folder)
                config_path = os.path.join(folder, "config.json")
                metrics_path = os.path.join(folder, "metrics.csv")
                episodes_path = os.path.join(folder, "episode_summaries.csv")
                
                if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
                    continue
                
                # Load config to get environment
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                environment = config.get('environment', 'unknown')
                
                # Load metrics
                metrics_df = pd.read_csv(metrics_path)
                metrics_df['experiment'] = exp_name
                metrics_df['timestamp_exp'] = exp_name.split('_')[1] + '_' + exp_name.split('_')[2]
                
                # Add to environment group
                if environment not in env_data:
                    env_data[environment] = []
                env_data[environment].append(metrics_df)
                
                print(f"  ✅ Processed {exp_name} ({environment})")
                
            except Exception as e:
                print(f"  ⚠️  Error processing {folder}: {e}")
        
        # Combine and save by environment
        for env, data_list in env_data.items():
            if data_list:
                combined_df = pd.concat(data_list, ignore_index=True)
                
                # Add required columns for original script compatibility
                if 'plan_triggered' not in combined_df.columns:
                    combined_df['plan_triggered'] = combined_df.get('replanning_occurred', 0).astype(int)
                
                if 'time' not in combined_df.columns:
                    combined_df['time'] = combined_df.get('timestamp', combined_df.get('step', 0))
                
                if 'final_reward' not in combined_df.columns:
                    combined_df['final_reward'] = combined_df.get('distance_improvement', 0)
                
                if 'success' not in combined_df.columns:
                    combined_df['success'] = (combined_df.get('collision', 1) == 0).astype(int)
                
                if 'location_x' not in combined_df.columns:
                    combined_df['location_x'] = combined_df.get('position_x', 0)
                
                # Save environment-specific file
                env_filename = f"{env}_metrics.csv"
                env_path = os.path.join(self.data_dir, env_filename)
                combined_df.to_csv(env_path, index=False)
                
                print(f"📊 Saved {len(combined_df)} rows for environment '{env}' to {env_filename}")
        
        print(f"✅ Processing complete! Data saved to {self.data_dir}")
        return list(env_data.keys())

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = ExperimentProcessor(base_dir)
    environments = processor.process_all_experiments()
    
    print(f"\n🎯 Found environments: {environments}")
    print(f"💡 You can now run the main analysis with: python analyze_envs.py")
