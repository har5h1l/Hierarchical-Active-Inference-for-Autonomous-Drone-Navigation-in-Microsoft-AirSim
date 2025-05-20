from data_collection import run_experiment

if __name__ == "__main__":
    # Answer "n" to the resume prompt automatically
    import builtins
    original_input = builtins.input
    
    def custom_input(prompt):
        if "resume" in prompt.lower():
            print("Automatically answering 'n' to resume prompt")
            return "n"
        return original_input(prompt)
    
    builtins.input = custom_input
      # Run the experiment
    run_experiment({
        'num_episodes': 5, 
        'precompile_julia': False,
        'max_steps_per_episode': 150,  # Allow more steps per episode
        'target_distance_range': (20.0, 60.0),  # Try some longer distances
        'episode_timeout': 180  # Allow more time per episode
    })
    # Restore original input function
    builtins.input = original_input
