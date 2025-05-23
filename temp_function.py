def generate_target_pool(start_pos: List[float], distance_range: Tuple[float, float],
                      client: airsim.MultirotorClient, num_targets: int = 100,
                      max_attempts: int = 300, seed: int = None,
                      ray_checks: int = 7) -> List[List[float]]:
    """Pre-generate a pool of valid target positions for use throughout the experiment
    
    Args:
        start_pos: Starting drone position [x, y, z]
        distance_range: (min_distance, max_distance) in meters
        client: AirSim client instance
        num_targets: Number of targets to generate for the pool
        max_attempts: Maximum sampling attempts per target
        seed: Random seed for deterministic behavior
        ray_checks: Number of rays to use for validating each target
        
    Returns:
        List[List[float]]: List of target positions [x, y, z] in NED coordinates
        
    Raises:
        ValueError: If unable to generate enough valid targets
    """
    logging.info(f"Generating pool of {num_targets} valid target locations...")
    
    # Create a separate random generator for deterministic target generation
    if seed is not None:
        target_rng = random.Random(seed)
    else:
        target_rng = random.Random()
    
    target_pool = []
    total_attempts = 0
    max_total_attempts = max_attempts * num_targets * 2  # Upper bound to prevent infinite loops
    
    # Start time to track performance
    start_time = time.time()
    
    # Try to generate the requested number of targets
    while len(target_pool) < num_targets and total_attempts < max_total_attempts:
        try:
            # For each target, we'll use a different "episode id" to ensure diversity
            # This lets us leverage the existing sample_visible_target logic directly
            fake_episode_id = len(target_pool)
            
            # Get a valid target location
            target_pos = sample_visible_target(
                start_pos,
                distance_range,
                client,
                max_attempts=max(50, max_attempts // num_targets),
                episode_id=fake_episode_id,
                seed=seed,
                ray_checks=ray_checks
            )
            
            if target_pos:
                target_pool.append(target_pos)
                
                # Log progress periodically
                if len(target_pool) % 10 == 0 or len(target_pool) == num_targets:
                    elapsed = time.time() - start_time
                    logging.info(f"Generated {len(target_pool)}/{num_targets} targets "
                                f"in {elapsed:.1f}s ({total_attempts} attempts)")
        
        except Exception as e:
            logging.warning(f"Error generating target {len(target_pool)}: {e}")
        
        total_attempts += 1
    
    # Check if we generated enough targets
    if len(target_pool) < num_targets:
        logging.warning(f"Could only generate {len(target_pool)}/{num_targets} valid targets "
                      f"after {total_attempts} attempts")
        if len(target_pool) == 0:
            raise ValueError("Failed to generate any valid targets for the pool")
    else:
        logging.info(f"Successfully generated pool of {len(target_pool)} targets "
                   f"in {time.time() - start_time:.1f}s")
    
    return target_pool
