#!/usr/bin/env julia

println("Starting comprehensive precompilation for Julia components...")

# Activate project environment
if Base.active_project() != abspath(joinpath(@__DIR__, "Project.toml"))
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(path=joinpath(@__DIR__, "actinf"))
end

# Import and precompile all necessary packages
println("Importing and precompiling packages...")
@time begin
    using JSON
    using LinearAlgebra
    using StaticArrays
    using ZMQ
    using actinf
end

# Force compilation of critical functions from all modules
println("Precompiling specific functionality...")
@time begin
    # StateSpace precompilation
    dummy_observation = actinf.StateSpace.DroneObservation()
    dummy_state = actinf.StateSpace.DroneState()
    actinf.StateSpace.create_state_from_observation(dummy_observation)
    actinf.StateSpace.calculate_suitability(1.0, 0.2)
    
    # Inference precompilation
    dummy_beliefs = actinf.Inference.initialize_beliefs(dummy_state)
    actinf.Inference.update_beliefs!(dummy_beliefs, dummy_state, obstacle_density=0.0)
    actinf.Inference.expected_state(dummy_beliefs)
    actinf.Inference.serialize_beliefs(dummy_beliefs)
    
    # Planning precompilation
    dummy_preference_model = actinf.Planning.PreferenceModel()
    dummy_planner = actinf.Planning.ActionPlanner()
    dummy_action = SVector{3, Float64}(1.0, 0.0, 0.0)
    actinf.Planning.evaluate_preference(dummy_state, dummy_preference_model)
    actinf.Planning.calculate_efe(
        dummy_state, 
        dummy_beliefs, 
        dummy_action, 
        dummy_preference_model
    )
    
    # Simulate a small planning cycle to precompile select_action
    dummy_position = SVector{3, Float64}(0.0, 0.0, 0.0)
    dummy_target = SVector{3, Float64}(10.0, 0.0, -3.0)
    actinf.Planning.select_action(
        dummy_state,
        dummy_beliefs,
        dummy_planner,
        dummy_position,
        dummy_target,
        obstacle_distance=10.0,
        obstacle_density=0.0
    )
    
    # ZMQ precompilation
    println("Precompiling ZMQ functionality...")
    dummy_context = ZMQ.Context()
    dummy_socket = ZMQ.Socket(dummy_context, ZMQ.REP)
    
    # Try to use the newer API style
    try
        # Try using newer direct socket methods
        ZMQ.set_linger(dummy_socket, 0)
        println("✓ Using newer ZMQ API (direct socket methods)")
    catch
        try
            # Fall back to older API if needed
            println("⚠️ Newer API not available, trying older ZMQ API")
            ZMQ.setsockopt(dummy_socket, ZMQ.LINGER, 0)
            println("✓ Using older ZMQ API (ZMQ.setsockopt)")
        catch e
            println("⚠️ Warning: Could not precompile ZMQ socket options: $e")
        end
    end
    
    # Always close resources
    ZMQ.close(dummy_socket)
    
    # Try terminating context with both APIs
    try
        ZMQ.close(dummy_context)
        println("✓ Context closed with newer API")
    catch
        try
            ZMQ.term(dummy_context)
            println("✓ Context terminated with older API")
        catch e
            println("⚠️ Warning: Could not precompile ZMQ context termination: $e")
        end
    end
end

println("Precompilation complete! All Julia components are ready for execution.")
println("This will significantly reduce runtime overhead during navigation.")