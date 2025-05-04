module actinf

using LinearAlgebra
using StaticArrays
using JSON

# Include and export submodules
include("StateSpace.jl")
include("Inference.jl")
include("Planning.jl")

# Re-export submodule contents
using .StateSpace
using .Inference
using .Planning

# Export types and functions
export DroneState, DroneObservation, create_state_from_observation  # from StateSpace
export DroneBeliefs, initialize_beliefs, update_beliefs!, expected_state  # from Inference
export serialize_beliefs, deserialize_beliefs  # from Inference
export PreferenceModel, evaluate_preference, ActionPlanner, select_action  # from Planning

end # module