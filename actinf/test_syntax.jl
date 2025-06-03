#!/usr/bin/env julia

try
    include("src/Inference.jl")
    println("✓ Inference.jl loads successfully")
catch e
    println("✗ Error: ", e)
    rethrow()
end
