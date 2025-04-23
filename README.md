# AirSim Project with RxInfer.jl

This project interfaces AirSim with Julia for probabilistic inference using RxInfer.jl.

## Structure
- scripts/ - Python scripts for AirSim control
- src/ - Julia modules for sensor processing, fusion, inference, and planning
- run.jl - Main entrypoint for the project

## Requirements
- Julia 1.7+
- Python 3.7+
- AirSim
- PyCall.jl
- RxInfer.jl

## Setup
1. Install AirSim following the [official documentation](https://microsoft.github.io/AirSim/build_windows/)
2. Install Julia dependencies:
```julia
using Pkg
Pkg.add(["PyCall", "RxInfer", "Images", "StaticArrays", "LinearAlgebra"])
```
3. Configure AirSim settings.json for your simulation

## Running
```
julia run.jl
```
