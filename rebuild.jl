#!/usr/bin/env julia

println("Starting package rebuild process...")

import Pkg
println("Activating main project...")
Pkg.activate(".")

println("Removing existing actinf development...")
Pkg.rm("actinf")

println("Adding actinf package in development mode...")
Pkg.develop(path="./actinf")

println("Installing dependencies...")
Pkg.instantiate()

println("Importing actinf to verify installation...")
using actinf

println("Package rebuild complete!")