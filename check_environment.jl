# Install required packages for active inference and check environment
import Pkg

# Show current directory for debugging
println("Working in directory: ", pwd())

# Activate the project
Pkg.activate(".")
println("Project activated. Status:")
Pkg.status()

# Add required packages
required_packages = [
    "JSON",
    "LinearAlgebra",
    "StaticArrays",
    "ZMQ"
]

# Check if packages are installed, and install if missing
for pkg in required_packages
    if pkg != "LinearAlgebra"  # LinearAlgebra is a standard library
        try
            Pkg.add(pkg)
            println("✅ $pkg is installed")
        catch e
            println("❌ Error adding $pkg: $e")
        end
    else
        println("✅ $pkg is a standard library")
    end
end

# Develop the actinf package
println("\nDeveloping actinf package...")
try
    Pkg.develop(path="actinf")
    println("✅ actinf package developed")
catch e
    println("❌ Error developing actinf: $e")
end

# Instantiate and precompile
println("\nInstantiating project...")
Pkg.instantiate()
println("✅ Project instantiated")

println("\nResolving project...")
Pkg.resolve()
println("✅ Project resolved")

println("\nPrecompiling dependencies...")
Pkg.precompile()
println("✅ Dependencies precompiled")

# Create status files to indicate success
try
    open(".precompilation_success", "w") do f
        write(f, "Precompilation completed on $(now())")
    end
    
    using JSON
    open(".precompilation_status.json", "w") do f
        write(f, JSON.json(Dict(
            "status" => "success",
            "message" => "Precompilation completed successfully",
            "timestamp" => string(now())
        )))
    end
    println("✅ Created success indicator files")
catch e
    println("❌ Error creating status files: $e")
end

# Test loading actinf
println("\nTesting actinf package loading...")
try
    @eval using actinf
    println("✅ actinf package loaded successfully")
    
    # Try importing key modules
    @eval using actinf.StateSpace
    println("✅ actinf.StateSpace loaded")
    
    @eval using actinf.Inference
    println("✅ actinf.Inference loaded")
    
    @eval using actinf.Planning
    println("✅ actinf.Planning loaded")
catch e
    println("❌ Error loading actinf modules: $e")
end

println("\nEnvironment check complete!")