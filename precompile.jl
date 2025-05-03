#!/usr/bin/env julia

# Precompilation script for Julia packages
# This ensures all required packages are properly compiled before running the ZMQ server

println("\n=== Starting Julia Precompilation ===")

# Get the current directory
current_dir = @__DIR__

# Activate the project environment
try
    import Pkg
    Pkg.activate(current_dir)
    println("✓ Project activated")
catch e
    println("❌ Error activating project: $e")
    exit(1)
end

# Add and precompile required packages
required_packages = [
    "ZMQ",
    "JSON",
    "LinearAlgebra",
    "StaticArrays",
    "RxInfer",
    "Distributions",
    "Random",
    "Statistics"
]

# Function to check if a package is installed
function is_package_installed(pkg_name)
    try
        Pkg.status(pkg_name)
        return true
    catch
        return false
    end
end

# Function to install a package
function install_package(pkg_name)
    try
        println("\nInstalling package: $pkg_name")
        Pkg.add(pkg_name)
        println("✓ Successfully installed $pkg_name")
        return true
    catch e
        println("❌ Failed to install $pkg_name: $e")
        return false
    end
end

# Function to precompile a package
function precompile_package(pkg_name)
    try
        println("\nPrecompiling package: $pkg_name")
        Pkg.precompile(pkg_name)
        println("✓ Successfully precompiled $pkg_name")
        return true
    catch e
        println("❌ Failed to precompile $pkg_name: $e")
        return false
    end
end

# Install and precompile all required packages
println("\nInstalling and precompiling packages...")
for pkg in required_packages
    if !is_package_installed(pkg)
        if !install_package(pkg)
            println("⚠️ Skipping $pkg due to installation failure")
            continue
        end
    end
    
    if !precompile_package(pkg)
        println("⚠️ Precompilation failed for $pkg, but continuing")
    end
end

# Special handling for the actinf package
println("\nProcessing actinf package...")
try
    # Get the path to the actinf package
    actinf_path = joinpath(current_dir, "actinf")
    
    # First, ensure the package is properly set up
    if !isfile(joinpath(actinf_path, "Project.toml"))
        println("❌ actinf package is not properly set up - missing Project.toml")
        exit(1)
    end
    
    # Add the local package
    try
        println("Adding local actinf package...")
        Pkg.develop(path=actinf_path)
        println("✓ Successfully added local actinf package")
    catch e
        println("❌ Failed to add local actinf package: $e")
        exit(1)
    end
    
    # Try to precompile
    if !precompile_package("actinf")
        println("⚠️ Initial precompilation failed, trying to build...")
        try
            println("Building actinf package...")
            Pkg.build("actinf")
            println("✓ Successfully built actinf package")
            
            # Try precompiling again after build
            if !precompile_package("actinf")
                println("❌ Precompilation still failed after build")
                exit(1)
            end
        catch e
            println("❌ Failed to build actinf package: $e")
            exit(1)
        end
    end
catch e
    println("❌ Error processing actinf package: $e")
    exit(1)
end

# Final status check
println("\n=== Precompilation Status ===")

# Check if all packages are installed
missing_packages = String[]
for pkg in required_packages
    if !is_package_installed(pkg)
        push!(missing_packages, pkg)
    end
end

if isempty(missing_packages)
    println("✓ All required packages are installed")
else
    println("⚠️ Missing packages: $(join(missing_packages, ", "))")
    println("Attempting to install missing packages...")
    for pkg in missing_packages
        install_package(pkg)
    end
end

# Verify actinf package specifically
try
    println("\nVerifying actinf package...")
    using actinf
    println("✓ actinf package loaded successfully")
catch e
    println("❌ actinf package failed to load: $e")
    exit(1)
end

# Final precompilation attempt
println("\nRunning final precompilation...")
try
    Pkg.precompile()
    println("✓ Final precompilation completed")
catch e
    println("⚠️ Final precompilation had some issues: $e")
    println("Continuing anyway as core functionality may still work")
end

println("\nPrecompilation process completed")