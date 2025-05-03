#!/usr/bin/env julia

# ZeroMQ-based server for drone state inference and planning
# Replaces file-based communication with a direct ZMQ REP socket

println("Starting Active Inference ZMQ Server...")

# ZMQ.jl API compatibility layer
# This ensures the script works with both newer and older versions of ZMQ.jl
module ZMQCompat
    # Export all the functions we'll use as compatibility wrappers
    export setup_zmq_api, bind, send, recv, close, set_linger, set_rcvhwm, set_sndhwm, set_tcp_keepalive
    
    # Global variables to store which API version we're using
    using_new_api = false
    zmq_module = nothing
    
    # Setup function to determine API version and configure wrappers
    function setup_zmq_api(zmq_mod)
        global zmq_module = zmq_mod
        global using_new_api = false
        
        # Test if we're using the newer API by trying to call set_linger directly
        try
            # Create temporary socket to test API
            ctx = zmq_mod.Context()
            sock = zmq_mod.Socket(ctx, zmq_mod.REP)
            
            # Try to use newer direct method
            zmq_mod.set_linger(sock, 0)
            global using_new_api = true
            println("✓ Using newer ZMQ.jl API with direct socket methods")
            
            # Clean up test socket
            zmq_mod.close(sock)
            zmq_mod.close(ctx)
        catch
            # Fall back to older API
            println("✓ Using older ZMQ.jl API with ZMQ.setsockopt")
        end
        
        return using_new_api
    end
    
    # Compatibility wrappers for ZMQ functions
    function bind(socket, address)
        if using_new_api
            return zmq_module.bind(socket, address)
        else
            return zmq_module.bind(socket, address)  # Same in both APIs
        end
    end
    
    function send(socket, message)
        if using_new_api
            return zmq_module.send(socket, message)
        else
            return zmq_module.send(socket, message)  # Same in both APIs
        end
    end
    
    function recv(socket)
        if using_new_api
            return zmq_module.recv(socket)
        else
            return zmq_module.recv(socket)  # Same in both APIs
        end
    end
    
    function close(obj)
        if using_new_api
            return zmq_module.close(obj)
        else
            # Determine if obj is a socket or context
            if typeof(obj) <: zmq_module.Socket
                return zmq_module.close(obj)  # Same in both APIs
            else
                # For context, use term in older API
                return zmq_module.term(obj)
            end
        end
    end
    
    function set_linger(socket, value)
        if using_new_api
            return zmq_module.set_linger(socket, value)
        else
            return zmq_module.setsockopt(socket, zmq_module.LINGER, value)
        end
    end
    
    function set_rcvhwm(socket, value)
        if using_new_api
            return zmq_module.set_rcvhwm(socket, value)
        else
            return zmq_module.setsockopt(socket, zmq_module.RCVHWM, value)
        end
    end
    
    function set_sndhwm(socket, value)
        if using_new_api
            return zmq_module.set_sndhwm(socket, value)
        else
            return zmq_module.setsockopt(socket, zmq_module.SNDHWM, value)
        end
    end
    
    function set_tcp_keepalive(socket, value)
        try
            if using_new_api
                return zmq_module.set_tcp_keepalive(socket, value)
            else
                return zmq_module.setsockopt(socket, zmq_module.TCP_KEEPALIVE, value)
            end
        catch
            println("TCP_KEEPALIVE not supported in this ZMQ version")
        end
    end
end

# Explicit package loading with error handling
function load_package(pkg_name)
    try
        println("Loading package: $pkg_name")
        @eval import Pkg
        try
            # First try to use the package if it's already loaded
            @eval using $pkg_name
            println("✓ Successfully loaded $pkg_name")
            return true
        catch e
            # If using fails, try adding and then using
            println("⚠️ Could not load $pkg_name directly, trying to add it...")
            try
                Pkg.add("$pkg_name")
                @eval using $pkg_name
                println("✓ Successfully added and loaded $pkg_name")
                return true
            catch e2
                println("❌ Failed to add and load $pkg_name: $e2")
                return false
            end
        end
    catch e
        println("❌ Critical error with $pkg_name: $e")
        return false
    end
end

# Ensure we can write to the status file before proceeding
global status_file_path = nothing
try
    # Create status file path - handle both Windows and UNIX paths
    if Sys.iswindows()
        # Use absolute path with proper Windows separators
        parent_dir = dirname(dirname(abspath(@__FILE__)))
        global status_file_path = joinpath(parent_dir, "zmq_server_running.status")
    else
        global status_file_path = joinpath(dirname(dirname(@__FILE__)), "zmq_server_running.status")
    end
    
    # Test that we can write to this location
    println("Testing if we can write status file to: $status_file_path")
    touch(status_file_path)
    rm(status_file_path)
    println("✓ Status file location is writable")
catch e
    # Try another location if the first one fails
    println("❌ Could not write to status file location: $e")
    try
        if Sys.iswindows()
            # Try temp directory on Windows
            global status_file_path = joinpath(tempdir(), "zmq_server_running.status")
        else
            global status_file_path = joinpath("/tmp", "zmq_server_running.status")
        end
        println("Trying alternative status file location: $status_file_path")
        touch(status_file_path)
        rm(status_file_path)
        println("✓ Alternative status file location is writable")
    catch e2
        println("❌ Could not write to alternative status file location: $e2")
        println("Will continue without status file. Client may not detect server properly.")
        global status_file_path = nothing
    end
end

# Activate project environment
println("Setting up Julia environment...")
if Base.active_project() != abspath(joinpath(dirname(@__DIR__), "Project.toml"))
    try
        import Pkg
        Pkg.activate(dirname(@__DIR__))
        println("✓ Project activated")
        try
            Pkg.develop(path=@__DIR__)  # Develop the actinf package
            println("✓ Actinf package developed")
        catch e
            println("⚠️ Could not develop actinf package: $e")
        end
    catch e
        println("❌ Error activating project: $e")
    end
end

# Check Julia version
println("Running on Julia version: $(VERSION)")

# Import all necessary packages with error handling
global packages_loaded = true
packages_loaded &= load_package(:ZMQ)
packages_loaded &= load_package(:JSON)
packages_loaded &= load_package(:LinearAlgebra)
packages_loaded &= load_package(:StaticArrays)

# Set up ZMQ compatibility layer
using .ZMQCompat
ZMQCompat.setup_zmq_api(ZMQ)

# Load actinf package (with special handling)
try
    println("Loading actinf package...")
    @eval using actinf
    println("✓ actinf package loaded")
catch e
    println("❌ Failed to load actinf package: $e")
    global packages_loaded = false
end

if !packages_loaded
    println("❌ Some required packages failed to load. Cannot continue.")
    exit(1)
end

# Define constants with more robust defaults
const SOCKET_ADDRESS = "tcp://*:5555"
const POLICY_LENGTH = 3  # Number of steps in the policy
const SOCKET_TIMEOUT = 10000  # 10 seconds timeout
const MAX_RETRIES = 3
const RETRY_DELAY = 2  # seconds

# Print server info
println("Server initialized with:")
println("- Address: $SOCKET_ADDRESS")
println("- Policy length: $POLICY_LENGTH")
println("- Socket timeout: $SOCKET_TIMEOUT ms")
println("- Max retries: $MAX_RETRIES")
if status_file_path !== nothing
    println("- Status file: $status_file_path")
else
    println("- Status file: DISABLED")
end

# Initialize beliefs as a global variable for persistence across requests
global current_beliefs = nothing

# Function to handle socket setup with proper error handling
function setup_socket()
    try
        ctx = ZMQ.Context()
        socket = ZMQ.Socket(ctx, ZMQ.REP)
        
        # Set socket options with proper error handling
        try
            ZMQCompat.set_linger(socket, 0)
            ZMQCompat.set_rcvhwm(socket, 100)
            ZMQCompat.set_sndhwm(socket, 100)
            ZMQCompat.set_tcp_keepalive(socket, 1)
            
            # Set timeouts
            if ZMQCompat.using_new_api
                socket.rcvtimeo = SOCKET_TIMEOUT
                socket.sndtimeo = SOCKET_TIMEOUT
            else
                ZMQ.setsockopt(socket, ZMQ.RCVTIMEO, SOCKET_TIMEOUT)
                ZMQ.setsockopt(socket, ZMQ.SNDTIMEO, SOCKET_TIMEOUT)
            end
            
            # Bind socket
            ZMQCompat.bind(socket, SOCKET_ADDRESS)
            println("✓ Socket bound successfully")
            return (ctx, socket)
        catch e
            println("❌ Error setting socket options: $e")
            ZMQCompat.close(socket)
            ZMQCompat.close(ctx)
            rethrow(e)
        end
    catch e
        println("❌ Error creating socket: $e")
        rethrow(e)
    end
end

# Function to handle client requests with retries
function handle_request(socket)
    retry_count = 0
    while retry_count < MAX_RETRIES
        try
            # Receive request
            request = String(ZMQCompat.recv(socket))
            println("Received request of length: $(length(request))")
            
            # Parse JSON with error handling
            try
                data = JSON.parse(request)
                
                # Process the request and generate response
                response = process_observation(data)
                
                # Send response
                ZMQCompat.send(socket, JSON.json(response))
                println("✓ Request processed successfully")
                return true
            catch e
                println("❌ Error processing request: $e")
                ZMQCompat.send(socket, JSON.json(Dict("error" => "Invalid request format")))
                return false
            end
        catch e
            println("❌ Error in request handling (attempt $(retry_count + 1)/$MAX_RETRIES): $e")
            retry_count += 1
            if retry_count < MAX_RETRIES
                println("Retrying in $RETRY_DELAY seconds...")
                sleep(RETRY_DELAY)
            end
        end
    end
    println("❌ Maximum retries reached, closing socket")
    return false
end

# Main server loop with proper cleanup
function run_server()
    println("\nStarting ZMQ server loop...")
    
    # Create status file to indicate server is running
    if status_file_path !== nothing
        try
            touch(status_file_path)
            println("✓ Created status file")
        catch e
            println("❌ Could not create status file: $e")
        end
    end
    
    # Setup socket
    ctx, socket = setup_socket()
    
    try
        while true
            try
                if !handle_request(socket)
                    println("Request handling failed, resetting socket...")
                    ZMQCompat.close(socket)
                    ZMQCompat.close(ctx)
                    ctx, socket = setup_socket()
                end
            catch e
                println("❌ Error in server loop: $e")
                println("Resetting socket and continuing...")
                ZMQCompat.close(socket)
                ZMQCompat.close(ctx)
                ctx, socket = setup_socket()
            end
        end
    finally
        # Cleanup
        println("\nCleaning up server resources...")
        if status_file_path !== nothing
            try
                rm(status_file_path)
                println("✓ Removed status file")
            catch e
                println("❌ Could not remove status file: $e")
            end
        end
        ZMQCompat.close(socket)
        ZMQCompat.close(ctx)
        println("✓ Server resources cleaned up")
    end
end

# Start the server
println("\n=== Starting ZMQ Server ===")
run_server()