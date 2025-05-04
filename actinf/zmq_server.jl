#!/usr/bin/env julia

# ZeroMQ-based server for drone state inference and planning
# Replaces file-based communication with a direct ZMQ REP socket

println("Starting Active Inference ZMQ Server...")

# Activate the project environment first to ensure packages are available
println("Activating project environment...")
import Pkg
Pkg.activate(dirname(@__FILE__))
println("Project activated")

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
    
    # Directly import needed components from modules
    @eval using actinf.StateSpace: DroneState, DroneObservation, create_state_from_observation
    @eval using actinf.Inference: DroneBeliefs, initialize_beliefs, update_beliefs!, expected_state, serialize_beliefs, deserialize_beliefs
    @eval using actinf.Planning: ActionPlanner, PreferenceModel, select_action
    println("✓ Module components imported")
catch e
    println("❌ Failed to load actinf package: $e")
    println("Trying to include manually...")
    
    try
        # Include files directly
        include(joinpath(@__DIR__, "src", "StateSpace.jl"))
        include(joinpath(@__DIR__, "src", "Inference.jl"))
        include(joinpath(@__DIR__, "src", "Planning.jl"))
        include(joinpath(@__DIR__, "src", "actinf.jl"))
        
        # Import modules
        using .StateSpace
        using .Inference
        using .Planning
        println("✓ actinf modules loaded manually")
    catch e2
        println("❌ Failed to manually include actinf modules: $e2")
        global packages_loaded = false
    end
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

# Global socket and context variables
global zmq_ctx = nothing
global zmq_socket = nothing
global connection_initialized = false

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

# Function to process an observation and produce the next action
function process_observation(data::Dict)
    # ...existing code...
end

# Function to explicitly set up the ZMQ connection
function _setup_zmq_connection()
    global zmq_ctx, zmq_socket, connection_initialized
    
    # If already initialized, return early
    if connection_initialized
        println("ZMQ connection already initialized.")
        return true
    end
    
    println("Setting up ZMQ connection...")
    try
        zmq_ctx = ZMQ.Context()
        zmq_socket = ZMQ.Socket(zmq_ctx, ZMQ.REP)
        
        # Set socket options with proper error handling
        ZMQCompat.set_linger(zmq_socket, 0)
        ZMQCompat.set_rcvhwm(zmq_socket, 100)
        ZMQCompat.set_sndhwm(zmq_socket, 100)
        ZMQCompat.set_tcp_keepalive(zmq_socket, 1)
        
        # Set timeouts
        if ZMQCompat.using_new_api
            zmq_socket.rcvtimeo = SOCKET_TIMEOUT
            zmq_socket.sndtimeo = SOCKET_TIMEOUT
        else
            ZMQ.setsockopt(zmq_socket, ZMQ.RCVTIMEO, SOCKET_TIMEOUT)
            ZMQ.setsockopt(zmq_socket, ZMQ.SNDTIMEO, SOCKET_TIMEOUT)
        end
        
        # Bind socket
        ZMQCompat.bind(zmq_socket, SOCKET_ADDRESS)
        println("✓ ZMQ connection established successfully")
        
        # Mark as initialized
        connection_initialized = true
        return true
    catch e
        println("❌ Error setting up ZMQ connection: $e")
        
        # Cleanup on failure
        if zmq_socket !== nothing
            try ZMQCompat.close(zmq_socket); catch; end
            zmq_socket = nothing
        end
        
        if zmq_ctx !== nothing
            try ZMQCompat.close(zmq_ctx); catch; end
            zmq_ctx = nothing
        end
        
        return false
    end
end

"""
    run_zmq_server()

Run a ZMQ server that processes incoming observation data using active inference.
"""
function run_zmq_server()
    global zmq_ctx, zmq_socket
    
    # Make sure ZMQ connection is set up first
    if !_setup_zmq_connection()
        println("Failed to set up ZMQ connection. Exiting.")
        return
    end
    
    # Create status file if path is specified
    if status_file_path !== nothing
        touch(status_file_path)
        println("✓ Created status file: $status_file_path")
    end
    
    println("Starting ZMQ server loop")
    
    # For improved performance, avoid garbage collection pauses during request handling
    ccall(:jl_gc_enable, Void, (Cint,), 0)
    
    try
        while true
            # Wait for a message with proper error handling
            try
                # Receive a request
                request_json = ZMQCompat.recv_string(zmq_socket)
                
                # Parse JSON request
                request = JSON.parse(request_json)
                
                # Handle ping specially for health checks
                if haskey(request, "ping")
                    # Reply to ping with pong
                    response = Dict("pong" => true, "message" => "pong-response")
                    response_json = JSON.json(response)
                    ZMQCompat.send_string(zmq_socket, response_json)
                    continue
                end
                
                # Process the observation data
                response = process_observation(request)
                
                # Send the response back
                response_json = JSON.json(response)
                ZMQCompat.send_string(zmq_socket, response_json)
                
            catch e
                # Handle any errors during processing
                if isa(e, InterruptException)
                    # Allow Ctrl+C to terminate
                    rethrow(e)
                end
                
                println("Error processing request: $e")
                
                # Attempt to send error response
                try
                    error_response = Dict(
                        "error" => "Error processing request: $(typeof(e))",
                        "message" => string(e),
                        "action" => [0.0, 0.0, 0.0],  # Default action
                        "waypoint" => [0.0, 0.0, 0.0]  # Default waypoint
                    )
                    error_json = JSON.json(error_response)
                    ZMQCompat.send_string(zmq_socket, error_json)
                catch send_err
                    println("Failed to send error response: $send_err")
                end
            end
        end
    catch e
        if isa(e, InterruptException)
            println("Server terminated by interrupt")
        else
            println("Server error: $e")
        end
    finally
        # Clean up resources
        if zmq_socket !== nothing
            try ZMQCompat.close(zmq_socket); catch; end
            zmq_socket = nothing
        end
        
        if zmq_ctx !== nothing
            try ZMQCompat.close(zmq_ctx); catch; end
            zmq_ctx = nothing
        end
        
        # Remove status file on exit
        if status_file_path !== nothing && isfile(status_file_path)
            try
                rm(status_file_path)
                println("✓ Removed status file")
            catch e
                println("Failed to remove status file: $e")
            end
        end
        
        # Re-enable garbage collection
        ccall(:jl_gc_enable, Void, (Cint,), 1)
    end
end

# Run the server when this script is executed directly
println("ZMQ server starting...")
run_zmq_server()