import airsim
import numpy as np
import time
from math import pi
import msgpackrpc
from sklearn.cluster import DBSCAN
from collections import defaultdict
import json
import platform
import os
import subprocess
import traceback
import zmq
import threading
import signal
import sys
from datetime import datetime

# Add these new imports to support enhanced ZMQ functionality
import errno
import tempfile
import random

class EnvironmentScanner:
    def __init__(self, client=None):
        self.client = client or airsim.MultirotorClient()
        try:
            self.client.confirmConnection()
        except Exception as e:
            raise Exception(f"Failed to connect to AirSim: {str(e)}")

    # ... existing EnvironmentScanner methods ...

class ZMQInterface:
    """Interface for communicating with the Julia Active Inference server via ZMQ."""
    
    def __init__(self, server_address="tcp://localhost:5555", timeout=10000, auto_start=True):
        self.server_address = server_address
        self.timeout = timeout  # in milliseconds
        self.auto_start = auto_start
        self.context = None
        self.socket = None
        self.julia_process = None
        self.status_file_path = None
        self.heartbeat_thread = None
        self.running = False
        self.last_heartbeat_time = None
        self.heartbeat_interval = 5  # seconds between heartbeats
        self.connection_healthy = True
        self.setup_zmq()
        
        # Set up clean shutdown handling
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        if auto_start:
            self.ensure_server_running()
            # Start heartbeat thread after connection is established
            self.start_heartbeat_thread()
    
    def handle_shutdown(self, sig, frame):
        """Handle process termination signals to clean up resources"""
        print("Shutdown signal received. Cleaning up ZMQ resources...")
        self.cleanup()
        sys.exit(0)
    
    def setup_zmq(self):
        """Initialize ZMQ context and socket with appropriate settings"""
        try:
            if self.context is None:
                self.context = zmq.Context()
            
            if self.socket is not None:
                self.socket.close()
                
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 1000)  # Wait up to 1 second when closing
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)  # Receive timeout
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)  # Send timeout
            
            # Set TCP keepalive to detect disconnections
            self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)  # Seconds before sending keepalive
            self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 15)  # Interval between keepalives
            
            # Connect to the server
            self.socket.connect(self.server_address)
            print(f"Connected to ZMQ server at {self.server_address}")
            return True
        except Exception as e:
            print(f"Error setting up ZMQ connection: {str(e)}")
            return False
    
    def reset_socket(self):
        """Reset and reconnect the ZMQ socket"""
        try:
            print("Resetting ZMQ socket...")
            if self.socket:
                self.socket.close()
            
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 1000)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self.socket.connect(self.server_address)
            print("Socket reset successful")
            return True
        except Exception as e:
            print(f"Error resetting socket: {str(e)}")
            return False

    def start_heartbeat_thread(self):
        """Start a background thread to periodically check server health"""
        if self.heartbeat_thread is not None and self.heartbeat_thread.is_alive():
            print("Heartbeat thread already running")
            return
        
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
        print("Heartbeat monitoring started")
    
    def _heartbeat_worker(self):
        """Background worker to periodically send heartbeats to server"""
        while self.running:
            try:
                time.sleep(self.heartbeat_interval)
                
                # Only send heartbeat if we haven't communicated recently
                current_time = time.time()
                if (self.last_heartbeat_time is None or 
                    current_time - self.last_heartbeat_time > self.heartbeat_interval):
                    # Use a ping with randomized ID to avoid message conflicts
                    ping_id = random.randint(10000, 99999)
                    ping_result = self.ping_server(f"ping-{ping_id}")
                    
                    if ping_result:
                        self.last_heartbeat_time = current_time
                        if not self.connection_healthy:
                            print("Connection restored!")
                            self.connection_healthy = True
                    else:
                        self.connection_healthy = False
                        print("Server heartbeat failed, will retry...")
                        
                        # If heartbeats fail consistently, try to restart server
                        if (self.last_heartbeat_time is None or 
                            current_time - self.last_heartbeat_time > 3 * self.heartbeat_interval):
                            print("Multiple heartbeats failed, attempting to restore connection")
                            self.restore_connection()
            except Exception as e:
                print(f"Error in heartbeat worker: {str(e)}")
                # Brief pause before continuing
                time.sleep(1)
    
    def stop_heartbeat_thread(self):
        """Stop the heartbeat monitoring thread"""
        self.running = False
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2)
            print("Heartbeat monitoring stopped")
    
    def restore_connection(self):
        """Attempt to restore ZMQ connection, potentially restarting the server"""
        print("Attempting to restore ZMQ connection...")
        
        # First try resetting the socket without restarting server
        if self.reset_socket():
            # Try a ping after reset
            if self.ping_server("ping-reset"):
                print("Connection restored after socket reset")
                self.last_heartbeat_time = time.time()
                self.connection_healthy = True
                return True
        
        # If socket reset didn't work, try to restart the server
        print("Socket reset failed, attempting to restart server...")
        self.stop_server()
        time.sleep(2)  # Give time for server shutdown
        
        if self.start_server():
            # Wait for server to initialize
            time.sleep(5)
            
            # Reset socket and try to connect
            if self.reset_socket() and self.ping_server("ping-restart"):
                print("Connection restored after server restart")
                self.last_heartbeat_time = time.time()
                self.connection_healthy = True
                return True
        
        print("Failed to restore connection")
        return False
    
    def ping_server(self, ping_message="ping"):
        """Send a ping to the server to check connection health"""
        try:
            # Use a temporary socket for ping to avoid interfering with request-reply pattern
            ping_context = zmq.Context()
            ping_socket = ping_context.socket(zmq.REQ)
            ping_socket.setsockopt(zmq.LINGER, 500)  # Short linger time for ping
            ping_socket.setsockopt(zmq.RCVTIMEO, 2000)  # Short timeout (2 seconds)
            ping_socket.setsockopt(zmq.SNDTIMEO, 2000)  # Short timeout (2 seconds)
            
            # Connect to server
            ping_socket.connect(self.server_address)
            
            # Send ping as JSON request
            ping_request = {"ping": True, "message": ping_message}
            ping_socket.send_string(json.dumps(ping_request))
            
            # Wait for response with timeout
            response_json = ping_socket.recv_string()
            response = json.loads(response_json)
            
            # Clean up
            ping_socket.close()
            ping_context.term()
            
            return response.get("pong", False) or "pong" in response
        except Exception as e:
            print(f"Ping error: {str(e)}")
            try:
                ping_socket.close()
                ping_context.term()
            except:
                pass
            return False
    
    def find_julia_executable(self):
        """Find the Julia executable path"""
        # Check common locations
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\Julia\bin\julia.exe",
                r"C:\Program Files (x86)\Julia\bin\julia.exe",
                # Add user directory locations
                os.path.expanduser(r"~\AppData\Local\Programs\Julia\bin\julia.exe"),
                os.path.expanduser(r"~\AppData\Local\Julia\bin\julia.exe")
            ]
        else:  # Unix-like systems
            common_paths = [
                "/usr/bin/julia",
                "/usr/local/bin/julia",
                "/opt/julia/bin/julia",
                os.path.expanduser("~/julia/bin/julia")
            ]
        
        # Check PATH environment
        try:
            # Use where on Windows, which on POSIX
            finder_cmd = "where" if platform.system() == "Windows" else "which"
            result = subprocess.run([finder_cmd, "julia"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   shell=True)
            if result.returncode == 0 and result.stdout.strip():
                julia_path = result.stdout.strip().split('\n')[0]
                if os.path.exists(julia_path):
                    print(f"Found Julia in PATH: {julia_path}")
                    return julia_path
        except Exception as e:
            print(f"Error finding Julia in PATH: {str(e)}")
        
        # Check common locations
        for path in common_paths:
            if os.path.exists(path):
                print(f"Found Julia at: {path}")
                return path
        
        # Last resort: try just "julia" and hope it's in PATH
        return "julia"
    
    def find_server_script(self):
        """Find the path to the ZMQ server script"""
        # Start from the current file's directory and search upwards
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check in parent directory and sibling directories
        potential_locations = [
            os.path.join(current_dir, "..", "actinf", "zmq_server.jl"),
            os.path.join(current_dir, "..", "zmq_server.jl"),
            os.path.join(current_dir, "zmq_server.jl")
        ]
        
        for script_path in potential_locations:
            script_path = os.path.normpath(script_path)
            if os.path.exists(script_path):
                print(f"Found server script at: {script_path}")
                return script_path
        
        # If not found, try a more exhaustive search starting from parent directory
        parent_dir = os.path.dirname(current_dir)
        print(f"Searching for zmq_server.jl in {parent_dir}...")
        
        for root, dirs, files in os.walk(parent_dir):
            if "zmq_server.jl" in files:
                script_path = os.path.join(root, "zmq_server.jl")
                print(f"Found server script at: {script_path}")
                return script_path
        
        # Last resort - look in workspace parent
        workspace_parent = os.path.dirname(os.path.dirname(current_dir))
        for root, dirs, files in os.walk(workspace_parent):
            if "zmq_server.jl" in files:
                script_path = os.path.join(root, "zmq_server.jl")
                print(f"Found server script at: {script_path}")
                return script_path
        
        print("zmq_server.jl not found in any expected location")
        return None
    
    def find_status_file(self):
        """Find the potential location of the ZMQ server status file"""
        # Check in common locations
        candidates = []
        
        # Check in project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        candidates.append(os.path.join(parent_dir, "zmq_server_running.status"))
        
        # Check in system temp directory
        candidates.append(os.path.join(tempfile.gettempdir(), "zmq_server_running.status"))
        
        # On Windows, check additional locations
        if platform.system() == "Windows":
            candidates.append(os.path.expanduser(r"~\AppData\Local\Temp\zmq_server_running.status"))
        else:
            candidates.append("/tmp/zmq_server_running.status")
        
        # Return the first status file found or None
        for status_path in candidates:
            if os.path.exists(status_path):
                print(f"Found status file at: {status_path}")
                return status_path
        
        # If no existing file is found, return the most likely location for creation
        default_status_path = os.path.join(parent_dir, "zmq_server_running.status")
        print(f"Using default status file location: {default_status_path}")
        return default_status_path
    
    def is_server_running(self):
        """Check if the Julia ZMQ server is running by checking status file and ping"""
        # First check for the status file
        self.status_file_path = self.find_status_file()
        
        if self.status_file_path and os.path.exists(self.status_file_path):
            print(f"Found server status file: {self.status_file_path}")
            
            # Check file freshness (should be less than 5 minutes old)
            file_age = time.time() - os.path.getmtime(self.status_file_path)
            if file_age > 300:  # 5 minutes in seconds
                print(f"Status file is {file_age:.0f} seconds old, may be stale")
            
            # Even if status file exists, try to ping the server to confirm it's responsive
            if self.ping_server():
                print("Server is running and responsive")
                return True
            else:
                print("Status file exists but server is not responding")
                # Clean up stale status file
                try:
                    os.remove(self.status_file_path)
                    print("Removed stale status file")
                except Exception as e:
                    print(f"Error removing stale status file: {str(e)}")
                return False
        
        # If no status file, try direct ping as a final check
        print("No status file found, trying direct ping")
        if self.ping_server():
            print("Server is running despite no status file")
            return True
            
        return False
    
    def start_server(self):
        """Start the Julia ZMQ server"""
        try:
            # Find Julia executable and server script
            julia_path = self.find_julia_executable()
            script_path = self.find_server_script()
            
            if not script_path:
                print("Cannot start server: zmq_server.jl not found")
                return False
            
            # Determine correct working directory (directory containing the script)
            working_dir = os.path.dirname(script_path)
            
            # Build command
            command = [julia_path, script_path]
            
            print(f"Starting Julia server with command: {' '.join(command)}")
            print(f"Working directory: {working_dir}")
            
            # Start Julia process with appropriate stdio redirection
            # Using subprocess.PIPE for stdout/stderr to prevent blocking
            self.julia_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Start threads to read output asynchronously
            threading.Thread(target=self._read_output, 
                             args=(self.julia_process.stdout, "JULIA"), 
                             daemon=True).start()
            threading.Thread(target=self._read_output, 
                             args=(self.julia_process.stderr, "JULIA ERR"), 
                             daemon=True).start()
            
            # Wait a moment for process to start
            time.sleep(1)
            
            # Check if process is still running
            if self.julia_process.poll() is not None:
                print(f"Julia process exited with code {self.julia_process.returncode}")
                return False
            
            print("Julia server started successfully")
            
            # Wait for the server to initialize (up to 15 seconds)
            max_wait = 15
            for i in range(max_wait):
                if self.is_server_running() or self.ping_server():
                    print(f"Server is ready after {i+1} seconds")
                    time.sleep(1)  # Give one more second for full initialization
                    return True
                time.sleep(1)
                print(f"Waiting for server to initialize... {i+1}/{max_wait}")
            
            print("Timed out waiting for server to initialize")
            return False
            
        except Exception as e:
            print(f"Error starting Julia server: {str(e)}")
            traceback.print_exc()
            return False
    
    def _read_output(self, pipe, prefix):
        """Read output from subprocess pipe and print with prefix"""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] {prefix}: {line.rstrip()}")
        except Exception as e:
            print(f"Error reading {prefix} output: {str(e)}")
        finally:
            pipe.close()
    
    def stop_server(self):
        """Stop the Julia ZMQ server"""
        try:
            if self.julia_process:
                print("Stopping Julia server...")
                
                # Try to terminate gracefully first
                self.julia_process.terminate()
                
                # Wait a bit for graceful shutdown
                for _ in range(5):
                    if self.julia_process.poll() is not None:
                        print(f"Julia process terminated with code {self.julia_process.returncode}")
                        break
                    time.sleep(0.5)
                
                # If still running, force kill
                if self.julia_process.poll() is None:
                    print("Process didn't terminate gracefully, forcing kill")
                    self.julia_process.kill()
                    self.julia_process.wait(timeout=2)
                
                self.julia_process = None
                
                # Clean up status file if it exists
                if self.status_file_path and os.path.exists(self.status_file_path):
                    try:
                        os.remove(self.status_file_path)
                        print("Removed status file")
                    except Exception as e:
                        print(f"Error removing status file: {str(e)}")
                
                return True
            else:
                print("No Julia process to stop")
                return False
        except Exception as e:
            print(f"Error stopping Julia server: {str(e)}")
            return False
    
    def ensure_server_running(self):
        """Ensure the Julia ZMQ server is running, starting it if necessary"""
        if not self.is_server_running():
            print("ZMQ server not running, attempting to start it")
            return self.start_server()
        return True
    
    def send_request(self, request_data):
        """Send a request to the Julia server and get the response"""
        # Update last heartbeat time as we're sending a real request
        self.last_heartbeat_time = time.time()
        
        # Retry mechanism with exponential backoff
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Ensure server is running before sending request
                if not self.connection_healthy:
                    self.restore_connection()
                
                # Convert request data to JSON and send
                request_json = json.dumps(request_data)
                self.socket.send_string(request_json)
                
                # Receive response
                response_json = self.socket.recv_string()
                
                # Parse JSON response
                response = json.loads(response_json)
                
                # Update last heartbeat time on successful communication
                self.last_heartbeat_time = time.time()
                self.connection_healthy = True
                
                return response
                
            except zmq.error.Again as e:
                # Timeout error
                print(f"Request timed out (attempt {attempt+1}/{max_retries}): {str(e)}")
                self.connection_healthy = False
                
                # Reset socket on timeout
                self.reset_socket()
                
                # Retry with increasing delay
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)
                    print(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                
            except zmq.error.ZMQError as e:
                if e.errno == errno.ENOTSOCK:
                    print("Socket became invalid, resetting...")
                    self.reset_socket()
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                else:
                    print(f"ZMQ error (attempt {attempt+1}/{max_retries}): {str(e)}")
                    self.connection_healthy = False
                    
                    # Try to restore connection
                    self.restore_connection()
                    
                    if attempt < max_retries - 1:
                        sleep_time = retry_delay * (2 ** attempt)
                        print(f"Retrying in {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error sending request (attempt {attempt+1}/{max_retries}): {str(e)}")
                traceback.print_exc()
                self.connection_healthy = False
                
                # More severe error, try to restore connection
                self.restore_connection()
                
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)
                    print(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
        
        # If all retries failed, return default response
        print("All retry attempts failed")
        return {"error": "Communication with Julia server failed after multiple attempts", 
                "action": [0.0, 0.0, 0.0],
                "waypoint": [0.0, 0.0, 0.0]}
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up ZMQ resources...")
        
        # Stop heartbeat thread
        self.stop_heartbeat_thread()
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
                print("Socket closed")
            except Exception as e:
                print(f"Error closing socket: {str(e)}")
        
        # Terminate context
        if self.context:
            try:
                self.context.term()
                print("Context terminated")
            except Exception as e:
                print(f"Error terminating context: {str(e)}")
        
        # Stop Julia server if we started it
        if self.auto_start and self.julia_process:
            self.stop_server()
    
    def __del__(self):
        """Destructor to clean up resources"""
        self.cleanup()