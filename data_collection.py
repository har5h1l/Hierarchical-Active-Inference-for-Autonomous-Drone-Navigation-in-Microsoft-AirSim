import airsim
import numpy as np
import time
import os
import json
import subprocess
import sys
import zmq
import math
import platform
import traceback
import logging
import pandas as pd
import random
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from matplotlib import pyplot as plt
from os import path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Custom NumPy-aware JSON encoder
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyJSONEncoder, self).default(obj)

def collect_system_info() -> Dict[str, Any]:
    """Collect system and device information for experiment metadata
    
    Returns:
        Dict containing system information including OS, hardware, and software details
    """
    system_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os": {
            "name": platform.system(),
            "version": platform.version(),
            "platform": platform.platform(),
            "release": platform.release(),
            "architecture": platform.machine()
        }
    }
    
    # Python information
    system_info["python"] = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler()
    }
    
    # Julia information
    try:
        julia_path = "julia"  # Default command assuming Julia is in PATH
        julia_version_cmd = subprocess.run([julia_path, "--version"], 
                                          capture_output=True, text=True, timeout=5)
        if julia_version_cmd.returncode == 0:
            system_info["julia"] = {
                "version": julia_version_cmd.stdout.strip(),
                "path": julia_path
            }
            
            # Try to get package versions
            get_pkg_versions_cmd = [
                julia_path, 
                "-e", 
                "using Pkg; Pkg.status()"
            ]
            pkg_info = subprocess.run(get_pkg_versions_cmd, capture_output=True, text=True, timeout=10)
            if pkg_info.returncode == 0:
                system_info["julia"]["packages"] = pkg_info.stdout.strip()
    except Exception as e:
        system_info["julia"] = {"error": str(e)}
    
    # CPU information
    try:
        import psutil
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=0.1)
        }
        system_info["cpu"] = cpu_info
    except ImportError:
        # Fallback CPU info without psutil
        system_info["cpu"] = {
            "processor": platform.processor(),
        }
    
    # Memory information
    try:
        import psutil
        mem = psutil.virtual_memory()
        system_info["memory"] = {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_percent": mem.percent
        }
    except ImportError:
        system_info["memory"] = {"note": "psutil not available for detailed memory info"}
    
    # GPU information
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu_info = []
        
        for gpu in gpus:
            gpu_info.append({
                "id": gpu.id,
                "name": gpu.name,
                "memory_total": gpu.memoryTotal,
                "memory_used": gpu.memoryUsed,
                "load": gpu.load
            })
            
        system_info["gpu"] = gpu_info if gpu_info else {"note": "No GPU detected"}
    except ImportError:
        # Try with subprocess for NVIDIA GPUs
        try:
            nvidia_smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv"],
                                       capture_output=True, text=True, timeout=5)
            if nvidia_smi.returncode == 0:
                system_info["gpu"] = {"nvidia_smi_output": nvidia_smi.stdout.strip()}
            else:
                system_info["gpu"] = {"note": "No GPU information available"}
        except:
            system_info["gpu"] = {"note": "Could not detect GPU information"}
    
    # AirSim version - check if we can get it
    try:
        airsim_version = airsim.__version__ if hasattr(airsim, "__version__") else "Unknown"
        system_info["airsim"] = {"version": airsim_version}
    except:
        system_info["airsim"] = {"version": "Unknown"}
    
    # Additional library versions
    system_info["libraries"] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "zmq": zmq.pyzmq_version()
    }
    
    return system_info

# Constants and hyperparameters
TARGET_LOCATION = [-20.0, -20.0, -30.0]  # [x, y, z] in NED coordinates
MARGIN = 1.5  # Safety margin for waypoint generation (meters)
WAYPOINT_SAMPLE_COUNT = 75  # Number of waypoints to consider
POLICY_LENGTH = 3  # Number of steps in the policy
DENSITY_RADIUS = 5.0  # Radius for density evaluation
ARRIVAL_THRESHOLD = 1.5 # meters
MAX_ITERATIONS = 100
DEFAULT_ZMQ_PORT = 5555  # Default ZMQ server port
ZMQ_TIMEOUT = 10000  # ZMQ socket timeout in milliseconds (10 seconds)
ZMQ_MAX_RETRIES = 3  # Maximum number of retries for ZMQ communication

# Default experiment configuration
DEFAULT_CONFIG = {
    "num_episodes": 10,
    "target_distance_range": (15.0, 50.0),  # min, max in meters
    "random_seed": 42,
    "max_steps_per_episode": 100,
    "output_dir": "experiment_results",
    "stuck_timeout": 15.0,  # seconds to consider the drone stuck
    "stuck_check_interval": 3.0,  # seconds between stuck checks
    "episode_timeout": 120,  # maximum episode duration in seconds
    "precompile_julia": True,  # whether to precompile Julia packages
    "zmq_port": DEFAULT_ZMQ_PORT,  # ZMQ server port
    
    # Raycasting debug configuration
    "debug_raycasting": False,  # Enable debug mode for raycasts
    "min_ray_checks": 3,  # Minimum number of rays for target validation
    "max_ray_checks": 7,  # Maximum number of rays for target validation 
    "visualize_raycasts": False,  # Visualize raycasts in AirSim
    "visualization_duration": 60.0  # Duration of visualization markers in seconds
}

# Add the airsim directory to the Python path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'airsim'))

# Import the scanner
try:
    from Sensory_Input_Processing import EnvironmentScanner
except ImportError:
    from airsim.Sensory_Input_Processing import EnvironmentScanner

class JuliaServer:
    """Manages the Julia environment and ZMQ server for active inference"""
    
    def __init__(self):
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.server_process = None
        self.julia_path = self._find_julia_executable()
    
    def _find_julia_executable(self):
        """Find the Julia executable on the system"""
        print("Locating Julia executable...")
        
        # Default command assuming Julia is in PATH
        julia_path = "julia"
        
        if platform.system() == "Windows":
            # Windows-specific Julia paths
            possible_paths = [
                r"C:\Julia-1.9.3\bin\julia.exe",
                r"C:\Julia-1.9.2\bin\julia.exe",
                r"C:\Julia-1.9.1\bin\julia.exe",
                r"C:\Julia-1.9.0\bin\julia.exe",
                r"C:\Julia-1.8.5\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.3\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.2\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.1\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.9.0\bin\julia.exe",
                r"C:\Users\%USERNAME%\AppData\Local\Programs\Julia-1.8.5\bin\julia.exe"
            ]
            
            # Expand %USERNAME% environment variable
            username = os.environ.get('USERNAME', '')
            possible_paths = [p.replace('%USERNAME%', username) for p in possible_paths]
            
            # Check each path
            for path in possible_paths:
                if os.path.exists(path):
                    julia_path = path
                    print(f"Found Julia at: {julia_path}")
                    break
        
        # Test if Julia is actually available
        try:
            result = subprocess.run([julia_path, "--version"], 
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"Julia version: {result.stdout.strip()}")
            else:
                print(f"Julia executable found but returned error: {result.stderr}")
        except Exception as e:
            print(f"Warning: Error testing Julia executable: {e}")
        
        return julia_path
    
    def precompile_packages(self):
        """Precompile Julia packages to speed up server startup"""
        print("\n==== Precompiling Julia Packages ====")
        
        # Check if precompilation has already been done
        precomp_success_flag = os.path.join(self.cwd, ".precompilation_success")
        if os.path.exists(precomp_success_flag):
            print(f"✅ Precompilation success flag found. Removing flag to force precompilation.")
            try:
                os.remove(precomp_success_flag)
            except Exception as e:
                print(f"Failed to remove precompilation flag: {e}")
        
        # Run precompile script
        precompile_script = os.path.join(self.cwd, "precompile.jl")
        if not os.path.exists(precompile_script):
            print(f"❌ Precompilation script not found at {precompile_script}")
            return False
        
        print("Running Julia precompilation (this may take a few minutes)...")
        try:
            # Use a more reliable process management approach
            if platform.system() == "Windows":
                # Windows-specific precompilation
                precompile_log = os.path.join(self.cwd, "julia_precompile.log")
                
                # Create a visible window process
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 1  # SW_SHOWNORMAL
                
                with open(precompile_log, "wb") as log_file:  # Use binary mode to avoid encoding issues
                    process = subprocess.Popen(
                        [self.julia_path, precompile_script],
                        cwd=self.cwd,
                        stdout=log_file,
                        stderr=log_file,
                        startupinfo=startupinfo
                    )
                
                # Wait with a reasonable timeout
                try:
                    print("Waiting for precompilation to complete (timeout: 10 min)...")
                    return_code = process.wait(timeout=600)
                    
                    if return_code == 0:
                        print("✅ Precompilation completed successfully")
                        return True
                    else:
                        print(f"⚠️ Precompilation returned code {return_code}")
                        
                        # Show the log tail with proper encoding handling
                        try:
                            with open(precompile_log, 'rb') as f:  # Use binary mode
                                log_data = f.read()
                                # Try different encodings to handle special characters
                                encodings = ['utf-8', 'latin-1', 'cp1252']
                                log_text = None
                                
                                for encoding in encodings:
                                    try:
                                        log_text = log_data.decode(encoding, errors='replace')
                                        break
                                    except Exception:
                                        continue
                                
                                if log_text:
                                    log_lines = log_text.splitlines()[-10:]
                                    print("Last 10 lines of precompilation log:")
                                    for line in log_lines:
                                        print(f"  {line}")
                        except Exception as e:
                            print(f"Could not read precompilation log: {e}")
                        
                        # Check status file which is JSON and should be more reliable
                        status_file = os.path.join(self.cwd, ".precompilation_status.json")
                        if os.path.exists(status_file):
                            try:
                                with open(status_file, 'r') as f:
                                    status_data = json.load(f)
                                    print(f"Precompilation status: {status_data.get('status')} - {status_data.get('message')}")
                            except Exception as e:
                                print(f"Could not read status file: {e}")
                                
                        # If success flag exists, consider it successful despite return code
                        if os.path.exists(precomp_success_flag):
                            print("✅ Success flag found despite return code. Continuing.")
                            return True
                            
                        return False
                except subprocess.TimeoutExpired:
                    print("⚠️ Precompilation timed out after 10 minutes")
                    process.terminate()
                    return False
            else:
                # Unix approach
                result = subprocess.run(
                    [self.julia_path, precompile_script],
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    encoding='latin-1',  # Use latin-1 encoding which can handle all byte values
                    timeout=600
                )
                
                if result.returncode == 0:
                    print("✅ Precompilation completed successfully")
                    return True
                else:
                    print(f"⚠️ Precompilation failed with code {result.returncode}")
                    print(f"Output: {result.stderr[:500]}")
                    
                    # Check for success flag anyway
                    if os.path.exists(precomp_success_flag):
                        print("✅ Success flag found despite return code. Continuing.")
                        return True
                        
                    return False
                
        except Exception as e:
            print(f"❌ Error during precompilation: {e}")
            traceback.print_exc()
            
            # Check for success flag anyway
            if os.path.exists(precomp_success_flag):
                print("✅ Success flag found despite exception. Continuing.")
                return True
                
            return False
    
    def start_server(self, port=DEFAULT_ZMQ_PORT):
        """Start the ZMQ server"""
        print("\n==== Starting ZMQ Server ====")
        
        # Check if server is already running
        server_running_flag = os.path.join(self.cwd, ".zmq_server_running")
        server_status_file = os.path.join(self.cwd, ".zmq_server_status.json")
        
        # First attempt to ping an already running server to verify it's truly responding
        if os.path.exists(server_running_flag) and os.path.exists(server_status_file):
            try:
                with open(server_status_file, 'r') as f:
                    status_data = json.load(f)
                    if status_data.get('status') == 'running':
                        port = status_data.get('port', DEFAULT_ZMQ_PORT)
                        print(f"Found server status file indicating a running server on port {port}")
                        
                        # Try to connect and ping the server to verify it's actually responding
                        print("Verifying server is responsive...")
                        try:
                            # Quick connection test
                            context = zmq.Context()
                            socket = context.socket(zmq.REQ)
                            socket.setsockopt(zmq.LINGER, 500)
                            socket.setsockopt(zmq.RCVTIMEO, 2000)  # Short timeout for ping
                            socket.setsockopt(zmq.SNDTIMEO, 2000)
                            socket.connect(f"tcp://localhost:{port}")
                            
                            # Send ping and wait for response
                            socket.send_string("ping")
                            response = socket.recv_string()
                            
                            if response == "pong":
                                print(f"✅ Existing ZMQ server is responsive on port {port}")
                                socket.close()
                                context.term()
                                return port
                            else:
                                print(f"⚠️ Unexpected response from server: {response}")
                        except Exception as e:
                            print(f"⚠️ Existing server not responsive: {e}")
                        finally:
                            try:
                                socket.close()
                                context.term()
                            except:
                                pass
                        
                        # If we get here, the server wasn't responsive
                        print("Detected a non-responsive server. Will restart it.")
                        # Clean up stale server flags
                        if os.path.exists(server_running_flag):
                            os.remove(server_running_flag)
                        # No return here - fall through to start a new server
            except Exception as e:
                print(f"Error reading server status: {e}")
                # Clean up potentially corrupted files
                for file in [server_running_flag, server_status_file]:
                    if os.path.exists(file):
                        try:
                            os.remove(file)
                        except:
                            pass
        
        # Find the server script
        server_script = os.path.join(self.cwd, "actinf", "zmq_server.jl")
        if not os.path.exists(server_script):
            print(f"❌ Server script not found at {server_script}")
            return None
        
        print(f"Starting ZMQ server from {server_script}...")
        
        try:
            # Make sure any existing server process is terminated before starting a new one
            if hasattr(self, 'server_process') and self.server_process:
                try:
                    if self.server_process.poll() is None:  # Still running
                        print("Terminating previous server process...")
                        self.server_process.terminate()
                        try:
                            self.server_process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            self.server_process.kill()
                            print("Force-killed previous server process")
                except:
                    pass
            
            # Julia command with additional flags to address the signal handler issue
            julia_cmd = [
                self.julia_path,
                "--project=.",  # Use project in current directory
                "--startup-file=no",  # Don't load startup file
                "--handle-signals=no",  # Disable Julia's signal handling
                server_script
            ]
            
            # Add environment variables to configure ZMQ server
            env = os.environ.copy()
            env["JULIA_ZMQ_PORT"] = str(port)
            env["JULIA_DEBUG"] = "ZMQServer"  # Enable debug logging for the server
            
            print(f"Starting Julia with command: {' '.join(julia_cmd)}")
            
            # Start the server as a background process with a visible console for easier debugging
            if platform.system() == "Windows":
                # Windows needs different flags to start detached with visible window
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 1  # SW_SHOWNORMAL
                
                self.server_process = subprocess.Popen(
                    julia_cmd,
                    cwd=self.cwd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    startupinfo=startupinfo,
                    env=env
                )
            else:
                # Unix-like systems
                self.server_process = subprocess.Popen(
                    julia_cmd,
                    cwd=self.cwd,
                    start_new_session=True,
                    env=env
                )
            
            # Wait for server to start and check status file for port
            print("Waiting for server to initialize...")
            
            # Try several times to get the server port from status file
            max_attempts = 15  # Increased timeout for server startup
            server_port = None
            
            for attempt in range(max_attempts):
                time.sleep(1)  # Wait a bit between checks
                
                if os.path.exists(server_status_file):
                    try:
                        with open(server_status_file, 'r') as f:
                            status_data = json.load(f)
                            if status_data.get('status') == 'running':
                                server_port = status_data.get('port', DEFAULT_ZMQ_PORT)
                                print(f"✅ ZMQ server started successfully on port {server_port}")
                                
                                # Verify server is actually responsive with a ping
                                try:
                                    context = zmq.Context()
                                    socket = context.socket(zmq.REQ)
                                    socket.setsockopt(zmq.LINGER, 500)
                                    socket.setsockopt(zmq.RCVTIMEO, 2000)
                                    socket.setsockopt(zmq.SNDTIMEO, 2000)
                                    socket.connect(f"tcp://localhost:{server_port}")
                                    
                                    socket.send_string("ping")
                                    response = socket.recv_string()
                                    
                                    if response == "pong":
                                        print(f"✅ ZMQ server verified responsive on port {server_port}")
                                        socket.close()
                                        context.term()
                                        break
                                    else:
                                        print(f"⚠️ Unexpected response from server: {response}")
                                except Exception as e:
                                    print(f"⚠️ Server not immediately responsive, will retry: {e}")
                                finally:
                                    try:
                                        socket.close()
                                        context.term()
                                    except:
                                        pass
                    except Exception as e:
                        print(f"Error reading status file (attempt {attempt+1}): {e}")
                
                # Check if process is still running
                if self.server_process.poll() is not None:
                    print(f"❌ Server process exited with code {self.server_process.returncode}")
                    # Try to read any error logs
                    log_file = os.path.join(self.cwd, "zmq_server.log")
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r') as f:
                                log_tail = f.readlines()[-20:]  # Last 20 lines
                                print("Server log tail:")
                                for line in log_tail:
                                    print(f"  {line.strip()}")
                        except Exception as log_e:
                            print(f"Error reading server log: {log_e}")
                    return None
                
                print(f"Waiting for server to start... (attempt {attempt+1}/{max_attempts})")
            
            # Return the port (or default if not found)
            return server_port or DEFAULT_ZMQ_PORT
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            traceback.print_exc()
            return None
    
    def shutdown(self):
        """Shutdown the server if running"""
        if self.server_process and self.server_process.poll() is None:
            print("Shutting down ZMQ server...")
            try:
                # Try to terminate gracefully first
                self.server_process.terminate()
                
                # Wait a bit to let it close
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.server_process.kill()
                    print("Server process had to be forcefully terminated")
            except Exception as e:
                print(f"Error shutting down server: {e}")

class ZMQInterface:
    """Interface for communicating with the Julia Active Inference server via ZMQ"""
    
    def __init__(self, server_port=DEFAULT_ZMQ_PORT):
        """Initialize ZMQ connection to Julia server"""
        self.server_address = f"tcp://localhost:{server_port}"
        print(f"Initializing ZMQ interface: {self.server_address}")
        
        # Initialize socket
        self.context = None
        self.socket = None
        
        # Connect to the server
        self._setup_connection()
    
    def _setup_connection(self):
        """Set up the ZMQ connection with proper error handling"""
        try:
            # Close existing connections first
            self._close_connection()
            
            # Create new context and socket
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            
            # Configure socket options
            self.socket.setsockopt(zmq.LINGER, 1000)  # Wait up to 1 second when closing
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT)  # Receive timeout
            self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT)  # Send timeout
            self.socket.setsockopt(zmq.REQ_RELAXED, 1)  # More relaxed REQ socket behavior
            self.socket.setsockopt(zmq.REQ_CORRELATE, 1)  # Correlate replies with requests
            
            # Connect to server
            print(f"Connecting to server at {self.server_address}...")
            self.socket.connect(self.server_address)
            
            # Test connection with ping
            print("Testing connection with ping...")
            try:
                self.socket.send_string("ping")
                poller = zmq.Poller()
                poller.register(self.socket, zmq.POLLIN)
                
                if poller.poll(5000):  # 5 second timeout for first ping
                    response = self.socket.recv_string()
                    if response == "pong":
                        print("✅ ZMQ connection established successfully")
                        return True
                    else:
                        print(f"⚠️ Unexpected ping response: {response}")
                else:
                    print("⚠️ No response to ping")
            except Exception as e:
                print(f"⚠️ Error during ping test: {e}")
            
            return False
        
        except Exception as e:
            print(f"❌ Failed to setup ZMQ connection: {e}")
            self._close_connection()
            return False
    
    def _close_connection(self):
        """Close the ZMQ connection cleanly"""
        if hasattr(self, 'socket') and self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        if hasattr(self, 'context') and self.context:
            try:
                self.context.term()
            except:
                pass
            self.context = None
    
    def send_observation_and_receive_action(self, observation):
        """Send observation data to Julia and receive next waypoint
        
        Args:
            observation: Dict containing drone observation data
                
        Returns:
            tuple: (next_waypoint, policy) or (None, None) if communication failed
        """
        try:
            print("\nSending observation to Active Inference engine...")
            
            # Validate the observation data to prevent sending invalid data
            # Check for NaN or Inf values in drone_position and target_position
            for key in ["drone_position", "target_position"]:
                if key in observation:
                    pos = observation[key]
                    if any(np.isnan(val) or np.isinf(val) for val in pos):
                        print(f"❌ Invalid {key} with NaN/Inf values: {pos}")
                        # Replace with a valid default if needed
                        if key == "drone_position":
                            observation[key] = [0.0, 0.0, -5.0]  # Default position
                        else:
                            observation[key] = [10.0, 0.0, -5.0]  # Default target
                        print(f"✓ Replaced with default: {observation[key]}")
            
            # Add a request type field to make the protocol more explicit
            observation["request_type"] = "get_waypoint"
            observation["client_timestamp"] = time.time()
            
            # Convert observation to JSON string
            obs_json = json.dumps(observation, cls=NumpyJSONEncoder)
            
            # Send with retry mechanism
            for retry in range(ZMQ_MAX_RETRIES):
                try:
                    # Reset socket on first retry and after errors
                    if retry > 0:
                        print(f"Retry {retry}/{ZMQ_MAX_RETRIES}: Resetting connection...")
                        self._setup_connection()
                        time.sleep(1)  # Give time for reconnection
                    
                    # Send observation data
                    self.socket.send_string(obs_json)
                    print(f"Sent observation data (attempt {retry+1}/{ZMQ_MAX_RETRIES})")
                    
                    # Wait for response
                    poller = zmq.Poller()
                    poller.register(self.socket, zmq.POLLIN)
                    
                    if poller.poll(ZMQ_TIMEOUT):
                        # Receive and parse response
                        response = self.socket.recv_string()
                        print(f"Received response ({len(response)} bytes)")
                        
                        # Validate the response to handle invalid JSON
                        try:
                            result = json.loads(response)
                        except json.JSONDecodeError as e:
                            print(f"❌ Invalid JSON response: {e}")
                            print(f"Response preview: {response[:100]}...")
                            continue  # Try again
                        
                        # Check for error message in response
                        if "error" in result:
                            print(f"❌ Server error: {result.get('message', 'Unknown error')}")
                            continue  # Try again
                        
                        # Extract waypoint and policy
                        waypoint = result.get("next_waypoint")
                        policy = result.get("policy", [])
                        
                        # Validate the waypoint
                        if waypoint is None or len(waypoint) != 3:
                            print(f"❌ Invalid waypoint format: {waypoint}")
                            continue  # Try again
                        
                        # Check for NaN or Inf in waypoint
                        if any(np.isnan(p) or np.isinf(p) for p in waypoint):
                            print(f"❌ Waypoint contains NaN/Inf values: {waypoint}")
                            continue  # Try again
                        
                        print(f"Next waypoint: {[round(p, 2) for p in waypoint]}")
                        return waypoint, policy
                    else:
                        print(f"Response timeout (attempt {retry+1}/{ZMQ_MAX_RETRIES})")
                        
                except zmq.ZMQError as e:
                    print(f"ZMQ error (attempt {retry+1}/{ZMQ_MAX_RETRIES}): {e}")
                except Exception as e:
                    print(f"Unexpected error (attempt {retry+1}/{ZMQ_MAX_RETRIES}): {e}")
                    traceback.print_exc()
            
            # If we get here, all retries failed
            print("❌ Failed to get valid response after all retries")
            
            # Last resort: try a simple ping to check if server is still responsive
            try:
                print("Sending ping to check server status...")
                self.socket.close()
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.RCVTIMEO, 2000)  # Short timeout for ping
                self.socket.connect(self.server_address)
                self.socket.send_string("ping")
                
                response = self.socket.recv_string()
                if response == "pong":
                    print("✅ Server is responsive to ping but failed to process observation")
                else:
                    print(f"❓ Server responded to ping with: {response}")
            except Exception as e:
                print(f"❌ Server is not responsive to ping: {e}")
            
            # Generate a simple fallback waypoint in the general target direction
            try:
                drone_pos = observation.get("drone_position", [0, 0, -5])
                target_pos = observation.get("target_position", [10, 0, -5])
                
                # Calculate direction to target
                direction = np.array(target_pos) - np.array(drone_pos)
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    # Normalize and scale to a small step (1-3 meters)
                    direction = direction / distance
                    step_size = min(distance / 2, 3.0)
                    fallback_waypoint = (np.array(drone_pos) + direction * step_size).tolist()
                    
                    print(f"⚠️ Using fallback waypoint: {[round(p, 2) for p in fallback_waypoint]}")
                    return fallback_waypoint, []
            except Exception as e:
                print(f"❌ Failed to generate fallback waypoint: {e}")
            
            return None, None
                
        except Exception as e:
            print(f"❌ Unhandled error in active inference processing: {e}")
            traceback.print_exc()
            return None, None
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self._close_connection()

class Scanner:
    """Scanner for obstacle detection using AirSim's LiDAR data"""
    
    def __init__(self, client, scan_range=20.0):
        self.client = client
        self.scan_range = scan_range
        
        # Enhanced parameters for better obstacle detection
        self.point_density = 1.2  # Increased point density for better resolution (was implicit default)
        self.min_points_per_voxel = 2  # Minimum points needed to consider a voxel as obstacle
        
        # Cache for recent scans to optimize performance during rapid rescanning
        self.last_scan_time = 0
        self.scan_cache_ttl = 0.1  # 100ms TTL for cached scan data
        self.cached_positions = []
        self.cached_distances = []
        
        # Parameters for directional scanning (used in real-time path monitoring)
        self.fov_angle = 120  # Field of view angle in degrees for forward scan
    def fetch_density_distances(self, use_cache=False):
        """Get obstacle positions and distances with orientation-aware transformation
        
        Args:
            use_cache: Whether to use cached results if available within TTL
            
        Returns:
            tuple: (obstacle_positions, obstacle_distances)
        """
        try:
            current_time = time.time()
            
            # Check if we can use cached results
            if use_cache and current_time - self.last_scan_time < self.scan_cache_ttl:
                return self.cached_positions, self.cached_distances
            
            # Initialize empty lists for return values
            obstacle_positions = []
            obstacle_distances = []

            # Get drone state
            drone_state = self.client.getMultirotorState().kinematics_estimated
            drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
            drone_orientation = [
                drone_state.orientation.w_val,
                drone_state.orientation.x_val, 
                drone_state.orientation.y_val,
                drone_state.orientation.z_val
            ]
            
            # Validate drone position
            if any(np.isnan(x) or np.isinf(x) for x in drone_pos):
                print("⚠️ Invalid drone position detected, using default [0,0,0]")
                drone_pos = [0.0, 0.0, 0.0]
            
            # Get lidar data
            lidar_data = self.client.getLidarData()
            
            if len(lidar_data.point_cloud) < 3:
                print("No lidar points detected")
                return [], []
            
            # Convert point cloud to positions
            try:
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            except (ValueError, TypeError) as e:
                print(f"⚠️ Error reshaping point cloud: {e}")
                return [], []

            # Quaternion utility functions
            def quaternion_multiply(q1, q2):
                w1, x1, y1, z1 = q1
                w2, x2, y2, z2 = q2
                w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                return [w, x, y, z]
            
            def quaternion_conjugate(q):
                w, x, y, z = q
                return [w, -x, -y, -z]
            
            def rotate_point_by_quaternion(point, quaternion):
                # Convert point to quaternion form
                point_quat = [0.0, point[0], point[1], point[2]]
                q_conj = quaternion_conjugate(quaternion)
                rotated = quaternion_multiply(
                    quaternion_multiply(quaternion, point_quat),
                    q_conj
                )
                return rotated[1:4]

            # Initialize voxel grid for improved point cloud processing
            # Decreased voxel size for better resolution
            voxel_size = 0.2  # Smaller voxels (in meters) for more precise detection (was implicitly larger)
            voxel_grid = {}
            
            # Process each point, placing them into voxels
            for point in points:
                # Skip invalid points
                if np.isnan(point).any() or np.isinf(point).any():
                    continue
                
                try:
                    # Transform point to global reference frame
                    global_point = rotate_point_by_quaternion(point, drone_orientation)
                    
                    # Add drone position
                    global_point = [
                        global_point[0] + drone_pos[0],
                        global_point[1] + drone_pos[1],
                        global_point[2] + drone_pos[2]
                    ]
                    
                    # Calculate distance
                    dist = np.sqrt(np.sum((np.array(drone_pos) - np.array(global_point)) ** 2))
                    
                    # Skip invalid distances
                    if np.isnan(dist) or np.isinf(dist):
                        continue
                    
                    # Only include obstacles within scan range
                    if dist <= self.scan_range and dist > 0.0:
                        # Calculate voxel index (discretize position)
                        voxel_idx = (
                            int(global_point[0] / voxel_size),
                            int(global_point[1] / voxel_size),
                            int(global_point[2] / voxel_size)
                        )
                        
                        # Add to voxel grid, keeping track of points and min distance
                        if voxel_idx in voxel_grid:
                            voxel_grid[voxel_idx]["count"] += 1
                            voxel_grid[voxel_idx]["min_dist"] = min(voxel_grid[voxel_idx]["min_dist"], dist)
                            # Update centroid with running average
                            current_count = voxel_grid[voxel_idx]["count"]
                            voxel_grid[voxel_idx]["position"] = [
                                (voxel_grid[voxel_idx]["position"][0] * (current_count-1) + global_point[0]) / current_count,
                                (voxel_grid[voxel_idx]["position"][1] * (current_count-1) + global_point[1]) / current_count,
                                (voxel_grid[voxel_idx]["position"][2] * (current_count-1) + global_point[2]) / current_count
                            ]
                        else:
                            voxel_grid[voxel_idx] = {
                                "count": 1,
                                "position": global_point,
                                "min_dist": dist
                            }
                except Exception:
                    continue
            
            # Extract obstacle positions and distances from voxels with sufficient density
            for voxel_data in voxel_grid.values():
                # Only consider voxels with enough points to be actual obstacles
                if voxel_data["count"] >= self.min_points_per_voxel:
                    obstacle_positions.append(voxel_data["position"])
                    obstacle_distances.append(voxel_data["min_dist"])
            
            # Update cache
            self.cached_positions = obstacle_positions
            self.cached_distances = obstacle_distances
            self.last_scan_time = current_time
            
            return obstacle_positions, obstacle_distances
        
        except Exception as e:
            print(f"Error in fetch_density_distances: {e}")
            traceback.print_exc()
            return [], []
    def scan_path_for_obstacles(self, start_pos, end_pos, safety_radius=3.0):
        """Scan specifically for obstacles along a planned movement path
        
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            safety_radius: Safety radius around path to check for obstacles (meters)
            
        Returns:
            tuple: (obstacle_detected, obstacle_position, obstacle_distance)
        """
        try:
            # Quick line of sight check using AirSim's built-in function first
            start_vector = airsim.Vector3r(start_pos[0], start_pos[1], start_pos[2])
            end_vector = airsim.Vector3r(end_pos[0], end_pos[1], end_pos[2])
            
            # Get drone's current orientation to properly account for forward-facing sensors
            try:
                drone_orientation = self.client.simGetVehiclePose().orientation
                current_yaw = airsim.to_eularian_angles(drone_orientation)[2]
                forward_direction = np.array([
                    math.cos(current_yaw),
                    math.sin(current_yaw),
                    0.0  # Keep in horizontal plane
                ])
            except Exception as e:
                logging.warning(f"Failed to get drone orientation for path scanning: {e}")
                forward_direction = None
            
            # This is much faster than processing point clouds for initial check
            los_clear = self.client.simTestLineOfSightBetweenPoints(start_vector, end_vector)
            
            # If line of sight is blocked, we have a definite obstacle
            if not los_clear:
                # Try to locate the obstacle more precisely using raycasting
                try:
                    # Cast multiple rays to better pinpoint obstacle location
                    movement_vector = np.array(end_pos) - np.array(start_pos)
                    movement_dist = np.linalg.norm(movement_vector)
                    
                    # Normalize to get direction
                    if movement_dist > 0:
                        direction = movement_vector / movement_dist
                    else:
                        direction = np.array([1.0, 0.0, 0.0])
                    
                    # Create test points along path
                    num_test_points = min(8, max(4, int(movement_dist / 1.5)))  # Increased number of test points
                    distances = np.linspace(0.5, movement_dist - 0.5, num_test_points)
                    
                    # Check each test point
                    for dist in distances:
                        test_point = np.array(start_pos) + direction * dist
                        test_vector = airsim.Vector3r(test_point[0], test_point[1], test_point[2])
                        
                        if not self.client.simTestLineOfSightBetweenPoints(start_vector, test_vector):
                            # Found obstacle - return estimated position and distance
                            obstacle_pos = test_point.tolist()
                            obstacle_dist = dist
                            return True, obstacle_pos, obstacle_dist
                    
                    # If no specific obstacle found but LOS check failed, use middle of path as estimate
                    middle_point = np.array(start_pos) + direction * (movement_dist / 2)
                    return True, middle_point.tolist(), movement_dist / 2
                except Exception as e:
                    # If precise location fails, log the error and just report an obstacle exists
                    logging.warning(f"Error during precise obstacle location: {e}")
                    return True, None, None
            
            # If built-in LOS check passed, do a more detailed check using point cloud for nearby obstacles
            obstacle_positions, obstacle_distances = self.fetch_density_distances(use_cache=True)
            
            if not obstacle_positions:
                return False, None, None  # No obstacles detected
            
            # Calculate path vector
            path_vector = np.array(end_pos) - np.array(start_pos)
            path_length = np.linalg.norm(path_vector)
            
            if path_length < 0.001:  # Effectively not moving
                return False, None, None
                
            path_direction = path_vector / path_length
            
            # If we have forward direction available, calculate angle between path and forward direction
            path_aligned_with_forward = True
            if forward_direction is not None:
                angle_cos = np.dot(path_direction, forward_direction)
                # If path is roughly aligned with forward direction (within 45 degrees)
                # use a smaller safety radius since our sensors are more accurate forward
                path_aligned_with_forward = angle_cos > 0.7  # cos(45°) ≈ 0.7
            
            # Check each obstacle to see if it's close to our path
            closest_obstacle_pos = None
            closest_obstacle_dist = float('inf')
            closest_perp_dist = float('inf')
            
            for i, obstacle_pos in enumerate(obstacle_positions):
                obstacle_vector = np.array(obstacle_pos) - np.array(start_pos)
                
                # Project obstacle vector onto path direction
                projection = np.dot(obstacle_vector, path_direction)
                  
                # Only consider obstacles that are ahead of us along the path
                # and within the path length (not past the destination)
                if 0 <= projection <= path_length:
                    # Calculate perpendicular distance from obstacle to path
                    closest_point = np.array(start_pos) + projection * path_direction
                    perpendicular_dist = np.linalg.norm(np.array(obstacle_pos) - closest_point)
                    
                    # Adaptive safety check: Use smaller safety radius for obstacles
                    # that are farther along the path, as they're less likely to be
                    # directly in our way when sensors are forward-facing
                    
                    # Calculate how far along the path this obstacle is (0.0 to 1.0)
                    path_fraction = projection / path_length if path_length > 0 else 0
                    
                    # Calculate effective safety radius based on several factors
                    effective_safety_radius = safety_radius
                    
                    # 1. Adjust based on path alignment with forward direction
                    if path_aligned_with_forward:
                        # If path is aligned with our sensors, we can be more precise
                        effective_safety_radius *= 0.9  # 10% smaller radius for better precision
                    else:
                        # If path is not aligned with our sensors, be more conservative
                        effective_safety_radius *= 1.1  # 10% larger radius for safety
                    
                    # 2. Adjust based on location along path
                    if path_fraction > 0.5:
                        # For obstacles in the second half of the path, reduce safety radius
                        # This is especially important with forward-facing sensors
                        distance_factor = 1.0 - ((path_fraction - 0.5) * 0.7)  # 1.0 at path_fraction=0.5, down to 0.65 at path_fraction=1.0
                        effective_safety_radius *= distance_factor
                    
                    # If obstacle is within effective safety radius of path
                    if perpendicular_dist <= effective_safety_radius:
                        # For distant obstacles, check if they're actually directly in our path
                        # by looking at the angle between our direction and the obstacle
                        if path_fraction > 0.7 and projection > 3.0:  # Only for far obstacles
                            # Get vector from start to obstacle
                            to_obstacle_vec = np.array(obstacle_pos) - np.array(start_pos)
                            to_obstacle_dist = np.linalg.norm(to_obstacle_vec)
                            if to_obstacle_dist > 0.001:
                                to_obstacle_vec = to_obstacle_vec / to_obstacle_dist
                                
                                # Calculate angle between path direction and obstacle direction
                                angle_cos = np.dot(path_direction, to_obstacle_vec)
                                
                                # If angle is too large, this is likely a side obstacle that won't be in our way
                                if angle_cos < 0.85:  # About 30 degrees
                                    continue
                        
                        # Keep track of the closest obstacle to the path
                        obstacle_dist = obstacle_distances[i]
                        if perpendicular_dist < closest_perp_dist:
                            closest_perp_dist = perpendicular_dist
                            closest_obstacle_pos = obstacle_pos
                            closest_obstacle_dist = obstacle_dist
            
            # If we found an obstacle too close to the path, return it
            if closest_obstacle_pos is not None:
                return True, closest_obstacle_pos, closest_obstacle_dist
            
            # No obstacles detected near the path
            return False, None, None
            
        except Exception as e:
            logging.error(f"Error in scan_path_for_obstacles: {e}")
            traceback.print_exc()
            return False, None, None
            
    def scan_in_direction(self, direction, max_distance=10.0, cone_angle=45.0):
        """Scan for obstacles in a specific direction with a cone-shaped field of view
        
        Args:
            direction: Direction vector [x, y, z]
            max_distance: Maximum distance to scan
            cone_angle: Half-angle of the cone field of view in degrees
            
        Returns:
            tuple: (obstacle_positions, obstacle_distances) within the cone
        """
        try:
            # First get all obstacle positions
            all_positions, all_distances = self.fetch_density_distances(use_cache=True)
            
            if not all_positions:
                return [], []
                
            # Get current drone position and orientation
            try:
                drone_state = self.client.getMultirotorState().kinematics_estimated
                drone_pos = np.array([
                    drone_state.position.x_val,
                    drone_state.position.y_val,
                    drone_state.position.z_val
                ])
                
                # Get drone orientation quaternion
                drone_quat = np.array([
                    drone_state.orientation.w_val,
                    drone_state.orientation.x_val, 
                    drone_state.orientation.y_val,
                    drone_state.orientation.z_val
                ])
                
                # Calculate forward vector from drone's actual orientation
                # Assuming NED coordinates where forward is along the x-axis (North) when the drone's yaw is 0
                # This vector represents the local "forward" direction of the drone before any rotation
                local_forward = np.array([1.0, 0.0, 0.0])  
                
                # Helper functions to rotate vectors using quaternions
                def quaternion_multiply(q1, q2):
                    w1, x1, y1, z1 = q1
                    w2, x2, y2, z2 = q2
                    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                    return np.array([w, x, y, z])
                
                def quaternion_conjugate(q):
                    w, x, y, z = q
                    return np.array([w, -x, -y, -z])
                
                def rotate_vector_by_quaternion(v, q):
                    # Convert vector to quaternion form (w=0)
                    v_quat = np.array([0.0, v[0], v[1], v[2]])
                    q_conj = quaternion_conjugate(q)
                    # Apply rotation: q * v * q^-1
                    rotated = quaternion_multiply(
                        quaternion_multiply(q, v_quat),
                        q_conj
                    )
                    # Extract vector part
                    return rotated[1:4]
                
                # Calculate actual forward direction based on drone's orientation
                drone_forward = rotate_vector_by_quaternion(local_forward, drone_quat)
                
                # Normalize drone forward vector
                drone_forward_norm = np.linalg.norm(drone_forward)
                if drone_forward_norm > 0.001:
                    drone_forward = drone_forward / drone_forward_norm
                    
                # Use a weighted combination of the provided direction and drone's forward vector
                # This ensures we primarily use the intended movement direction but
                # also account for the drone's actual orientation
                combined_direction = 0.7 * np.array(direction) + 0.3 * drone_forward
                
                # Normalize the combined direction
                combined_norm = np.linalg.norm(combined_direction)
                if combined_norm > 0.001:  # Avoid division by zero
                    direction = combined_direction / combined_norm
                else:
                    # If normalization fails, fall back to drone's forward direction
                    direction = drone_forward
                    
                logging.debug(f"Using combined direction vector for forward scanning: {direction}")
                
            except Exception as e:
                logging.warning(f"Error getting drone orientation for directional scan: {e}")
                # Fall back to just using the provided direction vector
                direction = np.array(direction)
            
            # Normalize direction vector
            direction = np.array(direction)
            norm = np.linalg.norm(direction)
            if norm < 0.001:  # Avoid division by zero
                return [], []
            direction = direction / norm
            
            # Calculate cosine of cone angle
            cos_angle = np.cos(np.radians(cone_angle))
            
            # Filter obstacles within the cone
            in_cone_positions = []
            in_cone_distances = []
            
            for i, pos in enumerate(all_positions):
                if i >= len(all_distances):  # Safety check
                    continue
                    
                # Skip if beyond max distance
                if all_distances[i] > max_distance:
                    continue
                    
                # Vector from drone to obstacle
                to_obstacle = np.array(pos) - drone_pos
                to_obstacle_dist = np.linalg.norm(to_obstacle)
                
                if to_obstacle_dist < 0.001:  # Avoid division by zero
                    continue
                    
                # Normalize
                to_obstacle = to_obstacle / to_obstacle_dist
                
                # Calculate dot product to get cosine of angle between vectors
                angle_cos = np.dot(direction, to_obstacle)
                
                # Apply distance-based filtering:
                # For farther obstacles, require them to be more closely aligned with the direction
                # This prevents excessive false positives for obstacles that aren't directly in our path
                required_cos_angle = cos_angle
                
                # For distant obstacles, tighten the cone angle
                if to_obstacle_dist > 4.0:
                    # Linearly increase required alignment from the base cos_angle at 4m
                    # to a much tighter angle at max_distance
                    # This makes the cone narrower for more distant obstacles, focusing only on
                    # those directly in front when they're far away
                    distance_factor = min(1.0, (to_obstacle_dist - 4.0) / (max_distance - 4.0))
                    
                    # Calculate tighter cone angle - at max distance it will be about half the original cone angle
                    narrowed_angle = cone_angle * (1.0 - 0.5 * distance_factor)
                    required_cos_angle = np.cos(np.radians(narrowed_angle))
                
                # If obstacle is within the (potentially narrowed) cone
                if angle_cos > required_cos_angle:
                    in_cone_positions.append(pos)
                    in_cone_distances.append(all_distances[i])
            
            return in_cone_positions, in_cone_distances
            
        except Exception as e:
            logging.error(f"Error in scan_in_direction: {e}")
            traceback.print_exc()
            return [], []

def generate_target_pool(start_pos: List[float], distance_range: Tuple[float, float],
                      client: airsim.MultirotorClient, num_targets: int = 100,
                      max_attempts: int = 300, seed: int = None,
                      ray_checks: int = 7) -> List[List[float]]:
    """Pre-generate a pool of valid target positions for use throughout the experiment
    
    Args:
        start_pos: Starting drone position [x, y, z]
        distance_range: (min_distance, max_distance) in meters
        client: AirSim client instance
        num_targets: Number of targets to generate for the pool
        max_attempts: Maximum sampling attempts per target
        seed: Random seed for deterministic behavior
        ray_checks: Number of rays to use for validating each target
        
    Returns:
        List[List[float]]: List of target positions [x, y, z] in NED coordinates
        
    Raises:
        ValueError: If unable to generate enough valid targets
    """
    logging.info(f"Generating pool of {num_targets} valid target locations...")
    
    # Create a separate random generator for deterministic target generation
    if seed is not None:
        target_rng = random.Random(seed)
    else:
        target_rng = random.Random()
    
    target_pool = []
    total_attempts = 0
    max_total_attempts = max_attempts * num_targets * 2  # Upper bound to prevent infinite loops
    
    # Start time to track performance
    start_time = time.time()
    
    # Try to generate the requested number of targets
    while len(target_pool) < num_targets and total_attempts < max_total_attempts:
        try:
            # For each target, we'll use a different "episode id" to ensure diversity
            # This lets us leverage the existing sample_visible_target logic directly
            fake_episode_id = len(target_pool)
            
            # Get a valid target location
            target_pos = sample_visible_target(
                start_pos,
                distance_range,
                client,
                max_attempts=max(50, max_attempts // num_targets),
                episode_id=fake_episode_id,
                seed=seed,
                ray_checks=ray_checks
            )
            
            if target_pos:
                target_pool.append(target_pos)
                
                # Log progress periodically
                if len(target_pool) % 10 == 0 or len(target_pool) == num_targets:
                    elapsed = time.time() - start_time
                    logging.info(f"Generated {len(target_pool)}/{num_targets} targets "
                                f"in {elapsed:.1f}s ({total_attempts} attempts)")
        
        except Exception as e:
            logging.warning(f"Error generating target {len(target_pool)}: {e}")
        
        total_attempts += 1
    
    # Check if we generated enough targets
    if len(target_pool) < num_targets:
        logging.warning(f"Could only generate {len(target_pool)}/{num_targets} valid targets "
                      f"after {total_attempts} attempts")
        if len(target_pool) == 0:
            raise ValueError("Failed to generate any valid targets for the pool")
    else:
        logging.info(f"Successfully generated pool of {len(target_pool)} targets "
                   f"in {time.time() - start_time:.1f}s")
    
    return target_pool

def move_to_waypoint(client, current_pos, waypoint, velocity=2, distance_to_target=None, high_density=False):
    """Move to waypoint with calculated yaw so drone faces direction of travel
    
    Args:
        client: AirSim client instance
        current_pos: Current drone position [x, y, z]
        waypoint: Target waypoint [x, y, z]
        velocity: Movement velocity in m/s
        distance_to_target: Optional distance to final target (for adaptive behavior)
        high_density: Whether we're in a high density obstacle area
        
    Returns:
        tuple: (obstacle_detected, obstacle_pos, obstacle_dist) - 
               Boolean indicating if obstacle was detected, the position and distance of the obstacle
    """
    try:
        # Calculate movement vector and distance
        movement_vector = np.array(waypoint) - np.array(current_pos)
        distance = np.linalg.norm(movement_vector)
          # Adjust velocity based on context
        adaptive_velocity = velocity
        if high_density:
            # More conservative velocity in high density areas
            adaptive_velocity = min(velocity, 1.5)
        
        # If we're close to target, reduce velocity for more precise control
        near_goal = distance_to_target is not None and distance_to_target < 5.0
        if near_goal:
            adaptive_velocity = min(adaptive_velocity, 1.5)
        
        # If velocity was adjusted, log it
        if adaptive_velocity != velocity:
            logging.debug(f"Adjusted velocity from {velocity:.1f} to {adaptive_velocity:.1f} m/s based on environment")
          # Create scanner for obstacle detection during movement (needed before checking minimum distance)
        scanner = Scanner(client, scan_range=8.0)  # Use a shorter scan range for immediate obstacles
          
        # Enforce minimum movement distance - with adaptive behavior based on context
        # Base minimum movement distance
        min_movement_distance = 1.0  # Default minimum to ensure significant movement
        
        # Reduce minimum distance when close to target for more precise positioning
        if distance_to_target is not None and distance_to_target < 2.0:
            min_movement_distance = 0.2  # Much smaller minimum when very close to target
            logging.debug(f"Very close to target ({distance_to_target:.2f}m) - allowing smaller movements (min: {min_movement_distance}m)")
        # Also reduce minimum distance in high-density areas for more precise obstacle avoidance
        elif high_density:
            min_movement_distance = 0.5  # Smaller minimum in obstacle-dense areas
            logging.debug(f"In high density area - allowing smaller movements (min: {min_movement_distance}m)")
        
        if distance < min_movement_distance:
            logging.warning(f"Waypoint too close ({distance:.2f}m), extending to minimum distance {min_movement_distance}m")
            # Extend the waypoint in the same direction to meet minimum distance
            if distance > 0.001:  # Avoid division by zero
                # Normalize the movement vector and scale to minimum distance
                normalized_vector = movement_vector / distance
                extended_vector = normalized_vector * min_movement_distance
                waypoint = (np.array(current_pos) + extended_vector).tolist()
                # Recalculate movement vector and distance
                movement_vector = np.array(waypoint) - np.array(current_pos)
                distance = np.linalg.norm(movement_vector)
            else:
                # If distance is effectively zero, create movement in a default direction
                logging.warning("Movement distance effectively zero, creating default movement direction")
                # Add movement in multiple directions as options
                possible_directions = [
                    [1.0, 0.0, 0.0],  # Forward
                    [0.0, 1.0, 0.0],  # Right
                    [-1.0, 0.0, 0.0], # Backward
                    [0.0, -1.0, 0.0]  # Left
                ]
                
                # Try to find a direction that moves toward the target if possible
                if distance_to_target is not None:
                    # Calculate direction to target
                    to_target = np.array(waypoint) - np.array(current_pos)
                    if np.linalg.norm(to_target) > 0.001:
                        to_target = to_target / np.linalg.norm(to_target)
                        possible_directions.insert(0, to_target.tolist())  # Add target direction as first option
                
                # Use the first valid direction that doesn't immediately lead to an obstacle
                valid_direction_found = False
                for direction in possible_directions:
                    test_waypoint = [
                        current_pos[0] + direction[0] * min_movement_distance,
                        current_pos[1] + direction[1] * min_movement_distance, 
                        current_pos[2] + direction[2] * min_movement_distance
                    ]
                    
                    # Check if this direction is obstacle-free
                    obstacle_check = scanner.scan_path_for_obstacles(
                        current_pos, test_waypoint, safety_radius=2.0
                    )
                    
                    if not obstacle_check[0]:  # No obstacle detected
                        waypoint = test_waypoint
                        logging.info(f"Using alternative direction: {direction}")
                        valid_direction_found = True
                        break
                
                # If no valid direction found, just use the first one
                if not valid_direction_found:
                    direction = possible_directions[0]
                    waypoint = [
                        current_pos[0] + direction[0] * min_movement_distance,
                        current_pos[1] + direction[1] * min_movement_distance, 
                        current_pos[2] + direction[2] * min_movement_distance
                    ]
                    logging.warning(f"No obstacle-free direction found, using default: {direction}")
                
                # Recalculate movement vector and distance
                movement_vector = np.array(waypoint) - np.array(current_pos)
                distance = np.linalg.norm(movement_vector)
          # Calculate yaw angle in radians
        # In NED coordinates, yaw is measured from North (x-axis) and increases clockwise
        # atan2(y, x) will give us the angle in the standard math frame, but we need to ensure
        # it's properly mapped to the NED coordinate system
        yaw = math.atan2(movement_vector[1], movement_vector[0])
        
        # Convert to degrees for display and for AirSim's YawMode
        yaw_degrees = math.degrees(yaw)
        
        # AirSim expects yaw in degrees, ranging from -180 to 180 or 0 to 360
        # We already have this range from atan2, but ensure it's clamped correctly
        if yaw_degrees < -180:
            yaw_degrees += 360
        elif yaw_degrees > 180:
            yaw_degrees -= 360
            
        # Get current drone orientation quaternion for sensor orientation correction
        drone_orientation = client.simGetVehiclePose().orientation
        current_yaw = airsim.to_eularian_angles(drone_orientation)[2]
        current_yaw_degrees = math.degrees(current_yaw)
          # Calculate the difference between desired yaw and current yaw
        # This is just for logging purposes now
        yaw_difference = yaw_degrees - current_yaw_degrees
        
        # Normalize to -180 to 180 range for smallest rotation
        if yaw_difference > 180:
            yaw_difference -= 360
        elif yaw_difference < -180:
            yaw_difference += 360
            
        # No pre-rotation step - we'll rely only on the YawMode in moveToPositionAsync 
        # to ensure the drone faces the correct direction during movement
        logging.debug(f"Moving with yaw: {yaw_degrees:.1f}° at velocity: {adaptive_velocity} m/s")
        logging.debug(f"Movement vector: [{movement_vector[0]:.2f}, {movement_vector[1]:.2f}, {movement_vector[2]:.2f}]")
          # Create scanner for obstacle detection during movement
        scanner = Scanner(client, scan_range=10.0)  # Increased range for better forward planning
        
        # Determine safety radius based on context
        # More conservative in high density areas or when far from goal
        safety_radius = 2.5  # Default safety radius
        if high_density:
            safety_radius = 3.5  # Increased safety in high density areas
            
        # When close to target but in a high-density environment, use sector scanning
        # to find potential passages to the target
        if high_density and distance_to_target and distance_to_target < 10.0:
            logging.info("Close to target in high-density area, performing sector scanning")
            
            # Determine direction to target
            if distance_to_target > 0.2:  # Only if target is not right under us
                target_vector = np.array(waypoint) - np.array(current_pos)
                if np.linalg.norm(target_vector) > 0.001:
                    target_vector = target_vector / np.linalg.norm(target_vector)
                    
                    # Scan in sectors focusing toward target direction
                    sector_results = []
                    sector_angles = [-30, -15, 0, 15, 30]  # Angles to check in degrees
                    
                    for angle in sector_angles:
                        # Create rotated direction vector
                        angle_rad = math.radians(angle)
                        rotated_x = target_vector[0] * math.cos(angle_rad) - target_vector[1] * math.sin(angle_rad)
                        rotated_y = target_vector[0] * math.sin(angle_rad) + target_vector[1] * math.cos(angle_rad)
                        rotated_direction = np.array([rotated_x, rotated_y, target_vector[2]])
                        
                        # Normalize direction
                        if np.linalg.norm(rotated_direction) > 0.001:
                            rotated_direction = rotated_direction / np.linalg.norm(rotated_direction)
                            
                            # Check distance to obstacles in this direction
                            obstacles, distances = scanner.scan_in_direction(
                                rotated_direction, max_distance=8.0, cone_angle=10.0
                            )
                            
                            min_distance = 100.0 if not distances else min(distances)
                            sector_results.append((angle, min_distance, rotated_direction))
                    
                    # Prefer sectors with greater obstacle clearance
                    if sector_results:
                        # Sort by distance (descending)
                        sector_results.sort(key=lambda x: x[1], reverse=True)
                        
                        # If best sector has good clearance, use it
                        best_sector = sector_results[0]
                        if best_sector[1] > 3.0:  # Good clearance threshold
                            # Update movement vector based on best sector
                            best_direction = best_sector[2]
                            movement_vector = best_direction * distance
                            logging.info(f"Using sector scan to find clearer path: {best_sector[0]}° with {best_sector[1]:.2f}m clearance")
                            
                            # Recalculate waypoint
                            waypoint = [
                                current_pos[0] + movement_vector[0],
                                current_pos[1] + movement_vector[1],
                                current_pos[2] + movement_vector[2]
                            ]
                            
                            # Calculate new yaw angle based on adjusted direction
                            yaw = math.atan2(movement_vector[1], movement_vector[0])
                            yaw_degrees = math.degrees(yaw)
          # Perform a pre-check for obstacles on the planned path (FOR INFORMATION ONLY)
        obstacle_detected, obstacle_pos, obstacle_dist = scanner.scan_path_for_obstacles(
            current_pos, waypoint, safety_radius=safety_radius
        )
          # Log the pre-check result but DO NOT trigger replanning yet
        if obstacle_detected:
            logging.warning(f"Pre-movement check: Obstacle detected on planned path at {obstacle_dist:.2f}m distance")
            logging.info(f"Proceeding with movement and will replan if obstacle persists during execution")
            # DO NOT return here - continue with movement instead of immediately triggering replanning
        
        # STEP 1: First move to position without forcing yaw rotation
        logging.debug(f"Moving to waypoint at velocity: {adaptive_velocity} m/s")
        
        # Execute the movement without changing yaw (or minimal change for stability)
        # Use MaxDegreeOfFreedom drivetrain for more freedom of movement
        move_task = client.moveToPositionAsync(
            waypoint[0], waypoint[1], waypoint[2],
            adaptive_velocity,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=current_yaw_degrees)  # Use current yaw
        )
        
        # Direction vector (normalized)
        if distance > 0:
            direction = movement_vector / distance
        else:
            direction = np.array([1.0, 0.0, 0.0])  # Default to forward direction            # Parameters for obstacle monitoring
        check_interval = 0.1  # Check for obstacles every 100ms
        start_time = time.time()
        progress_timeout = max(5.0, distance / adaptive_velocity * 3)  # Timeout is 3x expected travel time, minimum 5 seconds
        
        # More aggressive monitoring in high density areas
        if high_density:
            check_interval = 0.05  # More frequent checks in high density areas (50ms)
            progress_timeout = max(5.0, distance / adaptive_velocity * 2)  # Shorter timeout in dense areas
        
        # Monitor movement and check for obstacles
        obstacle_detected = False
        obstacle_pos = None
        obstacle_dist = None
        last_check_time = start_time
        
        # Use a timeout-based polling approach instead of relying on done() method
        movement_complete = False
        timeout_per_check = 0.05  # 50ms timeout for each join attempt
        
        # Set up adaptive obstacle detection parameters
        safety_scan_radius = safety_radius
        safety_threshold = 2.0  # Default obstacle distance threshold to trigger replanning
        
        if high_density:
            # In high density areas, use more aggressive scanning parameters
            safety_scan_radius = safety_radius * 1.2  # Increase scan radius
            safety_threshold = 2.5  # Higher threshold to trigger earlier replanning
        
        while not movement_complete:
            current_time = time.time()
            
            # Check if we've exceeded timeout (stuck detection)
            if current_time - start_time > progress_timeout:
                logging.warning("Movement taking too long, cancelling and replanning")
                client.cancelLastTask()
                # Create a virtual obstacle at the waypoint to avoid this path in future planning
                obstacle_pos = waypoint
                obstacle_dist = distance * 0.5  # Assume obstacle is halfway along path
                return True, obstacle_pos, obstacle_dist            # Only check for obstacles at specified intervals to avoid overloading            if current_time - last_check_time < check_interval:
                # Short sleep to prevent CPU hogging, but periodically check task status
                try:
                    # Check for collisions DURING movement - proactive collision detection
                    collision_info = client.simGetCollisionInfo()
                    if collision_info.has_collided:
                        logging.warning("Collision detected during movement, cancelling task immediately")
                        client.cancelLastTask()
                        # Force a hover to stabilize
                        client.hoverAsync().join()
                        obstacle_detected = True
                        # Use collision normal as a hint for better recovery direction
                        if collision_info.normal.x_val != 0 or collision_info.normal.y_val != 0 or collision_info.normal.z_val != 0:
                            obstacle_pos = [
                                collision_info.impact_point.x_val,
                                collision_info.impact_point.y_val,
                                collision_info.impact_point.z_val
                            ]
                        else:
                            obstacle_pos = waypoint  # Default if normal vector not available
                        obstacle_dist = 0.0  # We've already collided
                        return True, obstacle_pos, obstacle_dist
                except Exception as e:
                    logging.error(f"Error checking for collisions: {e}")
                      # Try to check if task is done without using timeout in join
                    try:
                        if move_task.is_done():
                            movement_complete = True  # Task is done
                            break
                        # If not done, we'll continue obstacle checking
                        time.sleep(timeout_per_check)
                    except Exception:
                        # Error checking task status, continue with obstacle checks
                        time.sleep(0.01)
                    continue
                
            last_check_time = current_time
                
            # Get current drone position
            try:
                drone_state = client.getMultirotorState().kinematics_estimated
                drone_pos = np.array([
                    drone_state.position.x_val,
                    drone_state.position.y_val,
                    drone_state.position.z_val
                ])
            except:
                # If we can't get position, wait and try again
                time.sleep(check_interval)
                continue
                  # Check remaining distance to waypoint
            remaining_dist = np.linalg.norm(np.array(waypoint) - drone_pos)
            
            # Check for collisions again
            collision_info = client.simGetCollisionInfo()
            if collision_info.has_collided:
                logging.warning("Collision detected during movement, cancelling task immediately")
                client.cancelLastTask()
                # Force a hover to stabilize
                client.hoverAsync().join()
                obstacle_detected = True
                # Use collision normal as a hint for better recovery direction
                if collision_info.normal.x_val != 0 or collision_info.normal.y_val != 0 or collision_info.normal.z_val != 0:
                    obstacle_pos = [
                        collision_info.impact_point.x_val,
                        collision_info.impact_point.y_val,
                        collision_info.impact_point.z_val
                    ]
                else:
                    obstacle_pos = waypoint  # Default if normal vector not available
                obstacle_dist = 0.0  # We've already collided
                return True, obstacle_pos, obstacle_dist
            
            if remaining_dist < 0.5:  # Almost at waypoint, just let it complete
                # Even when close to waypoint, still do a final safety check
                obs_check, _, obs_dist = scanner.scan_path_for_obstacles(
                    drone_pos.tolist(), waypoint, safety_radius=safety_scan_radius * 0.8
                )
                
                # If an obstacle is detected very close, stop anyway
                if obs_check and obs_dist is not None and obs_dist < 1.0:
                    logging.warning(f"Obstacle detected at {obs_dist:.2f}m when approaching waypoint")
                    client.cancelLastTask()
                    obstacle_detected = True
                    break
                
                try:
                    # Wait for the movement to actually complete
                    move_task.join()
                    movement_complete = True
                except:
                    # If the join fails, we'll continue - the drone is close enough anyway
                    pass
                break            # Use the optimized scanner methods to check for obstacles
            # First check directly along the path
            obstacle_detected, detected_obstacle_pos, detected_obstacle_dist = scanner.scan_path_for_obstacles(
                drone_pos.tolist(), waypoint, safety_scan_radius
            )
            
            # Get the current drone orientation to properly scan in the direction of travel
            try:
                drone_orientation = client.simGetVehiclePose().orientation
                current_yaw = airsim.to_eularian_angles(drone_orientation)[2]
                # Create direction vector from current yaw
                forward_direction = np.array([
                    math.cos(current_yaw),
                    math.sin(current_yaw),
                    0.0  # Keep scanning in horizontal plane
                ])
            except Exception as e:
                logging.warning(f"Failed to get drone orientation: {e}")
                # Fallback to using direction to waypoint
                forward_direction = direction
            
            if obstacle_detected:
                obstacle_pos = detected_obstacle_pos
                obstacle_dist = detected_obstacle_dist
                
                if obstacle_dist is not None:
                    logging.warning(f"Obstacle detected at {obstacle_dist:.2f}m during movement execution")
                    
                    # In high density areas, avoid obstacles more conservatively
                    if high_density or obstacle_dist < safety_threshold:
                        # Cancel the movement and signal replanning
                        client.cancelLastTask()
                        break
                    else:
                        # For distant obstacles in low-density areas, log but continue
                        # (we may get past them or they'll trigger cone detection if needed)
                        logging.info("Obstacle not imminent, continuing with caution")
                else:
                    logging.warning("Obstacle detected during movement execution (distance unknown)")
                    # Cancel the movement and signal replanning
                    client.cancelLastTask()
                    break            # Also check in the forward direction with a cone scan (can detect dynamic obstacles better)
            # Adapt cone parameters based on context
            cone_angle = 45.0  # Increased from 35.0 for wider field of view
            forward_scan_distance = 8.0  # Increased from 6.0 for longer range detection
              # Initialize variables to avoid UnboundLocalError
            cone_obstacles = []
            cone_distances = []
            closest_idx = 0
            
            # Set up thresholds for obstacle detection
            cone_threshold = 2.5  # Default threshold
            if high_density:
                cone_angle = 60.0  # Increased from 45.0 for much wider detection in complex environments
                forward_scan_distance = 10.0  # Increased from 8.0 for longer detection range
                cone_threshold = 3.5  # Increased threshold for high density areas# Multi-cone scanning for high density environments
            if high_density:
                # Use multiple overlapping cones to better detect passages and gaps
                cone_angles = [-30, -15, 0, 15, 30]  # Angles in degrees
                min_overall_dist = float('inf')
                min_cone_obstacles = None
                min_cone_idx = None
                
                # Scan in multiple directions to build a more comprehensive picture
                for angle_offset in cone_angles:
                    # Create rotated direction vector based on current yaw (forward direction)
                    angle_rad = math.radians(angle_offset)
                    rotated_x = forward_direction[0] * math.cos(angle_rad) - forward_direction[1] * math.sin(angle_rad)
                    rotated_y = forward_direction[0] * math.sin(angle_rad) + forward_direction[1] * math.cos(angle_rad)
                    rotated_direction = np.array([rotated_x, rotated_y, forward_direction[2]])
                    
                    # Normalize the rotated direction
                    if np.linalg.norm(rotated_direction) > 0.001:
                        rotated_direction = rotated_direction / np.linalg.norm(rotated_direction)
                        
                        # Perform cone scan in this direction
                        cone_obs, cone_dists = scanner.scan_in_direction(
                            rotated_direction, max_distance=forward_scan_distance, 
                            cone_angle=cone_angle/2  # Smaller angle for each sub-cone
                        )
                        
                        # Track closest obstacle across all cones
                        if cone_obs and cone_dists:
                            min_cone_dist = min(cone_dists)
                            if min_cone_dist < min_overall_dist:
                                min_overall_dist = min_cone_dist
                                min_cone_obstacles = cone_obs
                                min_cone_idx = cone_dists.index(min_cone_dist)
                
                # Use the minimum distance found in any cone direction
                if min_overall_dist < float('inf'):
                    cone_obstacles = min_cone_obstacles
                    cone_distances = [min_overall_dist]
                    closest_idx = 0
                    
                    # Check against threshold for high density areas
                    if min_overall_dist < cone_threshold:
                        obstacle_pos = min_cone_obstacles[min_cone_idx]
                        obstacle_dist = min_overall_dist
                        logging.warning(f"Forward obstacle detected at {min_overall_dist:.2f}m during movement (multi-cone scan)")
                        logging.debug(f"Obstacle position: {obstacle_pos}")
                        
                        # Cancel movement and signal replanning needed
                        client.cancelLastTask()
                        obstacle_detected = True
                        break
                else:
                    cone_obstacles = []
                    cone_distances = []
            else:                # Standard single cone scan for lower density environments
                cone_obstacles, cone_distances = scanner.scan_in_direction(
                    forward_direction, max_distance=forward_scan_distance, cone_angle=cone_angle
                )
            
            # Initialize min_cone_dist for safety
            min_cone_dist = float('inf')
            
            # Process obstacles if any were detected
            if cone_obstacles and cone_distances:
                min_cone_dist = min(cone_distances)
                
                # If obstacle is very close in forward direction, halt movement
                if min_cone_dist < cone_threshold:
                    closest_idx = cone_distances.index(min_cone_dist)
                    obstacle_pos = cone_obstacles[closest_idx]
                    obstacle_dist = min_cone_dist
                    logging.warning(f"Forward obstacle detected at {min_cone_dist:.2f}m during movement")
                    logging.debug(f"Obstacle position: {obstacle_pos}")
                    
                    # Cancel movement and signal replanning needed
                    client.cancelLastTask()
                    obstacle_detected = True
                    break            # Check if the movement has completed without using timeout in join
            try:
                if move_task.is_done():
                    movement_complete = True  # Task is done
                    break
                # If not done, wait a bit and continue obstacle checks
                time.sleep(timeout_per_check)
            except:
                # Error checking task status, continue with obstacle checks
                pass
                  # STEP 2: Now that movement is complete, rotate to face the movement direction
        # This ensures the drone's sensors are properly oriented for future observations
        if movement_complete and not obstacle_detected:
            logging.debug(f"Movement complete, rotating to face movement direction: {yaw_degrees:.1f}°")
            try:                # Rotate to the previously calculated yaw that points in the movement direction
                rotate_task = client.rotateToYawAsync(yaw_degrees)
                rotate_task.join()
                logging.debug(f"Yaw rotation complete, drone sensors now facing forward")
            except Exception as e:
                logging.warning(f"Failed to rotate after movement: {e}")
                # Continue without failing the movement - we at least reached the position
        
        # Return obstacle detection status to inform caller if replanning is needed
        return obstacle_detected, obstacle_pos, obstacle_dist
        
    except Exception as e:
        logging.error(f"Error in move_to_waypoint: {e}")
        traceback.print_exc()
        # In case of error, hover in place
        try:
            hover_task = client.hoverAsync()
            hover_task.join()  # Wait for hover command to complete
        except Exception as hover_e:
            logging.error(f"Additional error during hover: {hover_e}")
        return False, None, None  # No explicit obstacle detected on error

def run_episode(episode_id: int, client: airsim.MultirotorClient, 
                zmq_interface: ZMQInterface, scanner: Scanner, 
                config: Dict[str, Any], target_pool: List[List[float]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run a single navigation episode
    
    Args:
        episode_id: Episode identifier
        client: AirSim client instance
        target_pool: Pre-generated pool of valid target locations
        zmq_interface: ZMQ communication interface
        scanner: Scanner for obstacle detection
        config: Experiment configuration
        
    Returns:
        Tuple containing per-step metrics and episode summary
    """
    logging.info(f"=== Starting Episode {episode_id} ===")
    
    # Check AirSim connection before starting episode
    try:
        client.ping()
    except Exception as e:
        logging.error(f"Unable to connect to AirSim at start of episode {episode_id}: {e}")
        raise ConnectionError(f"AirSim connection failed: {e}")
    
    # Reset the drone
    try:
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
    except Exception as e:
        logging.error(f"Error resetting drone state: {e}")
        raise ConnectionError(f"AirSim reset failed: {e}")
    
    # Takeoff - CRITICAL: We must take off before checking target visibility
    logging.info(f"Episode {episode_id}: Taking off...")
    takeoff_task = client.takeoffAsync()
    try:
        takeoff_task.join()  # Wait for takeoff to complete
    except Exception as e:
        logging.error(f"Error during takeoff: {e}")
        return [], {
            "episode_id": episode_id,
            "status": "failed",
            "reason": "Takeoff failed"
        }
    time.sleep(2)  # Extended stabilization time to ensure proper altitude before sampling target
    
    # Get initial position - AFTER takeoff
    drone_state = client.getMultirotorState().kinematics_estimated
    drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
    
    # Ensure drone is at a reasonable altitude before sampling targets
    if drone_pos[2] > -1.0:  # In NED coordinates, negative Z is up
        logging.warning(f"Episode {episode_id}: Drone altitude too low ({drone_pos[2]:.2f}), adjusting height")
        # Move up to ensure better visibility for target sampling
        adjusted_height_pos = [drone_pos[0], drone_pos[1], -5.0]  # Minimum 5m above ground
        move_task = client.moveToPositionAsync(
            adjusted_height_pos[0], adjusted_height_pos[1], adjusted_height_pos[2], 2
        )
        try:
            move_task.join()  # Wait for height adjustment to complete
        except Exception as e:
            logging.error(f"Error during height adjustment: {e}")
            # Continue anyway - we'll try to work with the current height
        time.sleep(1)  # Wait for height adjustment
        
        # Update position after height adjustment
        drone_state = client.getMultirotorState().kinematics_estimated
        drone_pos = [drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val]
        logging.info(f"Episode {episode_id}: Adjusted drone height to {drone_pos[2]:.2f}m")
    
    # Configure number of rays for target validation based on environment complexity
    # Use more rays for later episodes which might be more challenging
    min_rays = config.get("min_ray_checks", 3)
    max_rays = config.get("max_ray_checks", 7)
    ray_checks = min(max_rays, min_rays + (episode_id // 3))  # Start with min_rays, gradually increase
    
    # Enable debug log level temporarily for detailed raycast information if needed
    original_log_level = logging.getLogger().level
    if config.get("debug_raycasting", False) and episode_id % 5 == 0:  # Debug every 5th episode
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(f"Episode {episode_id}: Enhanced logging enabled for raycast debugging")
        logging.debug(f"Using {ray_checks} rays for target validation")        # Get target from the pre-generated pool if available, otherwise fall back to sampling
    target_sampling_start = time.time()    
    try:
        # Use pre-generated target pool if available
        if target_pool and episode_id < len(target_pool):
            # Get the target corresponding to this episode
            target_pos = target_pool[episode_id]
            
            # Verify that this pre-generated target is still valid (environment may have changed)
            start_vector = airsim.Vector3r(drone_pos[0], drone_pos[1], drone_pos[2])
            target_vector = airsim.Vector3r(target_pos[0], target_pos[1], target_pos[2])
            los_valid = client.simTestLineOfSightBetweenPoints(start_vector, target_vector)
            
            if los_valid:
                target_sampling_time = time.time() - target_sampling_start
                logging.info(f"Episode {episode_id}: Using pre-generated target {episode_id} of {len(target_pool)}")
            else:
                # Target is no longer valid, fall back to on-the-fly generation
                logging.warning(f"Episode {episode_id}: Pre-generated target no longer valid, generating new target")
                target_pos = sample_visible_target(
                    drone_pos, 
                    config["target_distance_range"], 
                    client,
                    episode_id=episode_id,
                    seed=config["random_seed"],
                    ray_checks=ray_checks
                )
                target_sampling_time = time.time() - target_sampling_start
                logging.info(f"Episode {episode_id}: Fallback target sampling took {target_sampling_time:.2f}s with {ray_checks} rays")
        else:
            # No pool or index out of range, use on-the-fly generation
            if not target_pool:
                logging.warning(f"Episode {episode_id}: No target pool available")
            elif episode_id >= len(target_pool):
                logging.warning(f"Episode {episode_id}: Target pool exhausted (only {len(target_pool)} targets available)")
            
            target_pos = sample_visible_target(
                drone_pos, 
                config["target_distance_range"], 
                client,
                episode_id=episode_id,
                seed=config["random_seed"],
                ray_checks=ray_checks
            )
            target_sampling_time = time.time() - target_sampling_start
            logging.info(f"Episode {episode_id}: Target sampling took {target_sampling_time:.2f}s with {ray_checks} rays")
    except ValueError as e:
        logging.warning(f"Episode {episode_id}: {str(e)}")
        # Attempt to get a simpler target if sampling failed
        try:
            target_pos = [drone_pos[0] + 10.0, drone_pos[1], drone_pos[2]]
            logging.warning(f"Episode {episode_id}: Using fallback target at {target_pos}")
        except Exception as fallback_error:
            logging.error(f"Episode {episode_id}: Even fallback target failed: {fallback_error}")
            return [], {}  # Return empty results if we couldn't create any target
        
        # Restore original log level if it was changed
        if config.get("debug_rayca+sting", False) and episode_id % 5 == 0:
            logging.getLogger().setLevel(original_log_level)
            
        return [], {
            "episode_id": episode_id,
            "status": "skipped",
            "reason": "No valid target found"
        }
    
    # Restore original log level if it was changed
    if config.get("debug_raycasting", False) and episode_id % 5 == 0:
        logging.getLogger().setLevel(original_log_level)
    
    # Calculate initial distance to target (for percentage-based proximity calculations)
    initial_distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(drone_pos))
    logging.info(f"Episode {episode_id}: Initial distance to target: {initial_distance_to_target:.2f}m")
    
    # Initialize episode tracking
    step_metrics = []
    collisions = 0
    replanning_count = 0
    dynamic_replanning_count = 0  # Track dynamic replanning events specifically
    start_time = time.time()
    trajectory = [drone_pos]    # Initialize stuck detection variables
    last_significant_movement_time = start_time
    last_position = np.array(drone_pos)
    stuck_timeout = config.get("stuck_timeout", 15.0)  # Time in seconds to consider the drone stuck
    min_movement_threshold = 0.5  # Minimum movement in meters to consider significant
    stuck_check_interval = config.get("stuck_check_interval", 3.0)  # Only check for stuck condition every X seconds
    last_stuck_check_time = start_time
    time_since_movement = 0.0  # Initialize time_since_movement to avoid UnboundLocalError    # Initialize tracking variables for step metrics
    previous_distance_to_target = initial_distance_to_target  # Initialize distance tracker
    previous_step_waypoint = None  # Initialize waypoint tracker
    previous_vfe = 0.0  # Initialize previous VFE for delta calculation
    last_replanning_reason = "none"  # Track the reason for the last replanning event
    
    # Initialize collision tracking
    collision_info = client.simGetCollisionInfo()  # Get initial collision info
      # Initialize observation variable with default values to prevent UnboundLocalError
    observation = {
        "direct_path_clear": False,
        "direct_path_suitability": 0.0
    }
    
    # Initialize environment state variables
    high_density_area = False
    near_goal = False
    closest_obstacle_dist = float('inf')
    obstacle_positions = []
    obstacle_distances = []
    
    # Initialize oscillation detection variables
    position_history = []  # List to store previous positions
    position_visit_count = {}  # Dictionary to track location visit counts
    oscillation_threshold = 1.0  # Distance threshold in meters to consider positions as "similar"
    max_revisits = 10  # Maximum number of times a drone can revisit a similar location
    
    # Initial distance to target
    distance_to_target = initial_distance_to_target
    
    logging.info(f"Episode {episode_id}: Initial position: {[round(p, 2) for p in drone_pos]}")
    logging.info(f"Episode {episode_id}: Target position: {[round(p, 2) for p in target_pos]}")
    logging.info(f"Episode {episode_id}: Initial distance: {distance_to_target:.2f}m")
    
    # Additional validation: Double-check target visibility with multiple rays
    verify_rays = 3  # Use a few rays for verification
    verification_failures = 0
    
    for v_idx in range(verify_rays):
        try:
            # Slight perturbation for robustness
            perturb = 0.05  # 5cm perturbation
            offset = np.array([
                random.uniform(-perturb, perturb),
                random.uniform(-perturb, perturb),
                random.uniform(-perturb, perturb)
            ])
            
            # Create perturbed points
            start_point = airsim.Vector3r(
                drone_pos[0] + offset[0], 
                drone_pos[1] + offset[1], 
                drone_pos[2] + offset[2]
            )
            end_point = airsim.Vector3r(
                target_pos[0] + offset[0], 
                target_pos[1] + offset[1], 
                target_pos[2] + offset[2]
            )
            
            # Verify line of sight
            los_result = client.simTestLineOfSightBetweenPoints(start_point, end_point)
            if not los_result:
                verification_failures += 1
                logging.warning(f"Target verification ray {v_idx} failed")
        except Exception as e:
            logging.warning(f"Error during target verification: {e}")
            verification_failures += 1
    
    if verification_failures > 0:
        logging.warning(f"Episode {episode_id}: Target verification detected possible issues ({verification_failures}/{verify_rays} rays failed)")
    else:
        logging.info(f"Episode {episode_id}: Target verified with {verify_rays} rays")
    
    # Main navigation loop
    status = "timeout"  # Default status
    for step in range(config["max_steps_per_episode"]):
        step_start_time = time.time()
        
        # Check for episode timeout
        if time.time() - start_time > config.get("episode_timeout", 120):
            logging.warning(f"Episode {episode_id}: Timeout after {config.get('episode_timeout', 120)} seconds")
            status = "timeout"
            break
        
        # Periodically check AirSim connection
        if step % 5 == 0:  # Check every 5 steps
            try:
                client.ping()
            except Exception as e:
                logging.error(f"Lost connection to AirSim during episode {episode_id} at step {step}: {e}")
                status = "airsim_crash"
                raise ConnectionError(f"AirSim connection lost during episode: {e}")
            
        
        # Get current drone position
        try:
            drone_state = client.getMultirotorState().kinematics_estimated
            drone_pos = [
                drone_state.position.x_val,
                drone_state.position.y_val,
                drone_state.position.z_val
            ]
        except Exception as e:
            logging.error(f"Failed to get drone position at step {step}: {e}")
            if "Connection" in str(e) or "not responding" in str(e) or "timed out" in str(e):
                status = "airsim_crash"
                raise ConnectionError(f"AirSim connection lost while getting drone position: {e}")
            else:
                # For other errors, try to continue
                logging.warning("Attempting to continue episode despite error")
                continue
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(drone_pos))
        
        # Check if target reached - IMPORTANT: Do this before stuck detection
        if distance_to_target <= ARRIVAL_THRESHOLD:
            logging.info(f"Episode {episode_id}: Target reached at step {step}")
            status = "success"
            break
        
        # Check if drone has moved significantly
        current_position = np.array(drone_pos)
        movement = np.linalg.norm(current_position - last_position)
        
        if movement > min_movement_threshold:
            # Drone has moved significantly, update time and position
            last_significant_movement_time = time.time()
            last_position = current_position
            logging.debug(f"Episode {episode_id}: Significant movement detected: {movement:.2f}m")
          # Only check for stuck condition at specified intervals (not every step)
        # And don't check when drone is close to the target (within 2x the arrival threshold)
        current_time = time.time()
        if (current_time - last_stuck_check_time >= stuck_check_interval and 
                distance_to_target > ARRIVAL_THRESHOLD * 2):
            
            # Update the last check time
            last_stuck_check_time = current_time
            
            # Check if drone is stuck (no significant movement for too long)
            time_since_movement = current_time - last_significant_movement_time
            
            # Check for oscillation pattern (revisiting similar locations)
            # Convert current position to a tuple for consistent hashing
            current_pos_tuple = tuple(current_position)
            
            # Check if current position is similar to any previous position
            oscillation_detected = False
            max_visits = 0
            similar_position_key = None
            
            # If we have collision or a recovery maneuver, check for location similarity
            if collision_info.has_collided or dynamic_replanning_count > 0:
                # Add current position to history
                position_history.append(current_position)
                
                # Grid-based position tracking for more efficient similarity detection
                # Round position to grid cells based on oscillation threshold
                grid_pos = tuple(np.round(current_position / oscillation_threshold).astype(int))
                
                # Update visit count for this grid position
                if grid_pos in position_visit_count:
                    position_visit_count[grid_pos] += 1
                    max_visits = max(max_visits, position_visit_count[grid_pos])
                    if position_visit_count[grid_pos] > max_revisits:
                        oscillation_detected = True
                        similar_position_key = grid_pos
                else:
                    position_visit_count[grid_pos] = 1
                
                # Log detailed information when revisiting locations
                if grid_pos in position_visit_count and position_visit_count[grid_pos] > 1:
                    logging.warning(f"Episode {episode_id}: Drone revisited position {grid_pos} {position_visit_count[grid_pos]} times")
            
            # Terminate episode if oscillation pattern is detected
            if oscillation_detected:
                logging.error(f"Episode {episode_id}: Oscillation pattern detected! Position {similar_position_key} visited {position_visit_count[similar_position_key]} times (> {max_revisits})")
                logging.error(f"Episode {episode_id}: Terminating episode due to oscillation between collision and recovery")
                status = "oscillation"
                break
            
            # Enhanced stuck detection with different severity levels
            if time_since_movement > stuck_timeout:
                # Severe stuck condition - abort episode
                logging.error(f"Episode {episode_id}: Drone stuck for {time_since_movement:.1f}s (>= {stuck_timeout}s), aborting episode")
                status = "stuck"
                break
            elif time_since_movement > 10.0:
                # Moderate stuck condition - try more aggressive replanning
                logging.warning(f"Episode {episode_id}: No significant movement for {time_since_movement:.1f}s, attempting recovery maneuver")
                
                # Try a more aggressive recovery by moving upward slightly to get a better view
                try:
                    # Get current height and add 2 meters
                    recovery_pos = [drone_pos[0], drone_pos[1], drone_pos[2] - 2.0]  # Remember NED coordinates, negative Z is up
                    logging.info(f"Executing vertical recovery maneuver to position: {[round(p, 2) for p in recovery_pos]}")
                      # Move up
                    recovery_task = client.moveToPositionAsync(
                        recovery_pos[0], recovery_pos[1], recovery_pos[2], 2
                    )
                    
                    try:
                        recovery_task.join()  # Wait for recovery task to complete
                    except Exception as e:
                        # If an error occurs during join, log it and continue with replanning
                        logging.warning(f"Recovery task join failed: {e}")
                        pass
                    
                    # Refresh position after recovery attempt
                    drone_state = client.getMultirotorState().kinematics_estimated
                    drone_pos = [
                        drone_state.position.x_val,
                        drone_state.position.y_val,
                        drone_state.position.z_val
                    ]
                    
                    # Update last significant movement time to reset the stuck counter
                    last_significant_movement_time = current_time
                    
                except Exception as e:
                    logging.error(f"Error during recovery maneuver: {e}")
                
                # Force a replanning with enhanced obstacle avoidance parameters
                obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
                next_waypoint, policy, planning_error, inference_time_ms = get_next_waypoint(needs_replanning=True)
                
                if planning_error:
                    logging.error("Recovery planning failed, aborting episode")
                    status = "planning_failure"
                    break
                
                # Try to execute the new plan
                _, new_obstacle_pos, new_obstacle_dist = move_to_waypoint(client, drone_pos, next_waypoint, velocity=1.5)
                
            # Log stuck detection progress if approaching timeout
            elif time_since_movement > stuck_timeout * 0.6:  # At 60% of timeout
                logging.warning(f"Episode {episode_id}: No significant movement for {time_since_movement:.1f}s " +
                              f"(timeout: {stuck_timeout}s)")
                              
            # Update the time_since_movement for metrics
            step_data["time_since_movement"] = time_since_movement        # Check for collisions
        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided:
            collisions += 1
            logging.warning(f"Episode {episode_id}: Collision detected at step {step}")
              
            # Execute collision recovery movement (move backward and upward away from collision)
            try:
                # Get current drone state for orientation information
                drone_state = client.getMultirotorState().kinematics_estimated
                drone_orientation = client.simGetVehiclePose().orientation
                current_yaw = airsim.to_eularian_angles(drone_orientation)[2]
                
                # Determine optimal recovery direction based on collision normal if available
                use_collision_normal = False
                backward_direction = np.zeros(3)
                
                # Check if we have meaningful collision normal information
                if (collision_info.normal.x_val != 0 or 
                    collision_info.normal.y_val != 0 or 
                    collision_info.normal.z_val != 0):
                    
                    # Use collision normal as the preferred direction to move away from obstacle
                    collision_normal = np.array([
                        collision_info.normal.x_val,
                        collision_info.normal.y_val,
                        0.0  # Remove vertical component for horizontal movement
                    ])
                    
                    # Normalize if the vector has non-zero length
                    normal_length = np.linalg.norm(collision_normal)
                    if normal_length > 0.001:
                        collision_normal = collision_normal / normal_length
                        backward_direction = collision_normal
                        use_collision_normal = True
                        logging.info(f"Using collision normal for recovery: [{collision_normal[0]:.2f}, {collision_normal[1]:.2f}, {collision_normal[2]:.2f}]")
                
                # If collision normal isn't available or valid, use orientation-based recovery
                if not use_collision_normal:
                    # Calculate backward direction (opposite of current orientation)
                    backward_direction = np.array([
                        -math.cos(current_yaw),  # Negative of forward x component
                        -math.sin(current_yaw),  # Negative of forward y component
                        0.0  # Keep altitude constant initially
                    ])
                    logging.info(f"Using orientation-based recovery direction: [{backward_direction[0]:.2f}, {backward_direction[1]:.2f}, {backward_direction[2]:.2f}]")
                  
                # Scale to create a backward movement
                recovery_distance = 3.0  # Exactly 3 meters as required
                backward_vector = backward_direction * recovery_distance
                
                # Add upward movement (in NED, negative z is upward)
                # Upward movement to escape obstacles
                upward_distance = 3.0
                upward_vector = np.array([0.0, 0.0, -upward_distance])
                
                # Calculate recovery position (backward and upward)
                recovery_position = [
                    drone_pos[0] + backward_vector[0],
                    drone_pos[1] + backward_vector[1],
                    drone_pos[2] + upward_vector[2]  # Move upward
                ]
                
                # Verify the recovery position is different from current position
                min_expected_movement = 1.0  # Minimum expected movement in meters
                expected_movement = np.linalg.norm(np.array(recovery_position) - np.array(drone_pos))
                
                if expected_movement < min_expected_movement:
                    # If calculated movement is too small, use a default escape vector
                    logging.warning(f"Calculated recovery movement too small ({expected_movement:.2f}m), using default escape vector")
                    # Default to moving up and slightly back as a fallback
                    recovery_position = [
                        drone_pos[0] - 2.0,  # Move back by default
                        drone_pos[1],        # Maintain lateral position
                        drone_pos[2] - 3.0   # Move up significantly (remember NED coordinates)
                    ]
                    logging.info(f"Episode {episode_id}: Executing collision recovery - moving backward {recovery_distance:.1f}m and upward {upward_distance:.1f}m")
                
                # First hover to stabilize before recovery movement
                logging.debug("Putting drone in hover mode before recovery")
                try:
                    # Cancel any ongoing tasks first
                    client.cancelLastTask()
                    time.sleep(0.2)  # Short pause before starting hover
                      # Perform hover with stability check
                    hover_task = client.hoverAsync()
                    hover_task.join()  # Wait for hover to stabilize
                    time.sleep(0.5)  # Additional stabilization period
                    
                    # Get current position and velocity to verify stability
                    current_state = client.getMultirotorState().kinematics_estimated
                    current_vel = np.array([
                        current_state.linear_velocity.x_val,
                        current_state.linear_velocity.y_val,
                        current_state.linear_velocity.z_val
                    ])
                    
                    # If velocity is still high, wait a bit longer
                    current_speed = np.linalg.norm(current_vel)
                    if current_speed > 0.5:
                        logging.info(f"Drone still moving at {current_speed:.2f} m/s, waiting for additional stabilization")
                        time.sleep(1.0)  # Additional stabilization time
                except Exception as e:
                    logging.warning(f"Error during hover stabilization: {e}, continuing with recovery anyway")
                
                # Verify current position before recovery
                pre_recovery_state = client.getMultirotorState().kinematics_estimated
                pre_recovery_pos = [
                    pre_recovery_state.position.x_val,
                    pre_recovery_state.position.y_val,
                    pre_recovery_state.position.z_val
                ]
                logging.debug(f"Pre-recovery position: {[round(p, 2) for p in pre_recovery_pos]}")
                
                # Execute backward and upward movement with more robust parameters
                recovery_task = client.moveToPositionAsync(
                    recovery_position[0], recovery_position[1], recovery_position[2],
                    velocity=1.0,  # Slower velocity for careful movement
                    drivetrain=airsim.DrivetrainType.ForwardOnly,  # More precise control
                    yaw_mode=airsim.YawMode(False, 0)  # Maintain current yaw during recovery
                )
                
                # Wait for the recovery movement to complete with better error handling
                try:
                    # For a 3m movement at 1.0 m/s, should take about 3-5 seconds
                    # Use polling approach instead of direct join for better control
                    logging.debug("Waiting for collision recovery movement to complete...")
                    
                    start_time = time.time()
                    max_wait_time = 8.0  # Maximum seconds to wait for recovery (increased from previous timing)
                    check_interval = 0.5  # Check progress every half second
                    
                    # Poll until movement completes or timeout
                    while time.time() - start_time < max_wait_time:
                        # Check if task is done
                        try:
                            if recovery_task.is_done():
                                logging.debug("Recovery movement task completed normally")
                                break
                        except:
                            # If is_done() fails, just continue polling
                            pass
                            
                        # Check if we've moved from our original position
                        current_state = client.getMultirotorState().kinematics_estimated
                        current_check_pos = [
                            current_state.position.x_val,
                            current_state.position.y_val,
                            current_state.position.z_val
                        ]
                        
                        # Calculate distance moved
                        dist_moved = np.linalg.norm(np.array(current_check_pos) - np.array(pre_recovery_pos))
                        if dist_moved > 1.0:  # If we've moved at least 1 meter, that's progress
                            logging.debug(f"Recovery movement progressing, moved {dist_moved:.2f}m so far")
                        
                        # Short sleep before checking again
                        time.sleep(check_interval)
                    
                    # If we're here and time exceeded, we timed out
                    if time.time() - start_time >= max_wait_time:
                        logging.warning(f"Recovery movement timed out after {max_wait_time} seconds")
                        client.cancelLastTask()
                        # Force a hover to stabilize after cancellation
                        client.hoverAsync().join()
                except Exception as timeout_error:
                    logging.warning(f"Collision recovery movement failed: {timeout_error}")
                    # If an error occurs, hover to stabilize
                    time.sleep(0.5)
                    client.hoverAsync().join()
                    time.sleep(0.5)
                    client.cancelLastTask()
                  # Refresh drone position after recovery attempt
                drone_state = client.getMultirotorState().kinematics_estimated
                drone_pos = [
                    drone_state.position.x_val,
                    drone_state.position.y_val,
                    drone_state.position.z_val
                ]
                
                # Calculate how far we actually moved during recovery
                actual_movement = np.linalg.norm(np.array(drone_pos) - np.array(pre_recovery_pos))
                
                # Verify if recovery was successful by checking if we moved significantly
                if actual_movement < 0.5:  # If we moved less than 0.5 meters, recovery likely failed
                    logging.warning(f"Episode {episode_id}: Collision recovery barely moved the drone ({actual_movement:.2f}m), retrying with emergency protocol")
                    
                    # Try a more aggressive emergency recovery - just move up
                    emergency_position = [
                        drone_pos[0],        # Keep x position
                        drone_pos[1],        # Keep y position
                        drone_pos[2] - 5.0   # Move up significantly (negative Z is up in NED)
                    ]
                    
                    # Execute emergency maneuver with priority and higher velocity
                    client.hoverAsync().join()
                    time.sleep(0.5)
                    
                    try:
                        logging.info(f"Episode {episode_id}: Attempting emergency vertical escape to altitude {emergency_position[2]}")
                        emergency_task = client.moveToPositionAsync(
                            emergency_position[0], emergency_position[1], emergency_position[2],
                            velocity=2.0,  # Faster velocity for emergency
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,  # More freedom
                            yaw_mode=airsim.YawMode(False, 0)  # Hold yaw
                        )
                        
                        # Poll with shorter interval for emergency
                        start_time = time.time()
                        while time.time() - start_time < 5.0:  # Shorter timeout
                            try:
                                if emergency_task.is_done():
                                    break
                            except:
                                pass
                            time.sleep(0.2)
                        
                        # Refresh position after emergency attempt
                        drone_state = client.getMultirotorState().kinematics_estimated
                        drone_pos = [
                            drone_state.position.x_val,
                            drone_state.position.y_val,
                            drone_state.position.z_val
                        ]
                        
                        # Check emergency recovery success
                        emergency_movement = np.linalg.norm(np.array(drone_pos) - np.array(pre_recovery_pos))
                        if emergency_movement > 2.0:
                            logging.info(f"Episode {episode_id}: Emergency recovery successful, moved {emergency_movement:.2f}m")
                        else:
                            logging.warning(f"Episode {episode_id}: Emergency recovery had limited success, moved only {emergency_movement:.2f}m")
                    except Exception as e:
                        logging.error(f"Episode {episode_id}: Emergency recovery failed: {e}")
                        client.hoverAsync().join()  # Ensure stability
                else:
                    logging.info(f"Episode {episode_id}: Collision recovery successful, moved {actual_movement:.2f}m to new position: {[round(p, 2) for p in drone_pos]}")
                
                # First stabilize with a hover after any recovery attempt
                client.hoverAsync().join()
                time.sleep(0.5)  # Brief pause to fully stabilize
                
                # Check the current orientation after recovery
                drone_orientation_after_recovery = client.simGetVehiclePose().orientation
                current_yaw_after_recovery = airsim.to_eularian_angles(drone_orientation_after_recovery)[2]
                
                # Calculate yaw difference (in radians)
                # Use normalized angle difference calculation to handle wrap-around properly
                yaw_diff_rad = abs(((current_yaw - current_yaw_after_recovery + math.pi) % (2 * math.pi)) - math.pi)
                
                # Only rotate if the difference is significant (more than 5 degrees)
                yaw_threshold_rad = math.radians(5.0)
                if yaw_diff_rad > yaw_threshold_rad:
                    # Rotate to face the original forward direction again
                    # This ensures sensors are facing forward for observation gathering
                    try:
                        logging.debug(f"Rotating drone back to forward direction (current: {math.degrees(current_yaw_after_recovery):.1f}°, target: {math.degrees(current_yaw):.1f}°, diff: {math.degrees(yaw_diff_rad):.1f}°)")                    
                        rotate_task = client.rotateToYawAsync(math.degrees(current_yaw))
                        rotate_task.join()
                    except Exception as rotation_error:
                        logging.warning(f"Post-recovery rotation failed: {rotation_error}")
                        # Continue even if rotation failed
                else:
                    logging.debug(f"Skipping rotation as drone is already properly oriented (diff: {math.degrees(yaw_diff_rad):.1f}°)")
                
            except Exception as e:
                logging.error(f"Episode {episode_id}: Error during collision recovery: {e}")
                # Continue even if recovery failed
        
        # Get obstacle data
        obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
        
        # Calculate obstacle density metrics for adaptive planning
        closest_obstacle_dist = min(obstacle_distances) if obstacle_distances else float('inf')
        obstacle_count = len(obstacle_positions)
        
        # Determine if we're in a high-density obstacle area
        high_density_area = obstacle_count > 10 or closest_obstacle_dist < 3.0
        
        # Calculate proximity to goal for adaptive behavior
        near_goal = distance_to_target < 5.0  # Consider 5m to target as "near goal"
        
        # Calculate exploration vs exploitation balance based on distance to target
        # When far from goal, prioritize safety/exploration
        # When near goal, prioritize exploitation/goal-seeking
        exploration_factor = min(1.0, distance_to_target / 25.0)  # Scale from 0-1 based on distance
        exploration_factor = max(0.3, exploration_factor)  # Ensure minimum exploration of 0.3
        
        # Dynamically adjust obstacle avoidance priorities based on context
        obstacle_priority = 1.0  # Default priority
        
        # Increase obstacle priority when in dense areas or when far from goal
        if high_density_area:
            obstacle_priority = 2.0  # Double obstacle priority in dense areas
        
        # Further increase priority when both far from goal and in dense areas
        if high_density_area and not near_goal:
            obstacle_priority = 2.5  # Even higher priority for obstacle avoidance
            
        # Safety margin increases with obstacle density and distance from goal
        # (further from goal = prioritize safety)
        base_safety_margin = MARGIN
        if high_density_area:
            # Higher margin in dense areas
            base_safety_margin = max(2.5, MARGIN * 1.5)
        
        # When near goal, slightly reduce safety margin to allow closer approach
        # but only if not in a high-density area
        adaptive_safety_margin = base_safety_margin
        if near_goal and not high_density_area:
            # Slightly reduce margin near goal but maintain safety
            adaptive_safety_margin = max(0.8, base_safety_margin * 0.85) 
            logging.debug(f"Episode {episode_id}: Near goal - adjusted safety margin to {adaptive_safety_margin:.1f}m")
        elif near_goal and high_density_area:
            # Near goal but in high density, maintain higher safety
            logging.debug(f"Episode {episode_id}: Near goal in high density area - maintaining higher safety margin at {adaptive_safety_margin:.1f}m")
        
        # Modify the DENSITY_RADIUS parameter based on obstacle density and goal proximity
        adaptive_density_radius = DENSITY_RADIUS
        if high_density_area:
            # Reduce density radius in high-density areas for more precise planning
            adaptive_density_radius = max(2.0, DENSITY_RADIUS * 0.6)
            logging.debug(f"Episode {episode_id}: High obstacle density area - reduced density radius to {adaptive_density_radius:.1f}m")
        
        # Log obstacle detection info for debugging
        if step % 10 == 0 or high_density_area:  # Log more frequently in high-density areas
            logging.debug(f"Episode {episode_id}: Detected {len(obstacle_positions)} obstacles")
            if len(obstacle_positions) > 0:
                min_dist = min(obstacle_distances) if obstacle_distances else "N/A"
                logging.debug(f"Episode {episode_id}: Closest obstacle at {min_dist}m")
                logging.debug(f"Episode {episode_id}: Planning with exploration_factor={exploration_factor:.2f}, obstacle_priority={obstacle_priority:.2f}")
          # Main planning function - will be used for initial planning and dynamic replanning
        def get_next_waypoint(needs_replanning=False):
            nonlocal replanning_count, dynamic_replanning_count, last_replanning_reason
            
            if needs_replanning:
                reason = "obstacle_detected"
                logging.info(f"Episode {episode_id}: Triggering dynamic replanning due to detected obstacle")
                replanning_count += 1
                dynamic_replanning_count += 1
                last_replanning_reason = reason
                # Pass the replanning reason to the observation
                observation["replanning_reason"] = reason
            
            # Create observation for active inference with adaptive parameters
            observation = {
                "drone_position": drone_pos,
                "target_position": target_pos,
                "obstacle_positions": obstacle_positions,
                "obstacle_distances": obstacle_distances,
                "waypoint_count": WAYPOINT_SAMPLE_COUNT,
                "safety_margin": adaptive_safety_margin,
                "policy_length": POLICY_LENGTH,
                "density_radius": adaptive_density_radius,
                "near_goal": near_goal,  # Add flag to inform planner we're near the goal
                "distance_to_target": distance_to_target,  # Explicitly include distance to target
                "initial_distance": initial_distance_to_target,  # Provide initial distance for percentage calculations
                "dynamic_replanning": needs_replanning,  # Flag to inform planner this is a dynamic replanning request
                "exploration_factor": exploration_factor,  # Add balance factor to guide exploration vs. exploitation
                "obstacle_priority": obstacle_priority,  # Add obstacle priority factor for weighting
                "high_density_area": high_density_area,  # Explicitly flag if we're in a high-density area
                "obstacles_count": len(obstacle_positions),  # Pass the exact count of detected obstacles
                "closest_obstacle": closest_obstacle_dist if obstacle_distances else float('inf'),  # Pass distance to closest obstacle
                "critical_obstacle_avoidance": False  # Default to normal mode (may be updated if replanning)
            }
            
            # Add enhanced obstacle awareness based on context
            if distance_to_target > 15.0:
                # When far from target, increase obstacle repulsion weight
                observation["obstacle_repulsion_weight"] = 0.3
                logging.debug(f"Far from target ({distance_to_target:.2f}m), increasing obstacle repulsion")
            elif distance_to_target < 5.0:            # When close to target, slightly reduce obstacle repulsion to allow approach
                observation["obstacle_repulsion_weight"] = 0.1
                logging.debug(f"Close to target ({distance_to_target:.2f}m), optimizing for goal approach")
                
                # When very close to target (last 2 meters) in high-density area, drastically increase target preference
                if distance_to_target < 2.0 and high_density_area:
                    observation["target_preference_boost"] = 0.7  # Signal to Julia planner to boost target preference
                    logging.info(f"Very close to target ({distance_to_target:.2f}m) in dense area - boosting target preference")
                else:
                    observation["target_preference_boost"] = 0.0  # Normal behavior
            else:
                # Linear scaling for intermediate distances
                repulsion_weight = 0.1 + (0.2 * (distance_to_target - 5.0) / 10.0)
                observation["obstacle_repulsion_weight"] = repulsion_weight
                logging.debug(f"Medium distance ({distance_to_target:.2f}m), scaled obstacle repulsion: {repulsion_weight:.2f}")
            
            # Assess if direct path to goal is clear
            if obstacle_positions and len(obstacle_positions) > 0:
                direct_path_clear = True
                direct_path_suitability = 1.0
                
                # Use scanner to check if direct path to target is clear
                obstacle_detected, obstacle_pos, obstacle_dist = scanner.scan_path_for_obstacles(
                    drone_pos, target_pos, safety_radius=3.0
                )
                
                if obstacle_detected:
                    direct_path_clear = False
                    if obstacle_dist is not None:
                        # Calculate a suitability between 0-1 based on obstacle distance
                        # Close obstacles severely reduce suitability
                        direct_path_suitability = min(1.0, max(0.1, obstacle_dist / 15.0))
                    else:
                        direct_path_suitability = 0.3  # Default moderate-low suitability when obstacle distance unknown
                    
                    logging.debug(f"Direct path to target is blocked. Path suitability: {direct_path_suitability:.2f}")
                else:
                    logging.debug("Direct path to target appears clear")
                
                observation["direct_path_clear"] = direct_path_clear
                observation["direct_path_suitability"] = direct_path_suitability
            
            # Adjust waypoint sampling based on context
            if high_density_area:
                # Use more samples in complex environments
                observation["waypoint_count"] = int(WAYPOINT_SAMPLE_COUNT * 1.3)
                
            # Adjust policy length based on distance to target
            if near_goal:
                # Shorter policy horizon near goal for more precise control
                observation["policy_length"] = max(1, POLICY_LENGTH - 1)
            elif distance_to_target > 20.0:
                # Longer policy horizon when far away for better long-term planning
                observation["policy_length"] = min(5, POLICY_LENGTH + 1)
              # If we're replanning due to obstacle detection, enhance the obstacle data
            # to ensure the planner gives it proper consideration
            if needs_replanning:
                # Check if we're very close to target - adjust behavior accordingly
                very_close_to_target = distance_to_target < 2.0
                
                # Increase the safety margin during replanning to be more cautious
                # But use a more moderate increase when extremely close to the target
                if very_close_to_target and high_density_area:
                    observation["safety_margin"] = max(adaptive_safety_margin * 1.5, 3.0)  # More moderate increase
                    logging.info(f"Very close to target ({distance_to_target:.2f}m) - using moderate safety margin increase")
                else:
                    observation["safety_margin"] = max(adaptive_safety_margin * 2.0, 4.0)  # Standard substantial increase
                
                # Increase waypoint sampling to find more path options
                observation["waypoint_count"] = int(WAYPOINT_SAMPLE_COUNT * 2.0)  # Increased from 1.5
                
                # Add extra parameters to highlight obstacle importance
                # Reduce obstacle weights when very close to target to prevent excessive avoidance
                if very_close_to_target and high_density_area:
                    observation["obstacle_weight_factor"] = 2.0  # Reduced for very close to target
                    observation["obstacle_distance_weight"] = 2.0  # Reduced for very close to target
                    observation["target_preference_boost"] = 0.8  # Higher boost during replanning near target
                    logging.info(f"Final approach adjusted: reduced obstacle weights, increased target preference")
                else:
                    observation["obstacle_weight_factor"] = 3.0
                    observation["obstacle_distance_weight"] = 3.0
                
                observation["obstacle_density_weight"] = 1.5  # Increased from 1.0
                observation["replanning_obstacle_threshold"] = 5.0  # Increased from 4.0
                observation["exploration_factor"] = min(1.0, exploration_factor * 2.0)  # Increased from 1.5
                observation["obstacle_priority"] = max(3.0, obstacle_priority * 2.0)  # Increased from 2.0 and 1.5
                observation["suitability_threshold"] = 0.75  # Increased to exact value 0.75 as required
                
                # Add flag to indicate this is a critical avoidance scenario
                observation["critical_obstacle_avoidance"] = True
                
                logging.info(f"Enhanced planning parameters for obstacle avoidance: safety_margin={observation['safety_margin']:.2f}, " 
                             f"waypoint_count={observation['waypoint_count']}, "
                             f"obstacle_priority={observation['obstacle_priority']:.2f}")
            else:                # For normal planning, still ensure high suitability threshold
                observation["suitability_threshold"] = 0.75  # Increased to 0.75 for safer path selection
            
            # Verify the observation is valid before sending
            has_nans = any(np.isnan(np.array(drone_pos))) or any(np.isnan(np.array(target_pos)))
            has_infs = any(np.isinf(np.array(drone_pos))) or any(np.isinf(np.array(target_pos)))
            
            if has_nans or has_infs:
                logging.error(f"Episode {episode_id}: Invalid position data detected (NaN or Inf)")
                return None, None, True, 0  # Signal planning error, include 0 for inference time

            # Get next waypoint and planning metrics
            inference_start = time.time()
            next_waypoint, policy = zmq_interface.send_observation_and_receive_action(observation)
            inference_time_ms = (time.time() - inference_start) * 1000  # Convert to ms
              # Check if planning succeeded
            if next_waypoint is None:
                logging.error(f"Episode {episode_id}: Failed to get valid waypoint at step {step}")
                retry_count = 0
                # Try replanning up to 3 times
                while next_waypoint is None and retry_count < 3:
                    retry_count += 1
                    replanning_count += 1
                    last_replanning_reason = "planning_failure"
                    logging.info(f"Episode {episode_id}: Replanning attempt {retry_count}")
                    # Pass the replanning reason to the observation
                    observation["replanning_reason"] = "planning_failure"
                    next_waypoint, policy = zmq_interface.send_observation_and_receive_action(observation)
                
                if next_waypoint is None:
                    logging.error(f"Episode {episode_id}: Aborting after failed replanning")
                    return None, None, True, 0  # Signal planning error, include 0 for inference time
            
            # Verify waypoint is valid
            waypoint_array = np.array(next_waypoint)
            if np.any(np.isnan(waypoint_array)) or np.any(np.isinf(waypoint_array)):
                logging.error(f"Episode {episode_id}: Invalid waypoint received (contains NaN or Inf)")
                return None, None, True, 0  # Signal planning error, include 0 for inference time
                
            # If this is a replanning scenario, verify the waypoint doesn't lead through known obstacles
            if needs_replanning and obstacle_positions and len(obstacle_positions) > 0:
                # Check if the planned waypoint path goes through any known obstacles
                path_clear = True
                problem_obstacle = None
                problem_distance = None
                
                # Use the scanner to check the path specifically
                obstacle_detected, obstacle_pos, obstacle_dist = scanner.scan_path_for_obstacles(
                    drone_pos, next_waypoint, safety_radius=3.0  # Use larger safety radius during replanning
                )
                
                if obstacle_detected:
                    logging.warning(f"Episode {episode_id}: Planned waypoint would lead through obstacle at {obstacle_dist:.2f}m")
                    path_clear = False
                    problem_obstacle = obstacle_pos
                    problem_distance = obstacle_dist
                
                # If the path isn't clear, try an alternative approach - move slightly away from obstacles first
                if not path_clear:
                    logging.info(f"Episode {episode_id}: Attempting to generate safer waypoint away from obstacles")
                    
                    # Find direction away from the closest obstacle
                    obstacle_vec = np.array(drone_pos) - np.array(problem_obstacle)
                    distance_to_obstacle = np.linalg.norm(obstacle_vec)
                    
                    if distance_to_obstacle > 0.1:  # Ensure we're not too close to divide by zero
                        # Normalize and scale to create a small retreat vector
                        retreat_direction = obstacle_vec / distance_to_obstacle
                        retreat_distance = min(3.0, distance_to_obstacle * 0.5)  # Don't retreat too far
                        
                        # Create a new waypoint that moves away from the obstacle first
                        safe_waypoint = (np.array(drone_pos) + retreat_direction * retreat_distance).tolist()
                        
                        logging.info(f"Episode {episode_id}: Generated safety waypoint to retreat from obstacle " 
                                     f"at direction {[round(d, 2) for d in retreat_direction]}, " 
                                     f"distance {retreat_distance:.2f}m")
                        
                        # Replace the proposed waypoint with our safer alternative
                        next_waypoint = safe_waypoint
            
            return next_waypoint, policy, False, inference_time_ms  # No planning error, include inference time
        
        # Plan the initial waypoint
        next_waypoint, policy, planning_error, inference_time_ms = get_next_waypoint()
        
        # Check for planning errors
        if planning_error:
            status = "planning_failure"
            break
        
        # Calculate action magnitude (waypoint distance)
        action_magnitude = np.linalg.norm(np.array(next_waypoint) - np.array(drone_pos))
        
        # Adaptively adjust waypoint based on obstacle density and goal proximity
        # In high-density areas, we want to maintain the direction but potentially reduce step size
        if high_density_area and not near_goal and action_magnitude > 3.0:        # In high density areas (but not near goal), we limit large movements for safety
            # Scale down the waypoint distance to avoid large jumps in dense areas
            waypoint_array = np.array(next_waypoint)
            direction = (waypoint_array - np.array(drone_pos)) / action_magnitude
            adjusted_distance = min(3.0, action_magnitude)  # Cap at 3m in dense areas
            next_waypoint = (np.array(drone_pos) + direction * adjusted_distance).tolist()
            logging.debug(f"Episode {episode_id}: Scaled down waypoint in dense area from {action_magnitude:.2f}m to {adjusted_distance:.2f}m")
            # Update action magnitude after scaling
            action_magnitude = adjusted_distance
        
        # Near goal - prioritize direct approach to target when possible
        if near_goal:
            # Calculate direct path to target
            to_target = np.array(target_pos) - np.array(drone_pos)
            to_target_dist = np.linalg.norm(to_target)
            to_target_dir = to_target / to_target_dist if to_target_dist > 0 else np.array([1, 0, 0])
            
            # Check if direct path to target is obstacle-free
            direct_path_clear = False
            
            # Only perform this check if we're within a reasonable distance to the target
            if to_target_dist < 7.0:  # 7 meters or closer
                logging.debug(f"Episode {episode_id}: Near goal ({to_target_dist:.2f}m) - checking for direct path")
                obstacle_detected, _, _ = scanner.scan_path_for_obstacles(
                    drone_pos, target_pos, safety_radius=2.0  # Use a slightly smaller safety radius for direct approach
                )
                direct_path_clear = not obstacle_detected
                
                # If direct path is clear, use it instead of the computed waypoint
                if direct_path_clear:
                    # Choose step size based on distance to target
                    direct_step_size = min(to_target_dist * 0.8, 2.0)  # 80% of distance to target or max 2m
                    
                    # Create waypoint directly toward target
                    direct_waypoint = (np.array(drone_pos) + to_target_dir * direct_step_size).tolist()
                    logging.info(f"Episode {episode_id}: Clear path to target detected - moving directly toward goal")
                    
                    # Replace the computed waypoint with direct path to target
                    next_waypoint = direct_waypoint
                    action_magnitude = direct_step_size
            
            # Special case: When in high density areas near goal
            if high_density_area and not direct_path_clear:
                # Calculate direction of planned waypoint
                waypoint_array = np.array(next_waypoint)
                to_waypoint = waypoint_array - np.array(drone_pos)
                to_waypoint_dist = np.linalg.norm(to_waypoint)
                to_waypoint_dir = to_waypoint / to_waypoint_dist if to_waypoint_dist > 0 else np.array([1, 0, 0])
                
                # Calculate dot product to see if waypoint is roughly in direction of target
                alignment = np.dot(to_target_dir, to_waypoint_dir)
                
                # If waypoint is not aligned with target direction and we're near the goal
                # in a dense area, blend the directions to favor target approach
                if alignment < 0.7 and to_target_dist < 3.0:  # Less than 70% aligned and very close
                    # Create a blended direction (60% to target, 40% original waypoint direction)
                    blended_dir = 0.6 * to_target_dir + 0.4 * to_waypoint_dir
                    blended_dir = blended_dir / np.linalg.norm(blended_dir)  # Normalize
                    
                    # Use a conservative step size in this sensitive region
                    step_size = min(1.5, to_target_dist * 0.6)  # 60% of distance to target or 1.5m max
                    
                    # Create new waypoint that's more aligned with target
                    next_waypoint = (np.array(drone_pos) + blended_dir * step_size).tolist()
                    logging.debug(f"Episode {episode_id}: Near goal in dense area - adjusted waypoint for better target alignment")
                    
                    # Update action magnitude after adjustment
                    action_magnitude = step_size
          # Extract additional metrics from policy if available
        # These would need to be included in the Julia server response
        vfe = policy[0].get("vfe", 0.0) if policy and len(policy) > 0 else 0.0
        efe = policy[0].get("efe", 0.0) if policy and len(policy) > 0 else 0.0
        suitability = policy[0].get("suitability", 0.0) if policy and len(policy) > 0 else 0.0
        
        # Extract advanced inference metrics
        efe_pragmatic = policy[0].get("efe_pragmatic", 0.0) if policy and len(policy) > 0 else 0.0
        efe_epistemic = policy[0].get("efe_epistemic", 0.0) if policy and len(policy) > 0 else 0.0
        efe_vs_vfe_gap = policy[0].get("efe_vs_vfe_gap", 0.0) if policy and len(policy) > 0 else 0.0
        suitability_std = policy[0].get("suitability_std", 0.0) if policy and len(policy) > 0 else 0.0
        action_heading_angle_rad = policy[0].get("action_heading_angle_rad", 0.0) if policy and len(policy) > 0 else 0.0
        action_heading_angle_deg = policy[0].get("action_heading_angle_deg", 0.0) if policy and len(policy) > 0 else 0.0
        replanning_triggered_reason = policy[0].get("replanning_triggered_reason", "none") if policy and len(policy) > 0 else "none"
        
        # Calculate delta_vfe (change in VFE from previous step)
        delta_vfe = vfe - previous_vfe if step > 1 else 0.0
        previous_vfe = vfe  # Store for next step
        
        # Record step metrics
        step_data = {
            # Basic episode information
            "episode_id": episode_id,
            "step": step,
            "timestamp": time.time(),
              # Distance metrics
            "distance_to_target": distance_to_target,
            "normalized_distance": distance_to_target / initial_distance_to_target,
            "distance_improvement": previous_distance_to_target - distance_to_target,
            
            # Drone position and waypoint information
            "position_x": drone_pos[0],
            "position_y": drone_pos[1],
            "position_z": drone_pos[2],
            "waypoint_x": next_waypoint[0],
            "waypoint_y": next_waypoint[1],
            "waypoint_z": next_waypoint[2],
            
            # Control metrics
            "action_magnitude": action_magnitude,
            "policy_length": len(policy) if policy else 0,
            "control_error": np.linalg.norm(np.array(next_waypoint) - np.array(drone_pos)) if previous_step_waypoint is not None else 0,
            "velocity_x": drone_state.linear_velocity.x_val,
            "velocity_y": drone_state.linear_velocity.y_val,
            "velocity_z": drone_state.linear_velocity.z_val,
            "speed": np.linalg.norm([drone_state.linear_velocity.x_val, 
                                   drone_state.linear_velocity.y_val,
                                   drone_state.linear_velocity.z_val]),
              # Active inference metrics (internal model)
            "vfe": vfe,
            "efe": efe,
            "delta_vfe": delta_vfe,
            "efe_vs_vfe_gap": efe_vs_vfe_gap,
            "efe_pragmatic": efe_pragmatic,
            "efe_epistemic": efe_epistemic,
            "suitability": suitability,
            "suitability_std": suitability_std,
            "precision_factor": policy[0].get("precision_factor", 1.0) if policy and len(policy) > 0 else 1.0,
            "entropy": policy[0].get("entropy", 0.0) if policy and len(policy) > 0 else 0.0,
            "model_confidence": policy[0].get("confidence", 0.0) if policy and len(policy) > 0 else 0.0,
            "predicted_efe_improvement": policy[0].get("efe_improvement", 0.0) if policy and len(policy) > 0 else 0.0,
            "action_heading_angle_rad": action_heading_angle_rad,
            "action_heading_angle_deg": action_heading_angle_deg,
              # Planning metrics            "waypoint_options_count": policy[0].get("waypoint_options", 0) if policy and len(policy) > 0 else 0,
            "planning_iterations": policy[0].get("iterations", 0) if policy and len(policy) > 0 else 0,
            "planning_time_ms": inference_time_ms,
            "replanning_occurred": replanning_count > 0,
            "dynamic_replanning": dynamic_replanning_count > 0,
            "replanning_triggered_reason": last_replanning_reason,
            "direct_path_clear": observation.get("direct_path_clear", False),
            "direct_path_suitability": observation.get("direct_path_suitability", 0.0),
            
            # Environment state metrics
            "collision": collision_info.has_collided,
            "obstacles_count": len(obstacle_positions),
            "closest_obstacle": closest_obstacle_dist if obstacle_distances else float('inf'),
            "obstacle_density": len(obstacle_positions) / DENSITY_RADIUS if len(obstacle_positions) > 0 else 0.0,
            "high_density_area": high_density_area,
            "near_goal": near_goal,
            "time_since_movement": time_since_movement
        }
        
        # Store the current distance for next iteration
        previous_distance_to_target = distance_to_target
        # Store current waypoint for next iteration
        previous_step_waypoint = next_waypoint
        step_metrics.append(step_data)
        
        # Move drone to waypoint with continuous obstacle monitoring
        # Our enhanced move_to_waypoint function now returns obstacle information
        obstacle_detected, obstacle_pos, obstacle_dist = move_to_waypoint(client, drone_pos, next_waypoint, velocity=2)
          # If obstacle detected during movement, trigger dynamic replanning
        if obstacle_detected:
            logging.info(f"Episode {episode_id}: Obstacle detected during movement, triggering immediate replanning")
            last_replanning_reason = "mid_movement_obstacle"
            
            # Update drone position after obstacle detection
            drone_state = client.getMultirotorState().kinematics_estimated
            drone_pos = [
                drone_state.position.x_val,
                drone_state.position.y_val,
                drone_state.position.z_val
            ]
            
            # Refresh obstacle data after detection (get latest obstacle information)
            obstacle_positions, obstacle_distances = scanner.fetch_density_distances()
            
            # Add the newly detected obstacle to our list if it's not already included
            # This ensures the planner explicitly knows about this specific obstacle
            if obstacle_pos is not None:
                # Check if this obstacle is already in our list (within a small threshold)
                obstacle_already_included = False
                threshold = 0.5  # Consider obstacles within 0.5m to be the same
                
                for existing_pos in obstacle_positions:
                    if np.linalg.norm(np.array(existing_pos) - np.array(obstacle_pos)) < threshold:
                        obstacle_already_included = True
                        break
                
                if not obstacle_already_included:
                    logging.info(f"Adding newly detected obstacle at {[round(p, 2) for p in obstacle_pos]} to planning data")
                    obstacle_positions.append(obstacle_pos)
                    if obstacle_dist is not None:
                        obstacle_distances.append(obstacle_dist)
                    else:
                        # If distance is unknown, estimate it as the distance from current position
                        est_dist = np.linalg.norm(np.array(obstacle_pos) - np.array(drone_pos))
                        obstacle_distances.append(est_dist)
                        logging.debug(f"Estimated obstacle distance: {est_dist:.2f}m")
                else:
                    logging.debug("Detected obstacle already included in obstacle list")
              # Put the drone in hover mode before replanning
            logging.info(f"Episode {episode_id}: Putting drone in hover mode before dynamic replanning")
            client.hoverAsync().join()
            time.sleep(0.5)  # Brief pause to stabilize in hover mode
            
            # Get new waypoint with dynamic replanning flag
            next_waypoint, policy, planning_error, replanning_inference_time_ms = get_next_waypoint(needs_replanning=True)
            
            if planning_error:
                status = "planning_failure"
                break
                
            # Immediately move to the new waypoint
            logging.info(f"Episode {episode_id}: Executing replanned path to avoid obstacle")
            # Note: using the updated obstacle data for the next movement
            _, _, _ = move_to_waypoint(client, drone_pos, next_waypoint)
        
        # Record trajectory
        trajectory.append(next_waypoint)
        
        # Brief pause to stabilize
        time.sleep(0.2)
        
        # Check step duration for performance monitoring
        step_duration = time.time() - step_start_time
        if step_duration > 5.0:  # If a step takes more than 5 seconds, it might indicate issues
            logging.warning(f"Episode {episode_id}: Step {step} took {step_duration:.2f}s (unusually long)")
      # Episode ended - calculate final metrics
    episode_duration = time.time() - start_time

    # Final position and distance
    try:
        drone_state = client.getMultirotorState().kinematics_estimated
        final_pos = [
            drone_state.position.x_val,
            drone_state.position.y_val,
            drone_state.position.z_val
        ]
        final_distance = np.linalg.norm(np.array(target_pos) - np.array(final_pos))
    except Exception as e:
        logging.error(f"Error getting final drone position: {e}")
        # Use the last known position if available, otherwise use initial position
        if 'drone_pos' in locals() and drone_pos:
            final_pos = drone_pos
        else:
            final_pos = current_pos
        final_distance = np.linalg.norm(np.array(target_pos) - np.array(final_pos))
        logging.info(f"Using last known position for final metrics: {[round(p, 2) for p in final_pos]}")
    
    # Land the drone
    try:
        land_task = client.landAsync()
        try:
            land_task.join()  # Wait for landing to complete
        except Exception as e:
            logging.error(f"Error during landing: {e}")
            # We'll continue even if landing failed
        client.armDisarm(False)
    except Exception as e:
        logging.error(f"Error attempting to land drone: {e}")
        # Continue with metrics even if landing fails
      # Compile episode summary
    episode_summary = {
        # Basic episode metrics
        "episode_id": episode_id,
        "status": status,
        "steps_taken": len(step_metrics),
        "target_reached": status == "success",
        "goal_status": "success" if status == "success" else 
                      "partial" if final_distance < initial_distance_to_target/2 else 
                      "failed",
        
        # Distance metrics
        "final_distance": final_distance,
        "initial_distance": initial_distance_to_target,
        "distance_improvement": initial_distance_to_target - final_distance,
        "normalized_improvement": (initial_distance_to_target - final_distance) / initial_distance_to_target,
        "distance_improvement_percentage": ((initial_distance_to_target - final_distance) / initial_distance_to_target) * 100.0,
        "efficiency_metric": ((initial_distance_to_target - final_distance) / initial_distance_to_target) / len(step_metrics) if step_metrics else 0,
        
        # Path metrics
        "path_length": sum([m["action_magnitude"] for m in step_metrics]) if step_metrics else 0,
        "path_efficiency": initial_distance_to_target / sum([m["action_magnitude"] for m in step_metrics]) if step_metrics and sum([m["action_magnitude"] for m in step_metrics]) > 0 else 0,
        "avg_velocity": np.mean([m["speed"] for m in step_metrics]) if step_metrics else 0,
        "max_velocity": max([m["speed"] for m in step_metrics]) if step_metrics else 0,
        
        # Active inference metrics
        "avg_vfe": np.mean([m["vfe"] for m in step_metrics]) if step_metrics else 0,
        "avg_efe": np.mean([m["efe"] for m in step_metrics]) if step_metrics else 0,
        "avg_suitability": np.mean([m["suitability"] for m in step_metrics]) if step_metrics else 0,
        "min_suitability": min([m["suitability"] for m in step_metrics]) if step_metrics else 0,
        "avg_model_confidence": np.mean([m.get("model_confidence", 0) for m in step_metrics]) if step_metrics else 0,
        
        # Collision and obstacle metrics
        "collisions": collisions,
        "collisions_per_meter": collisions / sum([m["action_magnitude"] for m in step_metrics]) if step_metrics and sum([m["action_magnitude"] for m in step_metrics]) > 0 else 0,
        "max_obstacle_count": max([m["obstacles_count"] for m in step_metrics]) if step_metrics else 0,
        "avg_obstacle_count": np.mean([m["obstacles_count"] for m in step_metrics]) if step_metrics else 0,
        "min_obstacle_distance": min([m["closest_obstacle"] for m in step_metrics]) if step_metrics else float('inf'),
        
        # Planning metrics
        "replanning_count": replanning_count,
        "dynamic_replanning_count": dynamic_replanning_count,
        "replanning_percentage": (replanning_count / len(step_metrics) * 100) if step_metrics else 0,
        "dynamic_replanning_percentage": (dynamic_replanning_count / len(step_metrics) * 100) if step_metrics else 0,
        "avg_planning_time_ms": np.mean([m["planning_time_ms"] for m in step_metrics]) if step_metrics else 0,
        "max_planning_time_ms": max([m["planning_time_ms"] for m in step_metrics]) if step_metrics else 0,
        
        # Time metrics
        "duration_seconds": episode_duration,
        "avg_step_time": episode_duration / len(step_metrics) if step_metrics else 0,
        "timestamp_completed": time.time(),
        
        # Error states
        "stuck_detected": status == "stuck",
        "oscillation_detected": status == "oscillation",
        "airsim_crash": status == "airsim_crash"
    }
    
    logging.info(f"Episode {episode_id} completed: {status}")
    logging.info(f"  Steps: {len(step_metrics)}, Duration: {episode_duration:.2f}s")
    logging.info(f"  Final distance: {final_distance:.2f}m, Collisions: {collisions}")
    logging.info(f"  Dynamic replanning events: {dynamic_replanning_count}")
    
    return step_metrics, episode_summary

def generate_target_pool(start_pos: List[float], distance_range: Tuple[float, float],
                      client: airsim.MultirotorClient, num_targets: int = 100,
                      max_attempts: int = 300, seed: int = None,
                      ray_checks: int = 7) -> List[List[float]]:
    """Pre-generate a pool of valid target positions for use throughout the experiment
    
    Args:
        start_pos: Starting drone position [x, y, z]
        distance_range: (min_distance, max_distance) in meters
        client: AirSim client instance
        num_targets: Number of targets to generate for the pool
        max_attempts: Maximum sampling attempts per target
        seed: Random seed for deterministic behavior
        ray_checks: Number of rays to use for validating each target
        
    Returns:
        List[List[float]]: List of target positions [x, y, z] in NED coordinates
        
    Raises:
        ValueError: If unable to generate enough valid targets
    """
    logging.info(f"Generating pool of {num_targets} valid target locations...")
    
    # Create a separate random generator for deterministic target generation
    if seed is not None:
        target_rng = random.Random(seed)
    else:
        target_rng = random.Random()
    
    target_pool = []
    total_attempts = 0
    max_total_attempts = max_attempts * num_targets * 2  # Upper bound to prevent infinite loops
    
    # Start time to track performance
    start_time = time.time()
    
    # Try to generate the requested number of targets
    while len(target_pool) < num_targets and total_attempts < max_total_attempts:
        try:
            # For each target, we'll use a different "episode id" to ensure diversity
            # This lets us leverage the existing sample_visible_target logic directly
            fake_episode_id = len(target_pool)
            
            # Get a valid target location
            target_pos = sample_visible_target(
                start_pos,
                distance_range,
                client,
                max_attempts=max(50, max_attempts // num_targets),
                episode_id=fake_episode_id,
                seed=seed,
                ray_checks=ray_checks
            )
            
            if target_pos:
                target_pool.append(target_pos)
                
                # Log progress periodically
                if len(target_pool) % 10 == 0 or len(target_pool) == num_targets:
                    elapsed = time.time() - start_time
                    logging.info(f"Generated {len(target_pool)}/{num_targets} targets "
                                f"in {elapsed:.1f}s ({total_attempts} attempts)")
        
        except Exception as e:
            logging.warning(f"Error generating target {len(target_pool)}: {e}")
        
        total_attempts += 1
    
    # Check if we generated enough targets
    if len(target_pool) < num_targets:
        logging.warning(f"Could only generate {len(target_pool)}/{num_targets} valid targets "
                      f"after {total_attempts} attempts")
        if len(target_pool) == 0:
            raise ValueError("Failed to generate any valid targets for the pool")
    else:
        logging.info(f"Successfully generated pool of {len(target_pool)} targets "
                   f"in {time.time() - start_time:.1f}s")
    
    return target_pool


def run_experiment(config: Dict[str, Any]) -> None:
    """Run a full navigation experiment with multiple episodes
    
    Args:
        config: Experiment configuration
    """
    # Merge with default config
    full_config = {**DEFAULT_CONFIG, **config}
    
    # Check for recovery data
    recovery_file = os.path.join("experiment_results", "recovery_data.json")
    resuming = False
    experiment_dir = None
    start_episode = 0
    all_episodes_metrics = []
    episode_summaries = []
    target_pool = []  # Initialize empty target pool
    
    if os.path.exists(recovery_file):
        try:
            with open(recovery_file, 'r') as f:
                recovery_data = json.load(f)
            
            # Check if recovery data is valid and ask user if they want to resume
            if recovery_data.get("experiment_dir") and os.path.exists(recovery_data.get("experiment_dir")):
                print("\nFound previous experiment recovery data:")
                print(f"  Experiment directory: {recovery_data.get('experiment_dir')}")
                print(f"  Last completed episode: {recovery_data.get('last_episode', 0)}")
                print(f"  Timestamp: {recovery_data.get('timestamp', 'unknown')}")
                
                # Ask user if they want to resume
                resume_input = input("\nDo you want to resume this experiment? (y/n): ").strip().lower()
                if resume_input == 'y' or resume_input == 'yes':
                    resuming = True
                    experiment_dir = recovery_data.get("experiment_dir")
                    start_episode = recovery_data.get("last_episode", 0) + 1
                    
                    # Load existing metrics and summaries
                    metrics_file = os.path.join(experiment_dir, "metrics.csv")
                    summary_file = os.path.join(experiment_dir, "episode_summaries.csv")
                    
                    if os.path.exists(metrics_file):
                        try:
                            metrics_df = pd.read_csv(metrics_file)
                            all_episodes_metrics = metrics_df.to_dict('records')
                            logging.info(f"Loaded {len(all_episodes_metrics)} existing metrics records")
                        except Exception as e:
                            logging.error(f"Error loading existing metrics: {e}")
                    
                    if os.path.exists(summary_file):
                        try:
                            summary_df = pd.read_csv(summary_file)
                            episode_summaries = summary_df.to_dict('records')
                            logging.info(f"Loaded {len(episode_summaries)} existing episode summaries")
                        except Exception as e:
                            logging.error(f"Error loading existing summaries: {e}")
                    
                    # Load config from the experiment
                    config_file = os.path.join(experiment_dir, "config.json")
                    if os.path.exists(config_file):
                        try:
                            with open(config_file, 'r') as f:
                                saved_config = json.load(f)
                                # Update the full_config with the saved configuration
                                full_config.update(saved_config)
                                logging.info(f"Loaded configuration from previous experiment")
                        except Exception as e:
                            logging.error(f"Error loading saved configuration: {e}")
                    
                    logging.info(f"Resuming experiment from episode {start_episode}")
                    print(f"\nResuming experiment from episode {start_episode}...")
                else:
                    # User chose not to resume, remove recovery file
                    os.remove(recovery_file)
                    print("\nStarting new experiment...")
        except Exception as e:
            logging.error(f"Error reading recovery data: {e}")
            # Proceed with a new experiment
    
    # Set up a new experiment if not resuming
    if not resuming:
        # Set up logging for this experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = full_config.get("output_dir", "experiment_results")
        experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
    
    # Configure log file
    log_file = os.path.join(experiment_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Create debug directories if needed
    if full_config.get("debug_raycasting", False):
        debug_dir = os.path.join(experiment_dir, "debug_raycasts")
        os.makedirs(debug_dir, exist_ok=True)
    
    # Set random seed
    random_seed = full_config.get("random_seed")
    if random_seed is not None:
        logging.info(f"Using random seed: {random_seed}")
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Log configuration
    logging.info(f"Starting experiment with configuration:")
    for key, value in full_config.items():
        logging.info(f"  {key}: {value}")
      # Save configuration
    config_file = os.path.join(experiment_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(full_config, f, cls=NumpyJSONEncoder, indent=2)
    
    # Collect and log system information
    if not resuming:
        system_info = collect_system_info()
        system_info_file = os.path.join(experiment_dir, "system_info.json")
        with open(system_info_file, 'w') as f:
            json.dump(system_info, f, cls=NumpyJSONEncoder, indent=2)
        logging.info("System information collected and saved")
    
    # Create all episode metrics lists if not resuming
    if not resuming:
        all_episodes_metrics = []
        episode_summaries = []
    
    # Define a function to save recovery data
    def save_recovery_data(ep_id):
        recovery_data = {
            "experiment_dir": experiment_dir,
            "last_episode": ep_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_episodes": full_config["num_episodes"]
        }
        try:
            with open(os.path.join("experiment_results", "recovery_data.json"), 'w') as f:
                json.dump(recovery_data, f, indent=2)
            logging.info(f"Saved recovery data for episode {ep_id}")
        except Exception as e:
            logging.error(f"Failed to save recovery data: {e}")
    
    # Function to initialize AirSim connection with retry
    def initialize_airsim_with_retry(max_retries=3, retry_delay=5):
        for attempt in range(max_retries):
            try:
                client = airsim.MultirotorClient()
                client.confirmConnection()
                client.enableApiControl(True)
                client.armDisarm(True)
                logging.info("AirSim connection established successfully")
                return client
            except Exception as e:
                logging.error(f"Failed to connect to AirSim (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error("Maximum retry attempts reached. Please ensure AirSim is running.")
                    raise ConnectionError("Failed to connect to AirSim after multiple attempts")
    try:
        # Initialize AirSim client with retry
        client = initialize_airsim_with_retry()
        
        # Initialize scanner
        scanner = Scanner(client)
        
        # Start the Julia server
        julia_server = JuliaServer()
        
        # Precompile Julia packages (only if needed)
        if full_config.get("precompile_julia", True):
            julia_server.precompile_packages()
        
        # Start the server
        server_port = full_config.get("zmq_port", DEFAULT_ZMQ_PORT)
        julia_server.start_server(port=server_port)
        
        # Initialize ZMQ interface
        zmq_interface = ZMQInterface(server_port=server_port)
        
        # Generate pool of target locations for the experiment
        # Only if we're not resuming or if we're resuming but no target pool exists
        if not resuming or not os.path.exists(os.path.join(experiment_dir, "target_pool.json")):
            try:
                # Get initial drone position to use as starting point for generating targets
                drone_pose = client.simGetVehiclePose()
                start_pos = [drone_pose.position.x_val, drone_pose.position.y_val, drone_pose.position.z_val]
                
                # Number of targets to pre-generate (1.5x the number of episodes to have extras)
                num_targets = min(200, int(full_config["num_episodes"] * 1.5))
                
                # Generate the pool
                target_pool = generate_target_pool(
                    start_pos, 
                    full_config["target_distance_range"],
                    client,
                    num_targets=num_targets,
                    max_attempts=300,
                    seed=full_config.get("random_seed"),
                    ray_checks=7
                )
                
                # Save the target pool to file for recovery purposes
                target_pool_file = os.path.join(experiment_dir, "target_pool.json")
                with open(target_pool_file, 'w') as f:
                    json.dump(target_pool, f)
                logging.info(f"Saved pool of {len(target_pool)} targets to {target_pool_file}")
            except Exception as e:
                logging.error(f"Failed to generate target pool: {e}")
                target_pool = []  # Reset to empty if generation fails
                
                # Try again with reduced parameters if the first attempt failed
                try:
                    logging.warning("Retrying target pool generation with reduced parameters...")
                    
                    # Try with fewer targets and simpler validation
                    reduced_targets = min(50, int(full_config["num_episodes"] * 0.75))
                    target_pool = generate_target_pool(
                        start_pos, 
                        full_config["target_distance_range"],
                        client,
                        num_targets=reduced_targets,
                        max_attempts=200,
                        seed=full_config.get("random_seed"),
                        ray_checks=3  # Use fewer ray checks for faster generation
                    )
                    
                    # Save the reduced pool
                    target_pool_file = os.path.join(experiment_dir, "target_pool.json")
                    with open(target_pool_file, 'w') as f:
                        json.dump(target_pool, f)
                    logging.info(f"Saved reduced pool of {len(target_pool)} targets to {target_pool_file}")
                    
                except Exception as retry_error:
                    logging.error(f"Second attempt to generate target pool also failed: {retry_error}")
                    target_pool = []  # Will fall back to on-the-fly generation
        else:
            # Load existing target pool if resuming
            target_pool_file = os.path.join(experiment_dir, "target_pool.json")
            try:
                with open(target_pool_file, 'r') as f:
                    target_pool = json.load(f)
                logging.info(f"Loaded existing pool of {len(target_pool)} targets from {target_pool_file}")
            except Exception as e:
                logging.error(f"Failed to load target pool: {e}")
                target_pool = []
        
        # Run episodes
        for episode_id in range(start_episode, full_config["num_episodes"]):
            try:
                # Check AirSim connection before each episode
                try:
                    client.ping()
                except Exception as e:
                    logging.error(f"Lost connection to AirSim before episode {episode_id}: {e}")
                    logging.info("Attempting to reconnect to AirSim...")
                    client = initialize_airsim_with_retry()
                    scanner = Scanner(client)  # Reinitialize scanner with new client
                
                logging.info(f"Starting episode {episode_id} of {full_config['num_episodes']}")
                  # Run episode
                step_metrics, episode_summary = run_episode(
                    episode_id, client, zmq_interface, scanner, full_config, target_pool
                )
                
                # Track metrics
                if step_metrics:
                    all_episodes_metrics.extend(step_metrics)
                    episode_summaries.append(episode_summary)
                    
                    # Visualize debug data if enabled and available
                    if full_config.get("debug_raycasting", False):
                        debug_file = os.path.join("debug_raycasts", f"raycast_debug_ep{episode_id}.json")
                        if os.path.exists(debug_file):
                            # Move the debug file to the experiment directory
                            experiment_debug_dir = os.path.join(experiment_dir, "debug_raycasts")
                            os.makedirs(experiment_debug_dir, exist_ok=True)
                            experiment_debug_file = os.path.join(experiment_debug_dir, f"raycast_debug_ep{episode_id}.json")
                            try:
                                os.rename(debug_file, experiment_debug_file)
                            except Exception:
                                pass
                            
                            # Visualize raycasts if requested
                            if full_config.get("visualize_raycasts", False) and episode_id % 3 == 0:
                                visualization_duration = full_config.get("visualization_duration", 60.0)
                                visualize_raycasts(client, experiment_debug_file, duration=visualization_duration)
                
                # Export metrics after each episode (to avoid data loss in case of crash)
                if all_episodes_metrics:
                    metrics_file = os.path.join(experiment_dir, "metrics.csv")
                    pd.DataFrame(all_episodes_metrics).to_csv(metrics_file, index=False)
                    
                    summary_file = os.path.join(experiment_dir, "episode_summaries.csv")
                    pd.DataFrame(episode_summaries).to_csv(summary_file, index=False)
                
                # Save recovery data after each successful episode
                save_recovery_data(episode_id)
                
            except Exception as episode_error:
                logging.error(f"Error in episode {episode_id}: {episode_error}")
                logging.error(traceback.format_exc())
                
                # Save recovery data to allow resuming from next episode
                save_recovery_data(episode_id - 1)  # Save the last successful episode
                
                # Check if this is an AirSim connection issue
                if "Connection" in str(episode_error) or "confirmConnection" in str(episode_error) or "ping" in str(episode_error):
                    logging.error("AirSim connection issue detected - experiment will be paused")
                    print("\nAirSim connection lost. The experiment has been paused.")
                    print("You can restart AirSim and run the script again to resume from this point.\n")
                    # Exit the experiment loop but allow for clean shutdown
                    break
        
        # Shut down Julia server
        julia_server.shutdown()
        
        # Final export of all metrics
        if all_episodes_metrics:
            metrics_file = os.path.join(experiment_dir, "metrics.csv")
            pd.DataFrame(all_episodes_metrics).to_csv(metrics_file, index=False)
            
            summary_file = os.path.join(experiment_dir, "episode_summaries.csv")
            pd.DataFrame(episode_summaries).to_csv(summary_file, index=False)
            
            # Create plots
            try:
                plot_metrics(metrics_file, os.path.join(experiment_dir, "metrics_plots"))
                logging.info(f"Metrics plots saved to {os.path.join(experiment_dir, 'metrics_plots')}")
            except Exception as e:
                logging.error(f"Failed to create plots: {e}")
        
        # Remove recovery file if experiment completed successfully
        recovery_file = os.path.join("experiment_results", "recovery_data.json")
        if os.path.exists(recovery_file) and episode_id >= full_config["num_episodes"] - 1:
            try:
                os.remove(recovery_file)
                logging.info("Experiment completed successfully, removed recovery data")
            except Exception as e:
                logging.error(f"Failed to remove recovery file: {e}")
    
    except Exception as e:
        logging.error(f"Experiment error: {e}")
        logging.error(traceback.format_exc())
        # Save recovery data in case of crash
        if 'episode_id' in locals():
            save_recovery_data(episode_id - 1)
    
    logging.info(f"Experiment completed. Results saved to {experiment_dir}")

def main():
    """Main entry point with experiment configuration"""
    print("\n==== Autonomous Drone Navigation Experiment with Active Inference ====")
    
    # Ensure recovery directory exists
    os.makedirs("experiment_results", exist_ok=True)
    
    # Define experiment configuration (could be loaded from a file)
    config = DEFAULT_CONFIG.copy()
    
    # Display configuration
    print("\nExperiment configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run the experiment
    run_experiment(config)

def sample_visible_target(current_pos: List[float], distance_range: Tuple[float, float], 
                          client: airsim.MultirotorClient, max_attempts: int = 150,
                          episode_id: int = 0, seed: int = None, 
                          ray_checks: int = 7) -> List[float]:
    """Sample a random target position within distance bounds that is visible to the drone
    
    Args:
        current_pos: Current drone position [x, y, z]
        distance_range: (min_distance, max_distance) in meters
        client: AirSim client instance
        max_attempts: Maximum sampling attempts before giving up
        episode_id: Current episode ID to influence randomization
        seed: Random seed for deterministic behavior
        ray_checks: Number of rays to use for validating each target
        
    Returns:
        List[float]: Target position [x, y, z] in NED coordinates
        
    Raises:
        ValueError: If no valid target is found within max_attempts
    """
    min_dist, max_dist = distance_range
    
    # Create a separate random generator for deterministic target generation
    if seed is not None:
        # Use episode_id to get different but deterministic targets for each episode
        target_rng = random.Random(seed + episode_id)
    else:
        target_rng = random.Random()
    
    # Adapt the distance range based on the episode
    # This creates a progression of targets with varying distances
    if episode_id > 0:
        # The early episodes get shorter distances, later episodes get longer distances
        episode_factor = min(1.0, episode_id / 10)  # Scales from 0.1 to 1.0 over first 10 episodes
        
        # Adjust the distance range
        min_dist = min_dist + (episode_factor * 5)  # Gradually increase minimum distance
        max_dist = min(max_dist, min_dist + 20 + (episode_factor * 30))  # Gradually increase range span
    
    logging.info(f"Target distance range for episode {episode_id}: {min_dist:.1f}m - {max_dist:.1f}m")
    
    # Verify the drone's position is valid for sampling
    if current_pos[2] > -1.0:  # NED coordinates: Z is negative for altitude
        logging.warning(f"Sampling target with drone too close to ground (alt: {current_pos[2]:.2f}m)")
        # Force a minimum altitude for reliable sampling
        current_pos = [current_pos[0], current_pos[1], -5.0]
        logging.info(f"Adjusted sampling position to ensure minimum altitude: {current_pos}")
    
    valid_targets = []  # Track all valid targets for potential selection
    best_target = None  # Best target based on obstacle clearance
    best_obstacle_clearance = 0.0  # Track the best clearance found
    
    # Distribution of attempts to improve sampling efficiency
    # Start with wider distribution, then focus more narrowly if needed
    attempt_phase = 0
    phase_attempts = [max_attempts // 3, max_attempts // 3, max_attempts // 3 + max_attempts % 3]
    attempted_directions = set()
    
    for attempt in range(max_attempts):
        # Determine which phase we're in based on attempt number
        if attempt >= phase_attempts[0] + phase_attempts[1] and attempt_phase < 2:
            attempt_phase = 2  # Final phase - more targeted sampling
        elif attempt >= phase_attempts[0] and attempt_phase < 1:
            attempt_phase = 1  # Second phase - medium distribution
            
            # If we already have some valid targets, try to refine them instead of new random sampling
            if len(valid_targets) > 0:
                logging.info(f"Phase 2: Found {len(valid_targets)} valid targets, focusing on refining these")
        
        # Sampling strategy based on phase
        if attempt_phase == 0:
            # Phase 1: Wide sampling across all directions
            theta = target_rng.uniform(0, 2 * math.pi)  # Azimuth angle
            phi = target_rng.uniform(math.pi/6, math.pi/2.5)  # Between ~30° and ~70° from vertical
        elif attempt_phase == 1 and valid_targets:
            # Phase 2: Refine existing valid targets
            base_target = target_rng.choice(valid_targets)
            base_direction = np.array(base_target) - np.array(current_pos)
            base_distance = np.linalg.norm(base_direction)
            
            if base_distance > 0:
                # Perturb the direction slightly
                base_direction = base_direction / base_distance
                perturb_angle = target_rng.uniform(-math.pi/6, math.pi/6)  # ±30° perturbation
                
                # Create perturbed direction - rotate around vertical axis
                cos_angle = math.cos(perturb_angle)
                sin_angle = math.sin(perturb_angle)
                
                # Simple rotation around vertical axis (approximation)
                x = base_direction[0] * cos_angle - base_direction[1] * sin_angle
                y = base_direction[0] * sin_angle + base_direction[1] * cos_angle
                z = base_direction[2]
                
                # Normalize direction
                direction = np.array([x, y, z])
                direction = direction / np.linalg.norm(direction)
                
                # Convert back to spherical for consistent processing
                theta = math.atan2(direction[1], direction[0])
                phi = math.acos(direction[2])
            else:
                # Fallback to random sampling if base distance is too small
                theta = target_rng.uniform(0, 2 * math.pi)
                phi = target_rng.uniform(math.pi/6, math.pi/2.5)
        else:
            # Phase 3 or fallback: Try evenly spaced directions to cover more space efficiently
            # Generate points roughly evenly distributed on a sphere using golden spiral method
            n_points = 20  # Number of points for spiral distribution
            i = attempt % n_points
            golden_ratio = (1 + 5**0.5) / 2
            
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / n_points)
            
            # Skip directions too close to already attempted ones
            direction_key = f"{round(theta, 1)}_{round(phi, 1)}"
            if direction_key in attempted_directions:
                continue
            attempted_directions.add(direction_key)
        
        # Convert spherical to cartesian coordinates
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        
        # Sample random distance within range
        if attempt_phase < 2:
            # Normal random distance sampling for early phases
            distance = target_rng.uniform(min_dist, max_dist)
        else:
            # In the final phase, try strategic distances
            # For every third attempt, try the middle of the range which often works well
            if attempt % 3 == 0:
                distance = (min_dist + max_dist) / 2
            # For other attempts, still use random sampling
            else:
                distance = target_rng.uniform(min_dist, max_dist)
        
        # Scale direction vector by distance
        direction = np.array([x, y, z]) * distance
        
        # Calculate target position in NED coordinates
        target_pos = np.array(current_pos) + direction
        
        # Ensure z-coordinate is negative (below ground level in NED)
        target_pos[2] = min(target_pos[2], -2.0)  # Keep at least 2m below ground level
        
        # Convert to Vector3r for line of sight test
        start_vector = airsim.Vector3r(current_pos[0], current_pos[1], current_pos[2])
        target_vector = airsim.Vector3r(target_pos[0], target_pos[1], target_pos[2])
        
        # Flag to track if this target is valid across all ray checks
        target_valid = True
        valid_ray_count = 0
        
        # Use multiple rays with slight perturbations to verify line of sight
        # Adaptive ray pattern based on which phase we're in
        actual_ray_checks = ray_checks
        if attempt_phase == 2:
            # Use more rays in the final phase for more thorough testing
            actual_ray_checks = ray_checks + 2
            
        for ray_idx in range(actual_ray_checks):
            try:
                # For the first ray, test direct line of sight
                if ray_idx == 0:
                    # Use simTestLineOfSightBetweenPoints for better reliability
                    los_result = client.simTestLineOfSightBetweenPoints(start_vector, target_vector)
                else:
                    # For additional rays, add small perturbation with increasing radius
                    # Scale perturbation with ray index for better coverage
                    perturb_base = 0.1  # Base 10cm perturbation
                    perturb_scale = 1.0 + (ray_idx * 0.15)  # Scale up for later rays
                    perturb = perturb_base * perturb_scale
                    
                    # Use spiral pattern for better spatial coverage
                    golden_angle = math.pi * (3 - math.sqrt(5))  # Golden angle
                    theta_offset = ray_idx * golden_angle
                    z_offset = ray_idx / (actual_ray_checks - 1) * 2 - 1  # -1 to 1
                    
                    # Calculate offsets in a spiral pattern
                    r_xy = math.sqrt(1 - z_offset**2) * perturb
                    x_offset = r_xy * math.cos(theta_offset)
                    y_offset = r_xy * math.sin(theta_offset)
                    z_offset = z_offset * perturb * 0.5  # Reduce vertical perturbation
                    
                    offset = np.array([x_offset, y_offset, z_offset])
                    
                    # Create slightly perturbed start and end points
                    perturbed_start = airsim.Vector3r(
                        current_pos[0] + offset[0], 
                        current_pos[1] + offset[1], 
                        current_pos[2] + offset[2]
                    )
                    perturbed_target = airsim.Vector3r(
                        target_pos[0] + offset[0], 
                        target_pos[1] + offset[1], 
                        target_pos[2] + offset[2]
                    )
                    
                    # Check line of sight with perturbed rays
                    los_result = client.simTestLineOfSightBetweenPoints(perturbed_start, perturbed_target)
                
                if los_result:
                    valid_ray_count += 1
                    # For faster testing, consider success early if most rays pass
                    if valid_ray_count >= min(5, actual_ray_checks * 0.7):
                        break
                # If any ray test fails, mark the target as invalid
                else:
                    target_valid = False
                    break
                
            except Exception as e:
                logging.warning(f"Error in line of sight test for ray {ray_idx}: {e}")
                target_valid = False
                break
                
        # Consider target valid only if enough rays were valid
        target_valid = target_valid and (valid_ray_count >= min(3, actual_ray_checks * 0.5))
                
        # If target passed ray checks, proceed with validation
        if target_valid:
            # Target is visible - add to valid targets
            valid_targets.append(target_pos.tolist())
            
            # Check obstacle clearance around target
            try:
                # Use simplified clearance check with line of sight tests in multiple directions
                clearance_dirs = [
                    np.array([1, 0, 0]), np.array([-1, 0, 0]),
                    np.array([0, 1, 0]), np.array([0, -1, 0]),
                    np.array([0, 0, 1]), np.array([0, 0, -1])
                ]
                
                min_obstacle_dist = float('inf')
                
                for direction in clearance_dirs:
                    # Check line of sight in this direction from target
                    # Try distances from 1m to 10m
                    for dist in range(1, 11):
                        check_point = target_pos + (direction * dist)
                        check_vector = airsim.Vector3r(check_point[0], check_point[1], check_point[2])
                        
                        # If line of sight fails, we found the obstacle distance
                        if not client.simTestLineOfSightBetweenPoints(target_vector, check_vector):
                            if dist < min_obstacle_dist:
                                min_obstacle_dist = dist
                            break
                
                # If we found a valid obstacle distance
                if min_obstacle_dist < float('inf'):
                    # If this target has better clearance than our previous best, update it
                    if min_obstacle_dist > best_obstacle_clearance:
                        best_obstacle_clearance = min_obstacle_dist
                        best_target = target_pos.tolist()
                        logging.debug(f"Better target found with {min_obstacle_dist:.2f}m obstacle clearance")
                
                # If we found a really good target (>5m clearance), stop early
                if best_obstacle_clearance > 5.0:
                    logging.info(f"Found excellent target with {best_obstacle_clearance:.2f}m obstacle clearance")
                    return best_target
            except Exception as e:
                logging.debug(f"Error checking obstacle clearance near target: {e}")
            
            # Log that we found a valid target
            logging.debug(f"Found valid target at {[round(p, 2) for p in target_pos.tolist()]} (attempt {attempt+1})")
            
            # If this is the first valid target, we'll use it as our initial best
            if best_target is None:
                best_target = target_pos.tolist()
                logging.info(f"Found first valid target at {[round(p, 2) for p in best_target]}")
            
            # If we've found several valid targets, we can start being selective
            if len(valid_targets) >= 5 and attempt > max_attempts // 2:
                # Return the best target we've found so far
                if best_target:
                    logging.info(f"Found {len(valid_targets)} valid targets, selecting best with {best_obstacle_clearance:.2f}m clearance")
                    return best_target
                # Or just return the last valid target
                return valid_targets[-1]
    
    # After all attempts, return the best target if found
    if best_target:
        logging.info(f"Using best target found with {best_obstacle_clearance:.2f}m clearance")
        return best_target
    
    # Or return any valid target we found
    if valid_targets:
        logging.info(f"No ideal target found, using one of {len(valid_targets)} valid targets")
        return target_rng.choice(valid_targets)  # Random choice from valid targets
    
    # If we couldn't find ANY valid target, try one last direct check with a fixed target
    logging.warning("No valid targets found with random sampling, trying fixed fallback targets")
    
    # Try several fallback directions in priority order
    fallback_directions = [
        np.array([1.0, 0.0, 0.0]),   # Forward
        np.array([0.0, 1.0, 0.0]),   # Right
        np.array([0.0, -1.0, 0.0]),  # Left
        np.array([-1.0, 0.0, 0.0]),  # Behind
        np.array([0.7, 0.7, 0.0]),   # Forward-right
        np.array([0.7, -0.7, 0.0]),  # Forward-left
    ]
    
    # Try multiple distances for each direction
    fallback_distances = [min_dist, (min_dist + max_dist) / 2, max_dist * 0.8]
    
    for direction in fallback_directions:
        for distance in fallback_distances:
            try:
                # Create fallback target
                fallback_target_pos = np.array(current_pos) + (direction * distance)
                fallback_target_pos[2] = min(fallback_target_pos[2], -2.0)  # Keep below ground level
                
                # Check if this fallback target is valid
                fallback_vector3r = airsim.Vector3r(fallback_target_pos[0], fallback_target_pos[1], fallback_target_pos[2])
                fallback_los_result = client.simTestLineOfSightBetweenPoints(start_vector, fallback_vector3r)
                
                if fallback_los_result:
                    logging.info(f"Using fallback target at {[round(p, 2) for p in fallback_target_pos.tolist()]}")
                    return fallback_target_pos.tolist()
            except Exception as e:
                logging.warning(f"Error checking fallback target: {e}")
    
    # Last resort - just return the default target position
    logging.error("Failed to find any valid target, returning default position")
    default_target = [current_pos[0] + 10.0, current_pos[1], current_pos[2]]
    return default_target

def visualize_raycasts(client, raycast_data_file, duration=60.0):
    """Visualize the results of raycasting tests by drawing lines in the AirSim environment
    
    Args:
        client: AirSim client instance
        raycast_data_file: Path to a JSON file containing raycast debug information
        duration: Duration in seconds for visualization markers to remain visible
    """
    try:
        # Load raycast debug data
        with open(raycast_data_file, 'r') as f:
            raycast_data = json.load(f)
        
        # Clear previous visualizations
        client.simFlushPersistentMarkers()
        
        # Draw drone position
        drone_pos = raycast_data.get("drone_pos", [0, 0, 0])
        client.simAddMarker(
            airsim.Vector3r(drone_pos[0], drone_pos[1], drone_pos[2]),
            1.0,  # size
            [0, 0, 255, 255],  # blue
            "drone_position",
            duration  # duration in seconds
        )
        
        # Draw successful rays in green
        for test_result in raycast_data.get("ray_tests", []):
            if test_result.get("valid", False):
                target = test_result.get("target", [0, 0, 0])
                # Draw successful target
                client.simAddMarker(
                    airsim.Vector3r(target[0], target[1], target[2]),
                    0.5,  # size
                    [0, 255, 0, 255],  # green
                    f"valid_target_{target[0]}_{target[1]}_{target[2]}",
                    duration  # duration
                )
                
                # Draw rays for this target
                for ray in test_result.get("rays", []):
                    if ray.get("result", False):
                        origin = ray.get("origin", [0, 0, 0])
                        end = ray.get("end", [0, 0, 0])
                        # Draw successful ray
                        client.simDrawLine(
                            airsim.Vector3r(origin[0], origin[1], origin[2]),
                            airsim.Vector3r(end[0], end[1], end[2]),
                            0.05,  # thickness
                            [0, 255, 0, 255],  # green
                            duration  # duration
                        )
        
        # Draw failed rays in red
        for test_result in raycast_data.get("ray_tests", []):
            if not test_result.get("valid", True):
                target = test_result.get("target", [0, 0, 0])
                # Draw failed target
                client.simAddMarker(
                    airsim.Vector3r(target[0], target[1], target[2]),
                    0.5,  # size
                    [255, 0, 0, 255],  # red
                    f"invalid_target_{target[0]}_{target[1]}_{target[2]}",
                    duration  # duration
                )
                
                # Draw rays for this target
                for ray in test_result.get("rays", []):
                    origin = ray.get("origin", [0, 0, 0])
                    end = ray.get("end", [0, 0, 0])
                    color = [255, 0, 0, 255] if not ray.get("result", True) else [0, 255, 0, 255]
                    # Draw ray
                    client.simDrawLine(
                        airsim.Vector3r(origin[0], origin[1], origin[2]),
                        airsim.Vector3r(end[0], end[1], end[2]),
                        0.05,  # thickness
                        color,
                        duration  # duration
                    )
        
        # Draw all failed targets
        for failed_target in raycast_data.get("failures", []):
            client.simAddMarker(
                airsim.Vector3r(failed_target[0], failed_target[1], failed_target[2]),
                0.3,  # size
                [255, 0, 0, 128],  # red with some transparency
                f"failed_{failed_target[0]}_{failed_target[1]}_{failed_target[2]}",
                duration  # duration
            )
        
        logging.info(f"Visualized raycast debug data from {raycast_data_file} (visible for {duration}s)")
        return True
    except Exception as e:
        logging.error(f"Failed to visualize raycasts: {e}")
        traceback.print_exc()
        return False

def plot_metrics(metrics_file, output_dir):
    """Create visualizations of experiment metrics
    
    Args:
        metrics_file: Path to the CSV file containing metrics data
        output_dir: Directory to save plot images
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metrics
        df = pd.read_csv(metrics_file)
        
        # Group data by episode
        episode_groups = df.groupby('episode_id')
        num_episodes = len(episode_groups)
        
        # Plot 1: Distance to target over steps for each episode
        plt.figure(figsize=(12, 8))
        for episode_id, group in episode_groups:
            plt.plot(group['step'], group['distance_to_target'], label=f'Episode {episode_id}')
        
        plt.title('Distance to Target vs. Steps')
        plt.xlabel('Step')
        plt.ylabel('Distance (m)')
        plt.grid(True, alpha=0.3)
        
        # Only show legend if not too many episodes
        if num_episodes <= 10:
            plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'distance_vs_steps.png'), dpi=150)
        plt.close()
        
        # Plot 2: Normalized distance to target
        plt.figure(figsize=(12, 8))
        for episode_id, group in episode_groups:
            plt.plot(group['step'], group['normalized_distance'], label=f'Episode {episode_id}')
        
        plt.title('Normalized Distance to Target vs. Steps')
        plt.xlabel('Step')
        plt.ylabel('Normalized Distance')
        plt.grid(True, alpha=0.3)
        
        # Add a reference line at 0
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Only show legend if not too many episodes
        if num_episodes <= 10:
            plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'normalized_distance_vs_steps.png'), dpi=150)
        plt.close()
        
        # Plot 3: Obstacle count over steps
        plt.figure(figsize=(12, 8))
        for episode_id, group in episode_groups:
            plt.plot(group['step'], group['obstacles_count'], label=f'Episode {episode_id}')
        
        plt.title('Obstacle Count vs. Steps')
        plt.xlabel('Step')
        plt.ylabel('Number of Obstacles')
        plt.grid(True, alpha=0.3)
        
        # Only show legend if not too many episodes
        if num_episodes <= 10:
            plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'obstacles_vs_steps.png'), dpi=150)
        plt.close()
          # Plot 4: Planning time over steps
        plt.figure(figsize=(12, 8))
        for episode_id, group in episode_groups:
            plt.plot(group['step'], group['planning_time_ms'], label=f'Episode {episode_id}')
        
        plt.title('Planning Time vs. Steps')
        plt.xlabel('Step')
        plt.ylabel('Planning Time (ms)')
        plt.grid(True, alpha=0.3)
          # Only show legend if not too many episodes
        if num_episodes <= 10:
            plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'planning_time_vs_steps.png'), dpi=150)
        plt.close()
        
        # Plot 5: Action magnitude over steps
        plt.figure(figsize=(12, 8))
        for episode_id, group in episode_groups:
            plt.plot(group['step'], group['action_magnitude'], label=f'Episode {episode_id}')
        
        plt.title('Action Magnitude vs. Steps')
        plt.xlabel('Step')
        plt.ylabel('Action Magnitude (m)')
        plt.grid(True, alpha=0.3)
        
        # Only show legend if not too many episodes
        if num_episodes <= 10:
            plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'action_magnitude_vs_steps.png'), dpi=150)
        plt.close()
        
        # Plot 6: 2D trajectory for each episode (top-down view)
        for episode_id, group in episode_groups:
            plt.figure(figsize=(10, 10))
            plt.plot(group['position_x'], group['position_y'], 'b-', linewidth=2)
            plt.scatter(group['position_x'].iloc[0], group['position_y'].iloc[0], c='g', s=100, marker='o', label='Start')
            plt.scatter(group['position_x'].iloc[-1], group['position_y'].iloc[-1], c='r', s=100, marker='x', label='End')
            
            # Add waypoints as dots
            plt.scatter(group['waypoint_x'], group['waypoint_y'], c='gray', s=30, alpha=0.5, label='Waypoints')
            
            plt.title(f'Episode {episode_id} Trajectory (Top-Down View)')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')  # Equal aspect ratio
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, f'trajectory_episode_{episode_id}.png'), dpi=150)
            plt.close()
        
        # Plot 7: Summary metrics across episodes
        if num_episodes > 1:
            # Collect episode summary metrics
            episode_metrics = {
                'episode_id': [],
                'steps_taken': [],
                'final_distance': [],
                'success': []
            }
            
            for episode_id, group in episode_groups:
                episode_metrics['episode_id'].append(episode_id)
                episode_metrics['steps_taken'].append(len(group))
                episode_metrics['final_distance'].append(group['distance_to_target'].iloc[-1])
                # Consider success as getting close to target (within ARRIVAL_THRESHOLD)
                episode_metrics['success'].append(1 if group['distance_to_target'].iloc[-1] <= ARRIVAL_THRESHOLD else 0)
            
            summary_df = pd.DataFrame(episode_metrics)
            
            # Plot steps taken per episode
            plt.figure(figsize=(12, 6))
            bars = plt.bar(summary_df['episode_id'], summary_df['steps_taken'])
            
            # Color successful episodes differently
            for i, success in enumerate(summary_df['success']):
                bars[i].set_color('green' if success else 'orange')
            
            plt.title('Steps Taken per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(os.path.join(output_dir, 'steps_per_episode.png'), dpi=150)
            plt.close()
            
            # Plot final distance per episode
            plt.figure(figsize=(12, 6))
            bars = plt.bar(summary_df['episode_id'], summary_df['final_distance'])
            
            # Color successful episodes differently
            for i, success in enumerate(summary_df['success']):
                bars[i].set_color('green' if success else 'orange')
            
            plt.title('Final Distance to Target per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Distance (m)')
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(os.path.join(output_dir, 'final_distance_per_episode.png'), dpi=150)
            plt.close()
            
            # Plot success rate
            success_rate = summary_df['success'].mean() * 100
            plt.figure(figsize=(8, 6))
            plt.pie([success_rate, 100 - success_rate], 
                   labels=[f'Success ({success_rate:.1f}%)', f'Failure ({100-success_rate:.1f}%)'],
                   colors=['green', 'orange'], autopct='%1.1f%%', startangle=90)
            plt.title('Success Rate Across All Episodes')
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.savefig(os.path.join(output_dir, 'success_rate.png'), dpi=150)
            plt.close()
        
        logging.info(f"Successfully created {5 + num_episodes + (3 if num_episodes > 1 else 0)} plots in {output_dir}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating plots: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()









