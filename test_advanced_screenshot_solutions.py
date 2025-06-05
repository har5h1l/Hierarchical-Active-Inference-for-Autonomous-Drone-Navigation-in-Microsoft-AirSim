#!/usr/bin/env python3
"""
Advanced Screenshot Testing for Open3D Voxel Visualization

This script tests multiple approaches to fix the black screenshot issue:
1. Enhanced context management
2. Offscreen rendering
3. Alternative rendering backends
4. Graphics driver compatibility checks
"""

import open3d as o3d
import numpy as np
import time
import os
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_open3d_info():
    """Test Open3D capabilities and environment"""
    print("=== Open3D Environment Information ===")
    print(f"Open3D Version: {o3d.__version__}")
    print(f"CUDA Available: {o3d.core.cuda.device_count() > 0 if hasattr(o3d.core, 'cuda') else 'Unknown'}")
    
    # Check available devices
    try:
        if hasattr(o3d.core, 'Device'):
            cpu_device = o3d.core.Device("CPU:0")
            print(f"CPU Device: {cpu_device}")
    except:
        pass
    
    # Check GUI availability
    print(f"GUI Available: {hasattr(o3d.visualization, 'gui')}")
    
    # Check visualizer methods
    vis_methods = [method for method in dir(o3d.visualization.Visualizer()) 
                   if 'capture' in method.lower() or 'screenshot' in method.lower()]
    print(f"Available capture methods: {vis_methods}")
    
    print()

def create_test_scene(vis):
    """Create a simple test scene"""
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(coord_frame)
    
    # Add a sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    sphere.translate([0, 0, 0])
    vis.add_geometry(sphere)
    
    # Add a cube
    cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    cube.paint_uniform_color([0.0, 1.0, 0.0])  # Green
    cube.translate([3, 0, 0])
    vis.add_geometry(cube)
    
    # Add some point cloud data
    points = np.random.randn(1000, 3) * 2
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    vis.add_geometry(pcd)

def configure_lighting_and_rendering(vis):
    """Configure optimal lighting and rendering settings"""
    render_option = vis.get_render_option()
    
    # Enhanced lighting and rendering
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    render_option.point_size = 3.0
    render_option.line_width = 2.0
    render_option.show_coordinate_frame = True
    render_option.light_on = True
    
    # Additional rendering options
    if hasattr(render_option, 'mesh_shade_option'):
        render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
    
    # Camera setup
    view_control = vis.get_view_control()
    view_control.set_zoom(0.5)
    view_control.set_front([1, 1, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])

def extensive_render_preparation(vis, cycles=20):
    """Extensive rendering preparation"""
    print(f"Performing {cycles} render cycles...")
    for i in range(cycles):
        try:
            if not vis.poll_events():
                print(f"Warning: poll_events returned False at cycle {i}")
                break
            vis.update_renderer()
            time.sleep(0.05)
        except Exception as e:
            print(f"Error in render cycle {i}: {e}")
            break
    
    # Final stabilization
    time.sleep(1.0)
    print("Render preparation complete")

def test_standard_screenshot(vis, filename):
    """Test standard screenshot method"""
    print("Testing standard screenshot method...")
    try:
        extensive_render_preparation(vis)
        vis.capture_screen_image(filename)
        
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"Standard screenshot: {size} bytes")
            return size > 5000
        return False
    except Exception as e:
        print(f"Standard screenshot failed: {e}")
        return False

def test_buffer_screenshot(vis, filename):
    """Test float buffer screenshot method"""
    print("Testing float buffer screenshot method...")
    try:
        extensive_render_preparation(vis)
        
        # Capture float buffer
        buffer = vis.capture_screen_float_buffer(do_render=True)
        if buffer is None or len(buffer) == 0:
            print("Float buffer is empty")
            return False
        
        print(f"Float buffer size: {len(buffer)} elements")
        
        # Try to save buffer as image
        try:
            from PIL import Image
            
            # Estimate dimensions (assume square-ish window)
            total_pixels = len(buffer) // 3  # RGB
            height = int(np.sqrt(total_pixels))
            width = total_pixels // height
            
            print(f"Estimated dimensions: {width}x{height}")
            
            # Reshape and convert
            img_array = np.array(buffer).reshape((height, width, 3))
            img_array = (img_array * 255).astype(np.uint8)
            img_array = np.flipud(img_array)  # Flip for correct orientation
            
            # Save
            img = Image.fromarray(img_array, 'RGB')
            buffer_filename = filename.replace('.png', '_buffer.png')
            img.save(buffer_filename)
            
            if os.path.exists(buffer_filename):
                size = os.path.getsize(buffer_filename)
                print(f"Buffer screenshot: {size} bytes")
                return size > 5000
            
        except ImportError:
            print("PIL not available for buffer method")
        except Exception as e:
            print(f"Buffer processing failed: {e}")
        
        return False
    except Exception as e:
        print(f"Float buffer screenshot failed: {e}")
        return False

def test_offscreen_rendering():
    """Test offscreen rendering approach"""
    print("\n=== Testing Offscreen Rendering ===")
    try:
        # Try to create offscreen renderer
        if hasattr(o3d.visualization.rendering, 'OffscreenRenderer'):
            print("OffscreenRenderer available, testing...")
            
            width, height = 1024, 768
            renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            
            # Create scene
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            
            # Set up material
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.albedo = [1.0, 0.0, 0.0, 1.0]
            mat.shader = "defaultUnlit"
            
            # Add to scene
            renderer.scene.add_geometry("sphere", sphere, mat)
            
            # Set camera
            renderer.scene.camera.look_at([0, 0, 0], [3, 3, 3], [0, 0, 1])
            
            # Render
            img = renderer.render_to_image()
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            offscreen_filename = f"test_offscreen_{timestamp}.png"
            o3d.io.write_image(offscreen_filename, img)
            
            if os.path.exists(offscreen_filename):
                size = os.path.getsize(offscreen_filename)
                print(f"Offscreen rendering successful: {size} bytes")
                return True
            
        else:
            print("OffscreenRenderer not available in this Open3D version")
            
    except Exception as e:
        print(f"Offscreen rendering failed: {e}")
        traceback.print_exc()
    
    return False

def test_graphics_context_issues():
    """Test for graphics context issues"""
    print("\n=== Testing Graphics Context Issues ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test 1: Multiple visualizer instances
    print("Testing multiple visualizer instances...")
    for i in range(3):
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Test Window {i}", width=800, height=600, visible=True)
            
            create_test_scene(vis)
            configure_lighting_and_rendering(vis)
            
            # Test screenshot
            filename = f"test_multi_vis_{i}_{timestamp}.png"
            success = test_standard_screenshot(vis, filename)
            print(f"Visualizer {i} screenshot: {'SUCCESS' if success else 'FAILED'}")
            
            vis.destroy_window()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Visualizer {i} failed: {e}")
    
    # Test 2: Window visibility and focus
    print("\nTesting window visibility and focus...")
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Focus Test", width=1200, height=800, visible=True)
        
        create_test_scene(vis)
        configure_lighting_and_rendering(vis)
        
        # Test immediate screenshot
        filename1 = f"test_immediate_{timestamp}.png"
        success1 = test_standard_screenshot(vis, filename1)
        
        # Wait and test delayed screenshot
        print("Waiting 3 seconds before second screenshot...")
        time.sleep(3.0)
        filename2 = f"test_delayed_{timestamp}.png"
        success2 = test_standard_screenshot(vis, filename2)
        
        print(f"Immediate screenshot: {'SUCCESS' if success1 else 'FAILED'}")
        print(f"Delayed screenshot: {'SUCCESS' if success2 else 'FAILED'}")
        
        vis.destroy_window()
        
    except Exception as e:
        print(f"Window focus test failed: {e}")

def main():
    """Main test function"""
    print("Advanced Screenshot Solution Testing")
    print("====================================")
    
    # Test Open3D environment
    test_open3d_info()
    
    # Test offscreen rendering first (most likely to work)
    offscreen_success = test_offscreen_rendering()
    
    # Test standard visualization with various approaches
    print("\n=== Testing Standard Visualization ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        
        # Try different window configurations
        window_configs = [
            {"visible": True, "width": 1200, "height": 800},
            {"visible": True, "width": 800, "height": 600},
        ]
        
        for i, config in enumerate(window_configs):
            try:
                print(f"\nTesting window configuration {i+1}: {config}")
                
                vis.create_window(
                    window_name=f"Test Window Config {i+1}",
                    **config
                )
                
                # Set up scene
                create_test_scene(vis)
                configure_lighting_and_rendering(vis)
                
                # Test both screenshot methods
                base_filename = f"test_config_{i+1}_{timestamp}"
                
                standard_success = test_standard_screenshot(vis, f"{base_filename}_standard.png")
                buffer_success = test_buffer_screenshot(vis, f"{base_filename}_buffer.png")
                
                print(f"Config {i+1} - Standard: {'SUCCESS' if standard_success else 'FAILED'}, "
                      f"Buffer: {'SUCCESS' if buffer_success else 'FAILED'}")
                
                vis.destroy_window()
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Window config {i+1} failed: {e}")
                continue
    
    except Exception as e:
        print(f"Standard visualization test failed: {e}")
        traceback.print_exc()
    
    # Test graphics context issues
    test_graphics_context_issues()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Offscreen rendering: {'AVAILABLE' if offscreen_success else 'NOT AVAILABLE'}")
    
    # Check if any screenshots were created
    screenshot_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.png')]
    valid_screenshots = []
    
    for filename in screenshot_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            if size > 5000:
                valid_screenshots.append((filename, size))
                print(f"Valid screenshot: {filename} ({size} bytes)")
    
    print(f"\nTotal valid screenshots: {len(valid_screenshots)}")
    
    if len(valid_screenshots) == 0:
        print("\n⚠️  NO VALID SCREENSHOTS GENERATED")
        print("This indicates a fundamental issue with the graphics context or OpenGL setup.")
        print("Possible causes:")
        print("- Running in a headless environment without proper graphics support")
        print("- OpenGL driver issues")
        print("- Virtual machine or remote desktop limitations")
        print("- Windows graphics policy restrictions")
        
        # Additional diagnostics
        print("\n=== Additional Diagnostics ===")
        try:
            import platform
            print(f"Platform: {platform.platform()}")
            print(f"Python version: {platform.python_version()}")
            
            # Check for virtual environment indicators
            import sys
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                print("Running in virtual environment")
            
            # Check display environment
            if 'DISPLAY' in os.environ:
                print(f"DISPLAY: {os.environ['DISPLAY']}")
            else:
                print("No DISPLAY environment variable (may indicate headless)")
                
        except Exception as e:
            print(f"Diagnostic check failed: {e}")
    
    else:
        print(f"\n✅ SUCCESS: {len(valid_screenshots)} valid screenshots generated")
        print("The screenshot functionality is working with at least some methods.")

if __name__ == "__main__":
    main()
