#!/usr/bin/env python3
"""
Enhanced screenshot functionality test with better rendering timing.
This addresses the black screenshot issue by ensuring proper visualization content before capture.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from voxel_visualization import VoxelGridVisualizer
    print("✅ VoxelGridVisualizer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import VoxelGridVisualizer: {e}")
    sys.exit(1)

def test_enhanced_screenshot_functionality():
    """Test enhanced screenshot functionality with proper rendering timing"""
    print("\n=== Testing Enhanced Screenshot Functionality ===")
    
    # Create test directory
    test_dir = "enhanced_test_screenshots"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Initialize visualizer with optimal settings
        print("Creating visualizer with optimal settings...")
        visualizer = VoxelGridVisualizer(
            voxel_size=0.5,
            grid_size=50,
            max_path_points=100,
            visualization_range=25.0,
            update_rate=10.0
        )
        
        # Start the visualization
        print("Starting visualization...")
        visualizer.start_visualization("Enhanced Screenshot Test")
        
        # Wait longer for visualization to fully initialize
        print("Waiting for visualization to initialize...")
        time.sleep(5.0)
        
        # Set up more substantial test data to ensure visual content
        print("Setting up test environment...")
        
        # Set initial drone position
        drone_pos = [0.0, 0.0, -5.0]
        visualizer.update_drone_position(drone_pos)
        print(f"Set drone position: {drone_pos}")
        
        # Set target position (farther away for better visual contrast)
        target_pos = [15.0, 10.0, -5.0]
        visualizer.update_target_position(target_pos)
        print(f"Set target position: {target_pos}")
        
        # Create a more comprehensive obstacle field
        print("Creating comprehensive obstacle field...")
        test_obstacles = []
        
        # Create a grid of obstacles
        for x in range(-10, 11, 2):
            for y in range(-10, 11, 2):
                for z in range(-8, -2, 2):
                    # Skip some positions to create interesting patterns
                    if (x + y) % 4 == 0:
                        test_obstacles.append([float(x), float(y), float(z)])
        
        # Add some random obstacles for variety
        import random
        random.seed(42)
        for _ in range(50):
            x = random.uniform(-20, 20)
            y = random.uniform(-20, 20)
            z = random.uniform(-10, 0)
            test_obstacles.append([x, y, z])
        
        print(f"Created {len(test_obstacles)} test obstacles")
        
        # Update obstacles
        visualizer.update_obstacles(test_obstacles)
        
        # Create a visible path by moving the drone
        print("Creating visible drone path...")
        path_positions = [
            [0.0, 0.0, -5.0],
            [2.0, 1.0, -5.5],
            [4.0, 3.0, -6.0],
            [6.0, 6.0, -5.5],
            [8.0, 8.0, -5.0],
            [10.0, 9.0, -4.5]
        ]
        
        for pos in path_positions:
            visualizer.update_drone_position(pos)
            time.sleep(0.2)  # Small delay for smooth path creation
        
        # Wait for all updates to be processed
        print("Allowing time for all visualizations to update...")
        time.sleep(3.0)
        
        # Test multiple screenshots with different timing strategies
        screenshot_tests = [
            {
                "name": "immediate",
                "pre_delay": 0.0,
                "post_render_delay": 0.1,
                "description": "Immediate capture with minimal delay"
            },
            {
                "name": "delayed",
                "pre_delay": 1.0,
                "post_render_delay": 0.5,
                "description": "Delayed capture with extended rendering time"
            },
            {
                "name": "multiple_renders",
                "pre_delay": 0.5,
                "post_render_delay": 0.2,
                "description": "Multiple render cycles before capture"
            }
        ]
        
        successful_screenshots = 0
        
        for i, test in enumerate(screenshot_tests):
            print(f"\n--- Screenshot Test {i+1}: {test['name']} ---")
            print(f"Description: {test['description']}")
            
            # Pre-capture delay
            if test['pre_delay'] > 0:
                print(f"Pre-capture delay: {test['pre_delay']}s")
                time.sleep(test['pre_delay'])
            
            # For multiple renders test, do additional render cycles
            if test['name'] == 'multiple_renders':
                print("Performing multiple render cycles...")
                for _ in range(5):
                    if hasattr(visualizer, 'vis') and visualizer.vis:
                        visualizer.vis.poll_events()
                        visualizer.vis.update_renderer()
                        time.sleep(0.1)
            
            # Generate screenshot path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(test_dir, f"enhanced_test_{test['name']}_{timestamp}.png")
            
            print(f"Attempting to save screenshot: {screenshot_path}")
            
            # Enhanced screenshot capture
            success = False
            try:
                # Ensure the visualizer window is active and rendered
                if hasattr(visualizer, 'vis') and visualizer.vis:
                    # Force multiple render updates
                    for _ in range(3):
                        visualizer.vis.poll_events()
                        visualizer.vis.update_renderer()
                        time.sleep(test['post_render_delay'] / 3)
                    
                    # Final render before capture
                    visualizer.vis.poll_events()
                    visualizer.vis.update_renderer()
                    time.sleep(test['post_render_delay'])
                    
                    # Capture the screenshot
                    visualizer.vis.capture_screen_image(screenshot_path)
                    success = True
                else:
                    print("❌ Visualizer window not available")
                    
            except Exception as e:
                print(f"❌ Error during screenshot capture: {e}")
                success = False
            
            # Check results
            if success and os.path.exists(screenshot_path):
                file_size = os.path.getsize(screenshot_path)
                print(f"✅ Screenshot saved successfully!")
                print(f"   Path: {screenshot_path}")
                print(f"   Size: {file_size} bytes")
                
                # More detailed size analysis
                if file_size > 50000:  # 50KB threshold for meaningful content
                    print("✅ Screenshot file size suggests good content")
                    successful_screenshots += 1
                elif file_size > 5000:  # 5KB threshold for minimal content
                    print("⚠️  Screenshot file size suggests minimal content")
                    successful_screenshots += 0.5
                else:
                    print("❌ Screenshot file size too small - likely black/empty")
            else:
                print("❌ Failed to save screenshot or file not found")
        
        # Summary
        print(f"\n=== Screenshot Test Summary ===")
        print(f"Successful screenshots: {successful_screenshots}/{len(screenshot_tests)}")
        
        if successful_screenshots >= len(screenshot_tests) * 0.8:
            print("✅ Screenshot functionality working well!")
            return True
        elif successful_screenshots >= len(screenshot_tests) * 0.5:
            print("⚠️  Screenshot functionality partially working")
            return True
        else:
            print("❌ Screenshot functionality needs improvement")
            return False
            
    except Exception as e:
        print(f"❌ Error during enhanced screenshot test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        try:
            if 'visualizer' in locals() and visualizer:
                print("Cleaning up visualizer...")
                visualizer.stop_visualization()
                time.sleep(1.0)
                print("Visualizer cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    """Run enhanced screenshot tests"""
    print("🔍 Testing Enhanced Screenshot Functionality\n")
    
    # Set up detailed logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run enhanced test
    test_passed = test_enhanced_screenshot_functionality()
    
    print("\n=== Enhanced Test Results ===")
    if test_passed:
        print("✅ Enhanced screenshot test PASSED!")
        print("\n📝 Key improvements implemented:")
        print("   • Extended initialization wait time (5 seconds)")
        print("   • Comprehensive obstacle field creation")
        print("   • Visible drone path generation")
        print("   • Multiple rendering strategies")
        print("   • Enhanced error handling and diagnostics")
        print("   • Better file size analysis")
    else:
        print("❌ Enhanced screenshot test FAILED!")
        print("\n🔧 Troubleshooting suggestions:")
        print("   • Check if Open3D is properly installed")
        print("   • Verify graphics drivers are up to date")
        print("   • Ensure the visualization window has focus")
        print("   • Try running with different timing parameters")
        
    return test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
