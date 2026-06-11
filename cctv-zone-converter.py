#!/usr/bin/env python3
"""
cctv-zone-convert.py
A program to draw rectangles on an image and output their coordinates.
"""

import cv2
import numpy as np
import sys
import os
import argparse

class RectangleDrawer:
    def __init__(self, image_path):
        # Validate image path before loading
        self.validate_image_path(image_path)
        
        # Load the image
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Error: Could not load image from {image_path}")
            print("The file may be corrupted or in an unsupported format.")
            sys.exit(1)
        
        # Get original dimensions
        self.height, self.width = self.image.shape[:2]
        print(f"✓ Image loaded successfully: {self.width}x{self.height} pixels")
        print(f"  Path: {image_path}")
        print(f"  Channels: {self.image.shape[2] if len(self.image.shape) > 2 else 1}")
        
        # Create a copy for drawing
        self.display_image = self.image.copy()
        
        # Rectangle drawing variables
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.rectangles = []  # Store all drawn rectangles: (x, y, width, height)
        self.running = True  # Control flag for the main loop
        
        # Create window and set mouse callback
        self.window_name = "CCTV Zone Converter - Draw Rectangles (Press ESC or 'q' to quit)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n✓ Ready to draw rectangles!")
        self.show_help()
        
        # Main loop with proper exception handling
        self.run()
    
    def run(self):
        """Main loop with improved keyboard handling"""
        while self.running:
            try:
                cv2.imshow(self.window_name, self.display_image)
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC key to exit
                    break
                elif key == ord('q'):  # Q key to quit
                    break
                elif key == ord('r'):  # R key to reset (alternative)
                    self.reset_rectangles()
                elif key == ord('s'):  # S key to save coordinates to file
                    self.save_coordinates()
                elif key == ord('c'):  # C key to clear console output
                    self.clear_console()
                elif key == ord('h'):  # H key for help
                    self.show_help()
                elif key == ord('f'):  # F key to toggle fullscreen
                    self.toggle_fullscreen()
                    
            except KeyboardInterrupt:
                print("\n\n✓ Program terminated by user (Ctrl+C)")
                break
            except Exception as e:
                print(f"\n✗ Error in main loop: {e}")
                break
        
        # Clean up
        cv2.destroyAllWindows()
    
    def clear_console(self):
        """Clear the console output"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("✓ Console cleared")
        self.show_help()
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 
                              cv2.WINDOW_FULLSCREEN)
    
    def validate_image_path(self, image_path):
        """Validate the image path before attempting to load"""
        # Check if path is empty or None
        if not image_path:
            print("Error: Image path is empty or None")
            sys.exit(1)
        
        # Check if path is a string
        if not isinstance(image_path, str):
            print(f"Error: Image path must be a string, got {type(image_path).__name__}")
            sys.exit(1)
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: '{image_path}'")
            print("Please check the file path and try again.")
            sys.exit(1)
        
        # Check if it's a file (not a directory)
        if not os.path.isfile(image_path):
            print(f"Error: Path is not a file: '{image_path}'")
            sys.exit(1)
        
        # Check file extension
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        if not image_path.lower().endswith(valid_extensions):
            print(f"Warning: File extension may not be a valid image format.")
            print(f"         Supported formats: {', '.join(valid_extensions)}")
            # Don't exit here, let OpenCV try to load it
        
        # Check file size (not empty)
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            print(f"Error: Image file is empty (0 bytes): '{image_path}'")
            sys.exit(1)
        elif file_size < 100:  # Too small to be a valid image
            print(f"Warning: Image file is very small ({file_size} bytes). May be invalid.")
        
        print(f"✓ File validation passed: {os.path.basename(image_path)} ({file_size} bytes)")
    
    def mouse_callback(self, event, x, y, flags, param):
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
                self.end_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.end_point = (x, y)
                    self.update_display()
                    
            elif event == cv2.EVENT_LBUTTONUP:
                if self.drawing:
                    self.end_point = (x, y)
                    self.save_rectangle()
                    self.drawing = False
                    self.update_display()
        except Exception as e:
            print(f"Error in mouse callback: {e}")
    
    def update_display(self):
        """Update the display image with all rectangles and the current drawing rectangle"""
        try:
            # Start with a fresh copy of the original image
            self.display_image = self.image.copy()
            
            # Draw all saved rectangles
            for rect in self.rectangles:
                x, y, w, h = rect
                # Validate rectangle coordinates are within image bounds
                x = max(0, min(x, self.width - 1))
                y = max(0, min(y, self.height - 1))
                w = min(w, self.width - x)
                h = min(h, self.height - y)
                if w > 0 and h > 0:
                    cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Add rectangle number
                    idx = self.rectangles.index(rect) + 1
                    cv2.putText(self.display_image, str(idx), (x + 5, y + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw the current rectangle being drawn
            if self.drawing and self.start_point[0] != -1 and self.end_point[0] != -1:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                # Clamp coordinates to image bounds
                x1 = max(0, min(x1, self.width))
                y1 = max(0, min(y1, self.height))
                x2 = max(0, min(x2, self.width))
                y2 = max(0, min(y2, self.height))
                cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def save_rectangle(self):
        """Calculate rectangle parameters and save them"""
        try:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # Calculate rectangle coordinates (top-left corner, width, height)
            x = min(x1, x2)
            y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Validate rectangle bounds
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            width = min(width, self.width - x)
            height = min(height, self.height - y)
            
            # Only save if rectangle has non-zero area
            if width > 0 and height > 0:
                self.rectangles.append((x, y, width, height))
                
                # Output to console
                print(f"\n✓ Rectangle {len(self.rectangles)} saved:")
                print(f"  y={y}, x={x}, width={width}, height={height}")
                print(f"  Format: (y, x, width, height) = ({y}, {x}, {width}, {height})")
                print(f"  Alt format: x={x}, y={y}, w={width}, h={height}")
                print(f"  Area: {width * height} pixels")
                print("-" * 50)
            else:
                print("✗ Rectangle too small, not saved (minimum 1x1 pixel)")
        except Exception as e:
            print(f"Error saving rectangle: {e}")
    
    def reset_rectangles(self):
        """Clear all rectangles and reload the original image"""
        count = len(self.rectangles)
        self.rectangles = []
        self.display_image = self.image.copy()
        self.drawing = False
        print("\n" + "="*50)
        print(f"✓ ALL RECTANGLES CLEARED! ({count} rectangle(s) removed)")
        print("="*50 + "\n")
    
    def save_coordinates(self):
        """Save all rectangle coordinates to a file"""
        if not self.rectangles:
            print("✗ No rectangles to save!")
            return
        
        filename = "rectangles_output.txt"
        try:
            with open(filename, 'w') as f:
                f.write(f"CCTV Zone Converter - Rectangle Coordinates\n")
                f.write(f"="*50 + "\n")
                f.write(f"Image dimensions: {self.width} x {self.height} pixels\n")
                f.write(f"Total rectangles: {len(self.rectangles)}\n")
                f.write(f"="*50 + "\n\n")
                
                # CSV format header
                f.write("CSV Format (y,x,width,height):\n")
                f.write("y,x,width,height\n")
                
                for i, (x, y, w, h) in enumerate(self.rectangles, 1):
                    f.write(f"{y},{x},{w},{h}\n")
                
                f.write(f"\nDetailed format:\n")
                for i, (x, y, w, h) in enumerate(self.rectangles, 1):
                    f.write(f"Rectangle {i}: y={y}, x={x}, width={w}, height={h}\n")
            
            print(f"\n✓ Coordinates saved to '{filename}'")
            print(f"  Total rectangles: {len(self.rectangles)}")
            print(f"  File location: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"✗ Error saving file: {e}")
    
    def show_help(self):
        """Display help information"""
        print("\n" + "="*50)
        print("CONTROLS:")
        print("  🖱️  Mouse: Draw rectangles by clicking and dragging")
        print("  'r': Clear all rectangles")
        print("  's': Save coordinates to file (rectangles_output.txt)")
        print("  'c': Clear console output")
        print("  'f': Toggle fullscreen mode")
        print("  'h': Show this help message")
        print("  'q' or ESC: Quit the program")
        print("="*50 + "\n")

def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description='CCTV Zone Converter - Draw rectangles on images for CCTV zone configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg
  %(prog)s /path/to/your/image.png
  %(prog)s "C:\\Users\\username\\image.bmp"
  
Output Format:
  When you draw a rectangle, the console will display:
  y, x, width, height (top-left corner coordinates)
        """
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file (supported formats: jpg, jpeg, png, bmp, tiff, webp)'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Additional validation
    if not args.image_path.strip():
        print("Error: Image path cannot be empty or just whitespace")
        sys.exit(1)
    
    # Remove quotes if present (for paths with spaces)
    args.image_path = args.image_path.strip('\'"')
    
    return args

def main():
    """Main entry point with comprehensive parameter checking"""
    print("\n" + "="*50)
    print("CCTV ZONE CONVERTER")
    print("="*50)
    
    # Parse and validate command line arguments
    try:
        args = parse_arguments()
    except SystemExit:
        # argparse will exit on error, just propagate
        raise
    
    # Show the command being executed
    print(f"\nCommand: python {' '.join(sys.argv)}")
    print(f"Image parameter: '{args.image_path}'")
    
    # Create and run the rectangle drawer
    try:
        drawer = RectangleDrawer(args.image_path)
    except KeyboardInterrupt:
        print("\n\n✓ Program terminated by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your image file and try again.")
        sys.exit(1)
    
    print("\n✓ Program terminated normally")

if __name__ == "__main__":
    main()
