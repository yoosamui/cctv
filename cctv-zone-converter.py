#!/usr/bin/env python3
"""
cctv-zone.convert.py
A program to draw rectangles on an image and output their coordinates.
"""

import cv2
import numpy as np
import sys
import os

class RectangleDrawer:
    def __init__(self, image_path):
        # Load the image
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Error: Could not load image from {image_path}")
            sys.exit(1)
        
        # Get original dimensions
        self.height, self.width = self.image.shape[:2]
        print(f"Image loaded: {self.width}x{self.height} pixels")
        
        # Create a copy for drawing
        self.display_image = self.image.copy()
        
        # Rectangle drawing variables
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.rectangles = []  # Store all drawn rectangles: (x, y, width, height)
        
        # Create window and set mouse callback
        window_name = "CCTV Zone Converter - Draw Rectangles (Press F5 to clear)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Main loop
        while True:
            cv2.imshow(window_name, self.display_image)
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
                os.system('cls' if os.name == 'nt' else 'clear')
            elif key == ord('h'):  # H key for help
                self.show_help()
        
        cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, param):
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
    
    def update_display(self):
        """Update the display image with all rectangles and the current drawing rectangle"""
        # Start with a fresh copy of the original image
        self.display_image = self.image.copy()
        
        # Draw all saved rectangles
        for rect in self.rectangles:
            x, y, w, h = rect
            cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw the current rectangle being drawn
        if self.drawing and self.start_point[0] != -1 and self.end_point[0] != -1:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    def save_rectangle(self):
        """Calculate rectangle parameters and save them"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Calculate rectangle coordinates (top-left corner, width, height)
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Only save if rectangle has non-zero area
        if width > 0 and height > 0:
            self.rectangles.append((x, y, width, height))
            
            # Output to console
            print(f"Rectangle {len(self.rectangles)}: y={y}, x={x}, width={width}, height={height}")
            print(f"  Format: (y, x, width, height) = ({y}, {x}, {width}, {height})")
            print(f"  Alt format: x={x}, y={y}, w={width}, h={height}")
            print("-" * 50)
    
    def reset_rectangles(self):
        """Clear all rectangles and reload the original image"""
        self.rectangles = []
        self.display_image = self.image.copy()
        self.drawing = False
        print("\n" + "="*50)
        print("ALL RECTANGLES CLEARED! Press F5 to reset (or use 'r' key)")
        print("="*50 + "\n")
    
    def save_coordinates(self):
        """Save all rectangle coordinates to a file"""
        if not self.rectangles:
            print("No rectangles to save!")
            return
        
        filename = "rectangles_output.txt"
        with open(filename, 'w') as f:
            f.write(f"Image: {self.width}x{self.height}\n")
            f.write(f"Total rectangles: {len(self.rectangles)}\n")
            f.write("="*50 + "\n")
            for i, (x, y, w, h) in enumerate(self.rectangles, 1):
                f.write(f"Rectangle {i}: y={y}, x={x}, width={w}, height={h}\n")
        print(f"Coordinates saved to {filename}")
    
    def show_help(self):
        """Display help information"""
        print("\n" + "="*50)
        print("CONTROLS:")
        print("  Mouse: Draw rectangles by clicking and dragging")
        print("  F5 or 'r': Clear all rectangles")
        print("  's': Save coordinates to file (rectangles_output.txt)")
        print("  'c': Clear console output")
        print("  'h': Show this help message")
        print("  'q' or ESC: Quit the program")
        print("="*50 + "\n")

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python cctv-zone.convert.py <image_path>")
        print("Example: python cctv-zone.convert.py image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        sys.exit(1)
    
    # Create and run the rectangle drawer
    drawer = RectangleDrawer(image_path)

if __name__ == "__main__":
    main()
