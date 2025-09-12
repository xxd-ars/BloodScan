#!/usr/bin/env python3
import os
import sys
from PIL import Image
from glob import glob

def crop_images(input_dir, center_x_ratio=0.5, center_y_ratio=0.5, size_ratio=0.8):
    """
    Crop all jpg/png images in input_dir to square shape with specified center and size ratios
    
    Args:
        input_dir: Input directory path
        center_x_ratio: Center X position as ratio of image width (0-1)
        center_y_ratio: Center Y position as ratio of image height (0-1) 
        size_ratio: Square side length as ratio of min(width, height) (0-1)
    """
    output_dir = os.path.join(input_dir, 'cropped')
    os.makedirs(output_dir, exist_ok=True)
    
    patterns = [os.path.join(input_dir, f'*.{ext}') for ext in ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']]
    image_files = []
    for pattern in patterns:
        image_files.extend(glob(pattern))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                
                # Calculate crop parameters
                size = int(min(w, h) * size_ratio)
                center_x = int(w * center_x_ratio)
                center_y = int(h * center_y_ratio)
                
                # Calculate crop box
                left = max(0, center_x - size // 2)
                top = max(0, center_y - size // 2)
                right = min(w, left + size)
                bottom = min(h, top + size)
                
                # Adjust if crop area exceeds image bounds
                if right - left < size:
                    if left == 0:
                        right = min(w, size)
                    else:
                        left = max(0, right - size)
                
                if bottom - top < size:
                    if top == 0:
                        bottom = min(h, size)
                    else:
                        top = max(0, bottom - size)
                
                # Crop and save
                cropped = img.crop((left, top, right, bottom))
                filename = os.path.basename(img_path)
                name, ext = os.path.splitext(filename)
                if ext.lower() in ['.bmp']:
                    filename = name + '.jpg'
                output_path = os.path.join(output_dir, filename)
                cropped.save(output_path)
                
            print(f"Cropped: {filename}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Cropped images saved to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_crop.py <input_dir> [center_x_ratio] [center_y_ratio] [size_ratio]")
        print("Example: python image_crop.py ./images 0.5 0.5 0.8")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    center_x = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    center_y = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    size = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    
    crop_images(input_dir, center_x, center_y, size)

# & C:/Users/ASUS/anaconda3/envs/bloodScan/python.exe "c:/Users/ASUS/Documents/SJTU M2/Graduation Project/BloodScan/scripts/image_crop.py" ./paper\paper\evaluation 0.47 0.6 0.3