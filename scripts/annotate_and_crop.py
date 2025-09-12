#!/usr/bin/env python3
import json
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def annotate_and_crop(json_file_path, crop_left=800, crop_top=250, crop_width=1216, crop_height=1504):
    """
    Load JSON annotation, draw points, crop both images and save
    """
    # Load JSON annotation
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get folder path
    folder_path = os.path.dirname(json_file_path)
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    
    # Determine image paths based on channel type
    image_path_blue = data["imagePath"]
    if "T5" in image_path_blue:  # Blue light image
        image_path_normal = image_path_blue.replace("T5", "T3").rsplit("_", 1)[0] + f"_{int(image_path_blue.rsplit('_', 1)[1].split('.')[0]) - 2}.bmp"
    else:  # Normal image
        image_path_normal = image_path_blue
        image_path_blue = image_path_blue.replace("T3", "T5").rsplit("_", 1)[0] + f"_{int(image_path_blue.rsplit('_', 1)[1].split('.')[0]) + 2}.bmp"
    
    # Load images
    image_normal = cv2.imread(os.path.join(folder_path, image_path_normal))
    image_normal = cv2.cvtColor(image_normal, cv2.COLOR_BGR2RGB)
    image_blue = cv2.imread(os.path.join(folder_path, image_path_blue))
    image_blue = cv2.cvtColor(image_blue, cv2.COLOR_BGR2RGB)
    
    # Draw annotations
    for shape in data["shapes"]:
        points = shape["points"][0]
        label = shape["label"]
        x, y = int(points[0]), int(points[1])
        
        cv2.circle(image_normal, (x, y), radius=8, color=(255, 0, 0), thickness=-1)
        cv2.putText(image_normal, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(image_blue, (x, y), radius=8, color=(255, 0, 0), thickness=-1)
        cv2.putText(image_blue, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Create output directory
    output_dir = './annotated_cropped'
    os.makedirs(output_dir, exist_ok=True)
    
    # Crop and save images
    crop_box = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    
    # Convert to PIL for cropping
    pil_normal = Image.fromarray(image_normal)
    pil_blue = Image.fromarray(image_blue)
    
    cropped_normal = pil_normal.crop(crop_box)
    cropped_blue = pil_blue.crop(crop_box)
    
    # Resize to 1504x1504
    cropped_normal = cropped_normal.resize((1504, 1504), Image.LANCZOS)
    cropped_blue = cropped_blue.resize((1504, 1504), Image.LANCZOS)
    
    # Save cropped images
    normal_output = os.path.join(output_dir, f"{base_name}_normal_annotated.jpg")
    blue_output = os.path.join(output_dir, f"{base_name}_blue_annotated.jpg")
    
    cropped_normal.save(normal_output)
    cropped_blue.save(blue_output)
    
    print(f"Annotated and cropped images saved to: {output_dir}")

if __name__ == "__main__":
    class_name = 'class1' # 'class2'
    folder_path = './data/rawdata/' + class_name + '/'
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
    json_path = folder_path + '2022-04-15_084806_41_T5_2438.json'

    crop_left = 800
    crop_top = 250
    crop_width = 1216
    crop_height = 1504

    annotate_and_crop(json_path, crop_left, crop_top, crop_width, crop_height)