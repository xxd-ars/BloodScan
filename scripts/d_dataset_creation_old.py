#!/usr/bin/env python3
"""
Script to convert BMP files from rawdata directories to JPG format based on filename patterns from dual-modal directory.

This script:
1. Reads all JPG files from a given dataset directory
2. Extracts filename prefix (before third underscore)
3. Searches for matching BMP files in rawdata_cropped/class1 and rawdata_cropped_white/class1
4. Converts found BMP files to JPG format and saves to corresponding images_b and images_w directories
"""

import os
import shutil
import glob
from pathlib import Path
from PIL import Image

def extract_filename_prefix(jpg_filename):
    """Extract filename prefix before the third underscore."""
    parts = jpg_filename.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    return jpg_filename

def find_matching_bmp_files(prefix, search_dir):
    """Find BMP files that start with the given prefix in the search directory."""
    if not os.path.exists(search_dir):
        return []
    
    pattern = os.path.join(search_dir, f"{prefix}*.bmp")
    return glob.glob(pattern)

def convert_bmp_to_jpg(bmp_path, jpg_path, quality=100):
    """
    Convert BMP image to JPG format with no pixel compression.
    
    Args:
        bmp_path: Path to source BMP file
        jpg_path: Path to destination JPG file
        quality: JPG quality (100 = no compression)
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        with Image.open(bmp_path) as img:
            # Convert to RGB if necessary
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPG with maximum quality (no compression)
            img.save(jpg_path, 'JPEG', quality=quality, optimize=False)
            return True
    except Exception as e:
        print(f"    Conversion failed: {e}")
        return False

def rename_label_files(source_jpg_dir):
    """
    Rename txt files in the labels folder by removing content after the 4th underscore.
    
    Args:
        source_jpg_dir: Directory containing JPG files (labels folder should be in same parent directory)
    """
    # Get the labels directory path
    parent_dir = os.path.dirname(source_jpg_dir)
    labels_dir = os.path.join(parent_dir, "labels")
    
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return
    
    # Get all txt files in labels directory
    txt_pattern = os.path.join(labels_dir, "*.txt")
    txt_files = glob.glob(txt_pattern)
    
    if not txt_files:
        print(f"No txt files found in {labels_dir}")
        return
    
    print(f"Found {len(txt_files)} txt files in {labels_dir}")
    
    renamed_count = 0
    for txt_file in txt_files:
        txt_filename = os.path.basename(txt_file)
        
        # Split filename by underscore
        parts = txt_filename.split('_')
        
        # If filename has more than 5 parts, keep only first 5 parts + extension
        if len(parts) > 5:
            # Get the extension from the last part
            last_part = parts[-1]
            if '.' in last_part:
                extension = '.' + last_part.split('.')[-1]
            else:
                extension = ''
            
            # Create new filename with first 5 parts + extension
            new_filename = '_'.join(parts[:5]) + extension
            new_path = os.path.join(labels_dir, new_filename)
            
            # Check if new filename already exists
            if os.path.exists(new_path):
                print(f"Target file already exists, skipping: {new_filename}")
                continue
            
            # Rename the file
            os.rename(txt_file, new_path)
            renamed_count += 1
            print(f"Renamed {txt_filename} to {new_filename}")
    
    print(f"Renamed {renamed_count} label files")

def convert_dual_modal_files(source_jpg_dir, 
                             rawdata_cropped_dir, rawdata_cropped_white_dir,
                             dest_images_b_dir, dest_images_w_dir):
    """
    Convert BMP files from rawdata directories to JPG format based on JPG filename patterns.
    
    Args:
        source_jpg_dir: Directory containing JPG files
        rawdata_cropped_dir: Directory containing blue BMP files
        rawdata_cropped_white_dir: Directory containing white BMP files
        dest_images_b_dir: Destination directory for blue JPG files
        dest_images_w_dir: Destination directory for white JPG files
    """
    
    # Ensure destination directories exist
    os.makedirs(dest_images_b_dir, exist_ok=True)
    os.makedirs(dest_images_w_dir, exist_ok=True)
    
    # Get all JPG files from source directory
    jpg_pattern = os.path.join(source_jpg_dir, "*.jpg")
    jpg_files = glob.glob(jpg_pattern)
    
    if not jpg_files:
        print(f"No JPG files found in {source_jpg_dir}")
        return
    
    print(f"Found {len(jpg_files)} JPG files in {source_jpg_dir}")
    
    # Process each JPG file
    converted_b_count = 0
    converted_w_count = 0
    failed_count = 0
    
    for jpg_file in jpg_files:
        jpg_filename = os.path.basename(jpg_file)
        prefix = extract_filename_prefix(jpg_filename)
        
        # Search for matching BMP files in rawdata_cropped
        matching_b_files = find_matching_bmp_files(prefix, rawdata_cropped_dir)
        
        # Search for matching BMP files in rawdata_cropped_white
        matching_w_files = find_matching_bmp_files(prefix, rawdata_cropped_white_dir)
        
        # Convert matching blue BMP files to JPG
        for bmp_file in matching_b_files:
            bmp_filename = os.path.basename(bmp_file)
            jpg_filename = os.path.splitext(bmp_filename)[0] + ".jpg"
            dest_path = os.path.join(dest_images_b_dir, jpg_filename)
            
            if convert_bmp_to_jpg(bmp_file, dest_path):
                converted_b_count += 1
                print(f"Converted {bmp_filename} to {jpg_filename} in images_b")
            else:
                failed_count += 1
                print(f"Failed to convert {bmp_filename}")
        
        # Convert matching white BMP files to JPG
        for bmp_file in matching_w_files:
            bmp_filename = os.path.basename(bmp_file)
            jpg_filename = os.path.splitext(bmp_filename)[0] + ".jpg"
            dest_path = os.path.join(dest_images_w_dir, jpg_filename)
            
            if convert_bmp_to_jpg(bmp_file, dest_path):
                converted_w_count += 1
                print(f"Converted {bmp_filename} to {jpg_filename} in images_w")
            else:
                failed_count += 1
                print(f"Failed to convert {bmp_filename}")
        
        if not matching_b_files and not matching_w_files:
            print(f"No matching BMP files found for prefix: {prefix}")
    
    print(f"\nSummary:")
    print(f"Converted {converted_b_count} files to {dest_images_b_dir}")
    print(f"Converted {converted_w_count} files to {dest_images_w_dir}")
    print(f"Failed conversions: {failed_count}")

if __name__ == "__main__":
    """Main function to run the script."""

    # Configuration - modify these paths as needed
    source_jpg_dir = "datasets/Dual-Modal-1504-500-0/test/images"
    rawdata_cropped_dir = "data/rawdata_cropped/class1"
    rawdata_cropped_white_dir = "data/rawdata_cropped_white/class1"
    dest_images_b_dir = "datasets/Dual-Modal-1504-500-1/test/images_b"
    dest_images_w_dir = "datasets/Dual-Modal-1504-500-1/test/images_w"
    
    # Convert relative paths to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    source_jpg_dir = os.path.join(project_root, source_jpg_dir) if not os.path.isabs(source_jpg_dir) else source_jpg_dir
    rawdata_cropped_dir = os.path.join(project_root, rawdata_cropped_dir) if not os.path.isabs(rawdata_cropped_dir) else rawdata_cropped_dir
    rawdata_cropped_white_dir = os.path.join(project_root, rawdata_cropped_white_dir) if not os.path.isabs(rawdata_cropped_white_dir) else rawdata_cropped_white_dir
    dest_images_b_dir = os.path.join(project_root, dest_images_b_dir) if not os.path.isabs(dest_images_b_dir) else dest_images_b_dir
    dest_images_w_dir = os.path.join(project_root, dest_images_w_dir) if not os.path.isabs(dest_images_w_dir) else dest_images_w_dir
    
    print(f"Source JPG directory: {source_jpg_dir}")
    print(f"Rawdata cropped directory: {rawdata_cropped_dir}")
    print(f"Rawdata cropped white directory: {rawdata_cropped_white_dir}")
    print(f"Destination images_b directory: {dest_images_b_dir}")
    print(f"Destination images_w directory: {dest_images_w_dir}")
    print()
    
    # Rename label files first
    rename_label_files(source_jpg_dir)
    convert_dual_modal_files(source_jpg_dir, 
                             rawdata_cropped_dir, rawdata_cropped_white_dir,
                             dest_images_b_dir, dest_images_w_dir)