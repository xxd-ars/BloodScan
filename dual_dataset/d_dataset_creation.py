#!/usr/bin/env python3

import os
import glob
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

class DualModalDatasetCreator:
    def __init__(self, config):
        self.config = config
        self.source_jpg_dir = config.source_dataset / "images"
        self.rawdata_cropped_dir = config.rawdata_blue
        self.rawdata_cropped_white_dir = config.rawdata_white
        self.dest_images_b_dir = config.target_dataset / "images_b"
        self.dest_images_w_dir = config.target_dataset / "images_w"
        
    def extract_filename_prefix(self, jpg_filename):
        # Extract the base name before the roboflow suffix
        # From: 2022-03-28_103204_1_T5_2348_bmp.rf.xxx.jpg
        # To: 2022-03-28_103204_1_ (include trailing underscore for pattern matching)
        if '_bmp.rf.' in jpg_filename:
            base = jpg_filename.split('_bmp.rf.')[0]
            # Split by underscore and take first 3 parts + underscore
            parts = base.split('_')
            if len(parts) >= 3:
                return '_'.join(parts[:3]) + '_'
        
        # Fallback
        parts = jpg_filename.rsplit('.', 1)[0].split('_')
        if len(parts) >= 3:
            return '_'.join(parts[:3]) + '_'
        return jpg_filename.rsplit('.', 1)[0]
        
    def find_matching_bmp_files(self, prefix, search_dir):
        if not search_dir.exists():
            return []
        
        # Use glob pattern with the prefix ending with underscore
        # This will match both T5 and T3 variants flexibly
        # Example: 2022-03-28_103204_1_ matches both T5 and T3 files
        matching_files = list(search_dir.glob(f"{prefix}*.bmp"))
        return matching_files
        
    def convert_bmp_to_jpg(self, bmp_path, jpg_path, quality=100):
        with Image.open(bmp_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(jpg_path, 'JPEG', quality=quality, optimize=False)
            
    def copy_and_rename_labels(self):
        source_labels_dir = self.source_jpg_dir.parent / "labels"
        dest_labels_dir = self.dest_images_b_dir.parent / "labels"
        
        if not source_labels_dir.exists():
            return
            
        os.makedirs(dest_labels_dir, exist_ok=True)
        txt_files = list(source_labels_dir.glob("*.txt"))
        
        for txt_file in tqdm(txt_files, desc="Copying labels", leave=False):
            parts = txt_file.stem.split('_')
            new_filename = '_'.join(parts[:5]) + '.txt' if len(parts) > 5 else txt_file.name
            dest_path = dest_labels_dir / new_filename
            
            if not dest_path.exists():
                shutil.copy2(txt_file, dest_path)
    
    def process_conversions(self):
        os.makedirs(self.dest_images_b_dir, exist_ok=True)
        os.makedirs(self.dest_images_w_dir, exist_ok=True)
        
        jpg_files = list(self.source_jpg_dir.glob("*.jpg"))
        if not jpg_files:
            return 0, 0, 0
            
        converted_b = converted_w = 0
        
        for jpg_file in tqdm(jpg_files, desc="Converting images"):
            prefix = self.extract_filename_prefix(jpg_file.name)
            
            blue_files = self.find_matching_bmp_files(prefix, self.rawdata_cropped_dir)
            white_files = self.find_matching_bmp_files(prefix, self.rawdata_cropped_white_dir)
            
            for bmp_file in blue_files:
                dest_path = self.dest_images_b_dir / f"{bmp_file.stem}.jpg"
                self.convert_bmp_to_jpg(bmp_file, dest_path)
                converted_b += 1
                
            for bmp_file in white_files:
                dest_path = self.dest_images_w_dir / f"{bmp_file.stem}.jpg"
                self.convert_bmp_to_jpg(bmp_file, dest_path)
                converted_w += 1
                
        return converted_b, converted_w, 0
        
    def run(self):
        self.copy_and_rename_labels()
        b_count, w_count, _ = self.process_conversions()
        return b_count + w_count

if __name__ == "__main__":
    from d_dataset_config import DatasetConfig
    
    config = DatasetConfig(version=1, split="test")
    creator = DualModalDatasetCreator(config)
    creator.run()