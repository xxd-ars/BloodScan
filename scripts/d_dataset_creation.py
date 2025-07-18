#!/usr/bin/env python3

import os
import glob
from pathlib import Path
from PIL import Image

class DualModalDatasetCreator:
    def __init__(self, 
                 project_root,
                 source_jpg_dir,
                 rawdata_cropped_dir,
                 rawdata_cropped_white_dir,
                 dest_images_b_dir,
                 dest_images_w_dir):
        self.project_root = Path(project_root)
        self.source_jpg_dir = self.project_root / source_jpg_dir
        self.rawdata_cropped_dir = self.project_root / rawdata_cropped_dir
        self.rawdata_cropped_white_dir = self.project_root / rawdata_cropped_white_dir
        self.dest_images_b_dir = self.project_root / dest_images_b_dir
        self.dest_images_w_dir = self.project_root / dest_images_w_dir
        
    def extract_filename_prefix(self, jpg_filename):
        parts = jpg_filename.split('_')
        return '_'.join(parts[:3]) if len(parts) >= 3 else jpg_filename
        
    def find_matching_bmp_files(self, prefix, search_dir):
        if not search_dir.exists():
            return []
        return list(search_dir.glob(f"{prefix}*.bmp"))
        
    def convert_bmp_to_jpg(self, bmp_path, jpg_path, quality=100):
        with Image.open(bmp_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(jpg_path, 'JPEG', quality=quality, optimize=False)
            
    def rename_label_files(self):
        labels_dir = self.source_jpg_dir.parent / "labels"
        if not labels_dir.exists():
            return
            
        for txt_file in labels_dir.glob("*.txt"):
            parts = txt_file.stem.split('_')
            if len(parts) > 5:
                new_filename = '_'.join(parts[:5]) + '.txt'
                new_path = labels_dir / new_filename
                if not new_path.exists():
                    txt_file.rename(new_path)
                    
    def process_conversions(self):
        os.makedirs(self.dest_images_b_dir, exist_ok=True)
        os.makedirs(self.dest_images_w_dir, exist_ok=True)
        
        jpg_files = list(self.source_jpg_dir.glob("*.jpg"))
        if not jpg_files:
            return 0, 0, 0
            
        converted_b = converted_w = failed = 0
        
        for jpg_file in jpg_files:
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
                
        return converted_b, converted_w, failed
        
    def run(self):
        self.rename_label_files()
        b_count, w_count, failed = self.process_conversions()
        print(f"Converted: {b_count} blue, {w_count} white files")
        return b_count + w_count

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    creator = DualModalDatasetCreator(project_root,
                                      source_jpg_dir = "datasets/Dual-Modal-1504-500-0-mac/test/images",
                                      rawdata_cropped_dir = "data/rawdata_cropped/class1",
                                      rawdata_cropped_white_dir = "data/rawdata_cropped_white/class1",
                                      dest_images_b_dir = "datasets/Dual-Modal-1504-500-1-mac/test/images_b",
                                      dest_images_w_dir = "datasets/Dual-Modal-1504-500-1-mac/test/images_w")
    creator.run()