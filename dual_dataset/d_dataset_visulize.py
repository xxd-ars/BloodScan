#!/usr/bin/env python3

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import re
import glob

plt.rcParams['font.sans-serif'] = ['PingFang HK', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimSong', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DualModalVisualizer:
    def __init__(self, config):
        self.config = config
        self.target_dataset = Path(config.target_dataset)
        self.images_b_dir = self.target_dataset / "images_b"
        self.images_w_dir = self.target_dataset / "images_w"
        self.labels_dir = self.target_dataset / "labels"
        self.colors = {0: 'red', 1: 'green', 2: 'blue'}
        
    def load_image(self, image_path):
        img = Image.open(image_path)
        return np.array(img)
        
    def parse_labels(self, label_file, img_width, img_height):
        annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                coords = []
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        x = int(float(parts[i]) * img_width)
                        y = int(float(parts[i + 1]) * img_height)
                        coords.append((x, y))
                if coords:
                    annotations.append((class_id, coords))
        return annotations
        
    def get_corresponding_files(self, blue_image_path):
        blue_filename = os.path.basename(blue_image_path)
        pattern = r'(\d{4}-\d{2}-\d{2}_\d{6})_(\d+)_([A-Z]\d+)_(\d+)(.*)'
        base_name = blue_filename.replace('.jpg', '')
        match = re.match(pattern, base_name)
        
        if not match:
            return None, None
            
        timestamp = match.group(1)
        augmentation_suffix = match.group(5)
        
        white_pattern = f"{timestamp}_*{augmentation_suffix}.jpg"
        white_candidates = glob.glob(str(self.images_w_dir / white_pattern))
        
        if not white_candidates:
            return None, None
            
        white_image_path = white_candidates[0]
        label_filename = blue_filename.replace('.jpg', '.txt')
        label_path = self.labels_dir / label_filename
        
        return white_image_path, str(label_path)
        
    def draw_annotations(self, ax, img, annotations, title):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        
        for class_id, coords in annotations:
            if len(coords) < 3:
                continue
            color = self.colors.get(class_id, 'yellow')
            polygon = patches.Polygon(coords, linewidth=2, edgecolor=color, 
                                    facecolor=color, alpha=0.3)
            ax.add_patch(polygon)
            
            centroid_x = sum(x for x, y in coords) / len(coords)
            centroid_y = sum(y for x, y in coords) / len(coords)
            ax.text(centroid_x, centroid_y, str(class_id), 
                   fontsize=10, fontweight='bold', color='white',
                   ha='center', va='center')
                   
    def visualize_sample(self, blue_image_path=None):
        if blue_image_path is None:
            blue_images = [f for f in os.listdir(self.images_b_dir) if f.endswith('.jpg')]
            if not blue_images:
                print("No images found")
                return
            selected_image = random.choice(blue_images)
            blue_image_path = self.images_b_dir / selected_image
        
        white_image_path, label_path = self.get_corresponding_files(blue_image_path)
        
        if not white_image_path or not label_path:
            print(f"Missing files for {os.path.basename(blue_image_path)}")
            return
            
        if not os.path.exists(white_image_path) or not os.path.exists(label_path):
            print(f"Files not found")
            return
            
        blue_img = self.load_image(blue_image_path)
        white_img = self.load_image(white_image_path)
        
        img_height, img_width = blue_img.shape[:2]
        annotations = self.parse_labels(label_path, img_width, img_height)
        
        if not annotations:
            print("No annotations found")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        self.draw_annotations(ax1, blue_img, annotations, 
                            f'Blue Light\n{os.path.basename(blue_image_path)}')
        self.draw_annotations(ax2, white_img, annotations, 
                            f'White Light\n{os.path.basename(white_image_path)}')
        
        plt.tight_layout()
        plt.show()
        
    def run(self):
        print("Dual Modal Dataset Visualizer")
            
        sample_count = 0
        while True:
            sample_count += 1
            print(f"\nSample {sample_count}:")
            self.visualize_sample()
            
            user_input = input("\nPress Enter for next sample, 'q' to quit: ").strip().lower()
            if user_input == 'q':
                break

if __name__ == "__main__":
    from d_dataset_config import DatasetConfig
    
    config = DatasetConfig(version=1, split="test")
    visualizer = DualModalVisualizer(config, use_augmented=True)
    visualizer.run()