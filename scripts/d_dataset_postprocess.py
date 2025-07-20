#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def cleanup_and_rename_dataset(project_root, version):
    """
    Clean up intermediate directories and rename augmented directories to final names.
    
    Args:
        project_root: Path to project root
        version: Dataset version number
    """
    dataset_base = Path(project_root) / "datasets" / f"Dual-Modal-1504-500-{version}"
    
    splits = ["train", "valid", "test"]
    
    for split in splits:
        # Paths
        intermediate_dir = dataset_base / split
        augmented_dir = dataset_base / f"{split}_augmented_9"
        
        # Check if augmented directory exists
        if not augmented_dir.exists():
            print(f"Warning: Augmented directory not found: {augmented_dir}")
            continue
            
        # Remove intermediate directory if it exists
        if intermediate_dir.exists():
            # print(f"Removing intermediate directory: {intermediate_dir}")
            shutil.rmtree(intermediate_dir)
        
        # Rename augmented directory to final name
        print(f"Renaming {augmented_dir.name} -> {split}")
        augmented_dir.rename(intermediate_dir)
    
    print("Dataset cleanup and rename completed!")

def count_files_in_directory(directory, extension=".jpg"):
    """Count files with given extension in directory."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(extension)])

def generate_dataset_metadata(config):
    """
    Generate comprehensive metadata for the dataset.
    
    Args:
        config: DatasetConfig instance
        
    Returns:
        dict: Metadata dictionary
    """
    metadata = {
        "dataset_info": {
            "version": config.version,
            "created_at": datetime.now().isoformat(),
            "source_dataset": "Dual-Modal-1504-500-0",
            "target_dataset": f"Dual-Modal-1504-500-{config.version}"
        },
        "augmentation_strategies": config.strategies,
        "strategy_count": len(config.strategies),
        "splits": {}
    }
    
    # Count files for each split
    splits = ["train", "valid", "test"]
    dataset_base = config.project_root / "datasets" / f"Dual-Modal-1504-500-{config.version}"
    
    for split in splits:
        split_dir = dataset_base / split
        
        # Count original files (from source)
        source_split_dir = config.project_root / "datasets" / "Dual-Modal-1504-500-0" / split / "images"
        original_count = count_files_in_directory(source_split_dir)
        
        # Count augmented files
        images_b_count = count_files_in_directory(split_dir / "images_b")
        images_w_count = count_files_in_directory(split_dir / "images_w")
        labels_count = count_files_in_directory(split_dir / "labels", ".txt")
        
        metadata["splits"][split] = {
            "original_image_count": original_count,
            "augmented_blue_count": images_b_count,
            "augmented_white_count": images_w_count,
            "augmented_label_count": labels_count,
            "augmentation_factor": len(config.strategies)
        }
    
    # Calculate totals
    total_original = sum(split_data["original_image_count"] for split_data in metadata["splits"].values())
    total_augmented = sum(split_data["augmented_blue_count"] for split_data in metadata["splits"].values())
    
    metadata["totals"] = {
        "original_images": total_original,
        "augmented_images_per_modality": total_augmented,
        "total_augmented_images": total_augmented * 2,  # blue + white
        "total_labels": sum(split_data["augmented_label_count"] for split_data in metadata["splits"].values())
    }
    
    return metadata

def save_dataset_metadata(config):
    """Generate and save dataset metadata to JSON file."""
    metadata = generate_dataset_metadata(config)
    
    # Save to dataset directory
    dataset_dir = config.project_root / "datasets" / f"Dual-Modal-1504-500-{config.version}"
    metadata_file = dataset_dir / "dataset_info.json"
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset metadata saved to: {metadata_file}")
    print(f"Total original images: {metadata['totals']['original_images']}")
    print(f"Total augmented images: {metadata['totals']['total_augmented_images']}")
    
    return metadata_file