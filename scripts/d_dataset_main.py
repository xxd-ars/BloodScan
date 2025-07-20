import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from d_dataset_config import DatasetConfig
from d_dataset_creation import DualModalDatasetCreator
from d_dataset_augmentation import DataAugmenter
from d_dataset_visulize import DualModalVisualizer
from d_dataset_postprocess import cleanup_and_rename_dataset, save_dataset_metadata

def process_split(split_name, version):
    print(f"\n{'='*50}")
    print(f"Processing {split_name} split")
    print(f"{'='*50}")
    
    config = DatasetConfig(version=version, split=split_name)
    
    creator = DualModalDatasetCreator(config)
    creator.run()

    augmenter = DataAugmenter(config)
    augmenter.augment_dataset()

if __name__ == "__main__":
    version = 1
    splits = ["train", "valid", "test"]
    
    print("=" * 80)
    print("DUAL-MODAL DATASET PROCESSING PIPELINE")
    print("=" * 80)
    
    # Process all splits
    for split in splits:
        process_split(split, version)
    
    print(f"\n{'='*80}")
    print("POST-PROCESSING: CLEANUP AND METADATA GENERATION")
    print("=" * 80)
    
    # Cleanup and rename directories
    config = DatasetConfig(version=version)
    cleanup_and_rename_dataset(config.project_root, version)
    
    # Generate dataset metadata
    save_dataset_metadata(config)
    
    print(f"\n{'='*80}")
    print("DATASET PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"Final dataset location: datasets/Dual-Modal-1504-500-{version}/")
    print("=" * 80)
    
    # # Optional: Visualize test split
    # print("\nStarting visualization of test split...")
    # test_config = DatasetConfig(version=version, split="test")
    # visualizer = DualModalVisualizer(test_config)
    # visualizer.run()