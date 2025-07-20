import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from d_dataset_config import DatasetConfig
from d_dataset_creation import DualModalDatasetCreator
from d_dataset_augmentation import DataAugmenter
from d_dataset_visulize import DualModalVisualizer

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
    
    for split in splits:
        process_split(split, version)
    
    # # Visualize test split only
    # test_config = DatasetConfig(version=version, split="test")
    # visualizer = DualModalVisualizer(test_config)
    # visualizer.run()