import sys
from pathlib import Path
from tqdm import tqdm

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from d_dataset_config import DatasetConfig
from d_dataset_creation import DualModalDatasetCreator
from d_dataset_augmentation import DataAugmenter
from d_dataset_visulize import DualModalVisualizer
from d_dataset_postprocess import cleanup_and_rename_dataset, save_dataset_metadata

def process_split(split_name, version):
    config = DatasetConfig(version=version, split=split_name)
    
    creator = DualModalDatasetCreator(config)
    created_count = creator.run()
    
    augmenter = DataAugmenter(config)
    augmenter.augment_dataset()
    
    return created_count

if __name__ == "__main__":
    version = 1
    splits = ["train", "valid", "test"]
    
    print("🔄 DUAL-MODAL DATASET PROCESSING PIPELINE")
    print("=" * 50)
    
    # Process all splits with progress bar
    total_created = 0
    for split in tqdm(splits, desc="Processing splits"):
        created_count = process_split(split, version)
        total_created += created_count
        tqdm.write(f"✅ {split}: {created_count} files processed")
    
    # Post-processing
    print("\n🔧 POST-PROCESSING...")
    config = DatasetConfig(version=version)
    
    with tqdm(total=2, desc="Finalizing") as pbar:
        cleanup_and_rename_dataset(config.project_root, version)
        pbar.update(1)
        pbar.set_description("Cleaning up")
        
        save_dataset_metadata(config)
        pbar.update(1)
        pbar.set_description("Saving metadata")
    
    print(f"\n✅ PROCESSING COMPLETED!")
    print(f"📊 Total files processed: {total_created}")
    print(f"📁 Dataset location: datasets/Dual-Modal-1504-500-{version}/")