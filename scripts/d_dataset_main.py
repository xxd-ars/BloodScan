from scripts.d_dataset_creation import DualModalDatasetCreator
from scripts.d_dataset_augmentation import DataAugmenter
from scripts.d_dataset_visulize import DualModalVisualizer
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    creator = DualModalDatasetCreator(
        project_root,
        source_jpg_dir = "datasets/Dual-Modal-1504-500-0-mac/test/images",
        rawdata_cropped_dir = "data/rawdata_cropped/class1",
        rawdata_cropped_white_dir = "data/rawdata_cropped_white/class1",
        dest_images_b_dir = "datasets/Dual-Modal-1504-500-1-mac/test/images_b",
        dest_images_w_dir = "datasets/Dual-Modal-1504-500-1-mac/test/images_w")
    creator.run()

    augmenter = DataAugmenter("datasets/Dual-Modal-1504-500-1-mac/test")
    augmenter.augment_dataset()

    visualizer = DualModalVisualizer(target_dataset=Path(project_root / "datasets" / "Dual-Modal-1504-500-1-mac" / "test_augmented_9"))
    visualizer.run()