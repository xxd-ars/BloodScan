#!/usr/bin/env python3

from pathlib import Path

class DatasetConfig:
    def __init__(self, project_root=None, version=1, split="test", strategies=None):
        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.project_root = Path(project_root)
        self.version = version
        self.split = split
        
        # Source paths
        self.source_dataset = self.project_root / "datasets" / "Dual-Modal-1504-500-0" / split
        self.rawdata_blue = self.project_root / "data" / "rawdata_cropped" / "class1"
        self.rawdata_white = self.project_root / "data" / "rawdata_cropped_white" / "class1"
        
        # Target paths
        self.target_dataset = self.project_root / "datasets" / f"Dual-Modal-1504-500-{version}" / split
        
        # Strategies
        self.strategies = strategies or {
            # 原图 0
            '0': {},
            # 二元组合 1-4
            '1': {'rotation': 5,    'blur': 1.5},
            '2': {'rotation': -5,   'blur': 1.5},
            '3': {'rotation': 5,    'exposure': 0.9},
            '4': {'rotation': -5,   'exposure': 1.1},
            # 三元组合 5-8
            '5': {'rotation': 10,   'brightness': 1.15, 'blur': 1.2},
            '6': {'rotation': -10,  'brightness': 0.85, 'blur': 1.2},
            '7': {'rotation': 5,    'exposure': 0.9,    'blur': 1.2},
            '8': {'rotation': -5,   'exposure': 1.1,    'blur': 1.2},
        }
        
        # Augmented paths
        self.augmented_dataset = str(self.target_dataset) + "_augmented_" + str(len(self.strategies))