#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "dual_yolo" / "models" / "yolo11x-dseg-id.yaml"
BLUE_SOURCE = PROJECT_ROOT / "single_yolo" / "runs" / "single_blue_scratch" / "weights" / "best.pt"
WHITE_SOURCE = PROJECT_ROOT / "single_yolo" / "runs" / "single_white_scratch" / "weights" / "best.pt"
OUTPUT_PATH = PROJECT_ROOT / "dual_yolo" / "weights" / "dual_blue_white_from_single.pt"

# 如需自定义路径，可修改下方变量
BLUE_OVERRIDE: Optional[Path] = None
WHITE_OVERRIDE: Optional[Path] = None
MODEL_YAML: Path = DEFAULT_MODEL
OUTPUT_OVERRIDE: Optional[Path] = None

HEAD_MAPPING = {
    11: 25,
    12: 26,
    13: 27,
    14: 28,
    15: 29,
    16: 30,
    17: 31,
    18: 32,
    19: 33,
    20: 34,
    21: 35,
    22: 36,
    23: 37,
}


def normalize(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"].state_dict()
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    raise RuntimeError(f"无法解析的 checkpoint: {path}")


def assign(target: Dict[str, torch.Tensor], key: str, value: torch.Tensor) -> bool:
    tensor = target.get(key)
    if tensor is None or tensor.shape != value.shape:
        return False
    target[key] = value.clone()
    return True


def copy_blue_backbone(state: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    success = fail = 0
    for key, value in state.items():
        if not key.startswith("model."):
            continue
        parts = key.split(".")
        try:
            layer_id = int(parts[1])
        except ValueError:
            continue
        if layer_id > 10:
            continue
        if assign(target, key, value):
            success += 1
        else:
            fail += 1
    return success, fail


def copy_blue_head(state: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    success = fail = 0
    for key, value in state.items():
        if not key.startswith("model."):
            continue
        parts = key.split(".")
        try:
            layer_id = int(parts[1])
        except ValueError:
            continue
        if layer_id not in HEAD_MAPPING:
            continue
        mapped_layer = HEAD_MAPPING[layer_id]
        target_key = key.replace(f"model.{layer_id}.", f"model.{mapped_layer}.")
        if assign(target, target_key, value):
            success += 1
        else:
            fail += 1
    return success, fail


def copy_white_backbone(state: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    success = fail = 0
    for key, value in state.items():
        if not key.startswith("model."):
            continue
        parts = key.split(".")
        try:
            layer_id = int(parts[1])
        except ValueError:
            continue
        if layer_id > 10:
            continue
        target_key = key.replace(f"model.{layer_id}.", f"model.{layer_id + 11}.")
        if assign(target, target_key, value):
            success += 1
        else:
            fail += 1
    return success, fail


def main() -> None:
    blue_path = normalize(BLUE_OVERRIDE or BLUE_SOURCE)
    white_path = normalize(WHITE_OVERRIDE or WHITE_SOURCE)
    output_path = normalize(OUTPUT_OVERRIDE or OUTPUT_PATH)
    model_yaml = normalize(MODEL_YAML)

    if not blue_path.exists():
        raise FileNotFoundError(f"蓝光权重不存在: {blue_path}")
    if not white_path.exists():
        raise FileNotFoundError(f"白光权重不存在: {white_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blue_state = load_state(blue_path)
    white_state = load_state(white_path)
    model = YOLO(model_yaml)
    dual_state = model.model.state_dict()

    b_backbone_s, b_backbone_f = copy_blue_backbone(blue_state, dual_state)
    b_head_s, b_head_f = copy_blue_head(blue_state, dual_state)
    w_backbone_s, w_backbone_f = copy_white_backbone(white_state, dual_state)

    model.model.load_state_dict(dual_state)
    checkpoint = {
        "model": model.model,
        "optimizer": None,
        "best_fitness": None,
        "epoch": 0,
        "date": None,
    }
    torch.save(checkpoint, output_path)

    copied = b_backbone_s + b_head_s + w_backbone_s
    skipped = b_backbone_f + b_head_f + w_backbone_f

    print("✅ 已生成双模态预训练权重")
    print(f"   blue source : {blue_path}")
    print(f"   white source: {white_path}")
    print(f"   output      : {output_path}")
    print(f"   copied      : {copied}")
    if skipped:
        print(f"   skipped     : {skipped}")


if __name__ == "__main__":
    main()
