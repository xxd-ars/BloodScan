#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Optional, Tuple
import os, sys

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "dual_yolo" / "models" / "yolo11x-dseg-id-blue.yaml"
BLUE_SOURCE = PROJECT_ROOT / "single_yolo" / "runs" / "single_blue_scratch" / "weights" / "best.pt"
WHITE_SOURCE = PROJECT_ROOT / "single_yolo" / "runs" / "single_white_scratch" / "weights" / "best.pt"
OUTPUT_PATH = PROJECT_ROOT / "dual_yolo" / "weights" / "dual_yolo11x_bw.pt"

# ==== 手动配置区域 ====
BLUE_OVERRIDE: Optional[Path] = None
WHITE_OVERRIDE: Optional[Path] = None
MODEL_YAML: Path = DEFAULT_MODEL
OUTPUT_OVERRIDE: Optional[Path] = None
# ==== 手动配置结束 ====

BACKBONE_MAX_LAYER = 10
WHITE_OFFSET = 11


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


def copy_backbone(
    state: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    layer_offset: int = 0,
) -> Tuple[int, int]:
    success = fail = 0
    for key, value in state.items():
        if not key.startswith("model."):
            continue
        parts = key.split(".")
        try:
            layer_id = int(parts[1])
        except ValueError:
            continue
        if layer_id > BACKBONE_MAX_LAYER:
            continue
        target_key = key if layer_offset == 0 else key.replace(
            f"model.{layer_id}.", f"model.{layer_id + layer_offset}."
        )
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

    blue_success, blue_fail = copy_backbone(blue_state, dual_state, layer_offset=0)
    white_success, white_fail = copy_backbone(white_state, dual_state, layer_offset=WHITE_OFFSET)

    model.model.load_state_dict(dual_state)
    checkpoint = {
        "model": model.model,
        "optimizer": None,
        "best_fitness": None,
        "epoch": 0,
        "date": None,
    }
    torch.save(checkpoint, output_path)

    copied = blue_success + white_success
    skipped = blue_fail + white_fail

    print("✅ 已生成双模态 backbone 预训练权重")
    print(f"   blue source : {blue_path}")
    print(f"   white source: {white_path}")
    print(f"   output      : {output_path}")
    print(f"   copied      : {copied}")
    if skipped:
        print(f"   skipped     : {skipped}")


if __name__ == "__main__":
    main()
