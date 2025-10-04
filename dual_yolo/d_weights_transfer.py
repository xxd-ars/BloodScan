#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Optional, Tuple
import os, sys

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "dual_yolo" / "models" / "yolo11x-dseg-id.yaml"
DEFAULT_SOURCES: Dict[str, Path] = {
    "blue": PROJECT_ROOT / "single_yolo" / "runs" / "segment"/ "single_blue_scratch" / "weights" / "best.pt",
    "white": PROJECT_ROOT / "single_yolo" / "runs" / "segment"/ "single_white_scratch" / "weights" / "best.pt",
}
DEFAULT_OUTPUTS: Dict[str, Path] = {
    "blue"  : PROJECT_ROOT / "dual_yolo" / "runs" / "segment" / "dual_modal_scratch_id-blue"  / "weights" / "best.pt",
    "white" : PROJECT_ROOT / "dual_yolo" / "runs" / "segment" / "dual_modal_scratch_id-white" / "weights" / "best.pt",
}
# ===== 手动配置区域 =====
MODE: str = "blue"          # 可选 "blue" 或 "white"
SOURCE: Optional[Path] = None  # 指定单模态权重路径，留空则使用 DEFAULT_SOURCES[MODE]
OUTPUT: Optional[Path] = None  # 指定输出路径，留空则使用 DEFAULT_OUTPUTS[MODE]
MODEL_YAML: Path = DEFAULT_MODEL  # 双模态模型结构文件
# ===== 手动配置结束 =====

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
REVERSE_HEAD_MAPPING = {new: old for old, new in HEAD_MAPPING.items()}


def normalize_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def resolve_source(mode: str, override: Optional[Path]) -> Path:
    source = override if override is not None else DEFAULT_SOURCES[mode]
    source = normalize_path(source)
    if not source.exists():
        raise FileNotFoundError(f"单模态权重不存在: {source}")
    return source


def resolve_output(mode: str, override: Optional[Path]) -> Path:
    output = override if override is not None else DEFAULT_OUTPUTS[mode]
    output = normalize_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def load_single_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"].state_dict()
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    raise RuntimeError(f"不支持的checkpoint结构: {path}")


def map_source_key(target_key: str) -> Optional[str]:
    if not target_key.startswith("model."):
        return None
    parts = target_key.split(".")
    try:
        layer_id = int(parts[1])
    except ValueError:
        return None

    if layer_id <= 10:
        return target_key
    if 11 <= layer_id <= 21:
        return target_key.replace(f"model.{layer_id}.", f"model.{layer_id - 11}.")
    if layer_id in REVERSE_HEAD_MAPPING:
        original_id = REVERSE_HEAD_MAPPING[layer_id]
        return target_key.replace(f"model.{layer_id}.", f"model.{original_id}.")
    return None


def apply_transfer(
    single_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor]
) -> Tuple[int, int]:
    transferred = 0
    skipped = 0
    for key in target_state:
        source_key = map_source_key(key)
        if source_key is None:
            continue
        source_tensor = single_state.get(source_key)
        if source_tensor is None:
            skipped += 1
            continue
        if source_tensor.shape != target_state[key].shape:
            skipped += 1
            continue
        target_state[key] = source_tensor.clone()
        transferred += 1
    return transferred, skipped


def transfer(mode: str, source_path: Path, output_path: Path, model_yaml: Path) -> Path:
    single_state = load_single_state(source_path)
    model = YOLO(normalize_path(model_yaml))
    dual_state = model.model.state_dict()

    transferred, skipped = apply_transfer(single_state, dual_state)
    model.model.load_state_dict(dual_state)

    checkpoint = {
        "model": model.model,
        "optimizer": None,
        "best_fitness": None,
        "epoch": 0,
        "date": None,
    }
    torch.save(checkpoint, output_path)

    print(f"✅ 已转换 {mode} 权重")
    print(f"   source : {source_path}")
    print(f"   output : {output_path}")
    print(f"   copied : {transferred}")
    if skipped:
        print(f"   skipped: {skipped}")
    return output_path


def main() -> None:
    mode = MODE.lower()
    if mode not in DEFAULT_SOURCES:
        raise ValueError(f"MODE 仅支持 'blue' 或 'white'，当前为 {MODE}")

    source_path = resolve_source(mode, SOURCE)
    output_path = resolve_output(mode, OUTPUT)
    model_yaml = normalize_path(MODEL_YAML)
    transfer(mode, source_path, output_path, model_yaml)


if __name__ == "__main__":
    main()