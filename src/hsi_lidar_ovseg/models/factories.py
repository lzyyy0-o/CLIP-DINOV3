"""Offline constructors for locally cloned official encoder repositories."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from torch import nn


class ExternalSourceError(RuntimeError):
    """Raised when a configured third-party source tree is unavailable or incompatible."""


def add_source_path(source_dir: Path, required_child: str) -> Path:
    """Validate a source clone and add its import root without downloading anything."""

    resolved = source_dir.resolve()
    expected = resolved / required_child
    if not resolved.is_dir() or not expected.exists():
        raise ExternalSourceError(
            f"外部源码目录 {resolved} 缺少官方 {required_child} 模块; "
            "请检查 source_dir 是否指向完整克隆仓库"
        )
    root = str(resolved)
    if root not in sys.path:
        sys.path.insert(0, root)
    return resolved


def create_dinov3(*, source_dir: Path, model_name: str, in_channels: int = 3) -> nn.Module:
    """Build an official DINOv3 backbone with no weights and no network access."""

    if in_channels != 3:
        raise ExternalSourceError("DINOv3 官方主干必须以 3 通道构造; 请使用项目通道适配器")
    add_source_path(source_dir, "dinov3")
    try:
        backbones = importlib.import_module("dinov3.hub.backbones")
        factory = getattr(backbones, model_name)
    except (ImportError, AttributeError) as error:
        raise ExternalSourceError(f"DINOv3 不支持模型 {model_name}: {error}") from error
    if not model_name.startswith("dinov3_") or not callable(factory):
        raise ExternalSourceError(f"DINOv3 不支持模型 {model_name}")
    model = factory(pretrained=False)
    if not isinstance(model, nn.Module):
        raise ExternalSourceError(f"DINOv3 工厂 {model_name} 未返回 torch.nn.Module")
    return model
