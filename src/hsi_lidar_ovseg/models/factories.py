"""Offline constructors for locally cloned official encoder repositories."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from torch import nn

from hsi_lidar_ovseg.models.hypersigma_bridge import HyperSigmaBridge


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


def create_hypersigma(
    *,
    source_dir: Path,
    model_name: str,
    in_channels: int,
    pretrained_in_channels: int,
    feature_blocks: tuple[int, int, int, int] = (3, 5, 7, 11),
) -> HyperSigmaBridge:
    """Build official HyperSIGMA spatial and spectral branches without weights."""

    sizes = {
        "base": (768, 12, 12, 3),
        "large": (1024, 24, 16, 6),
        "huge": (1280, 32, 16, 8),
    }
    if model_name not in sizes:
        raise ExternalSourceError("HyperSIGMA model_name 必须为 base、large 或 huge")
    root = add_source_path(source_dir, "ImageClassification")
    import_root = str(root / "ImageClassification")
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
    try:
        spatial_module = importlib.import_module("model.SpatViT_fusion")
        spectral_module = importlib.import_module("model.SpecViT_fusion")
    except ImportError as error:
        raise ExternalSourceError(
            '无法导入 HyperSIGMA 官方实现; 请安装 pip install -e ".[pretrained]" 所需依赖'
        ) from error
    embed_dim, depth, heads, interval = sizes[model_name]
    common = {
        "img_size": 224,
        "in_chans": pretrained_in_channels,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": heads,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_path_rate": 0.1,
        "use_checkpoint": True,
        "use_abs_pos_emb": False,
        "interval": interval,
        "n_points": 8,
    }
    spatial_encoder = spatial_module.SpatViT(
        patch_size=16,
        num_classes=1,
        out_indices=list(feature_blocks),
        **common,
    )
    spectral_encoder = spectral_module.SpectralVisionTransformer(
        NUM_TOKENS=100,
        out_indices=[feature_blocks[0]],
        **common,
    )
    return HyperSigmaBridge(
        spatial_encoder=spatial_encoder,
        spectral_encoder=spectral_encoder,
        input_channels=in_channels,
        pretrained_in_channels=pretrained_in_channels,
        feature_blocks=feature_blocks,
    )
