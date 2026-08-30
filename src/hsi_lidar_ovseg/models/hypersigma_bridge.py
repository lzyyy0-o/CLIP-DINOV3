"""Bridge official HyperSIGMA spatial and spectral encoders to four dense features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from hsi_lidar_ovseg.config import ConfigError
from hsi_lidar_ovseg.models.input_adapter import ChannelAdapter


def _load_state_dict_strict(module: nn.Module, path: Path, name: str) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise ConfigError(f"无法读取 {name}权重 {path}: {error}") from error
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "model_state"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{name}权重文件不包含状态字典: {path}")
    state_dict = {str(key).removeprefix("module."): value for key, value in payload.items()}
    try:
        module.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ConfigError(f"{name}权重与模型不兼容 {path}: {error}") from error


class HyperSigmaBridge(nn.Module):
    """Apply a learnable spectral adapter and spectral gates to official spatial features."""

    def __init__(
        self,
        *,
        spatial_encoder: nn.Module,
        spectral_encoder: nn.Module,
        input_channels: int,
        pretrained_in_channels: int,
        feature_blocks: tuple[int, int, int, int] = (3, 5, 7, 11),
    ) -> None:
        super().__init__()
        if len(feature_blocks) != 4 or tuple(sorted(feature_blocks)) != feature_blocks:
            raise ValueError("HyperSIGMA feature_blocks 必须包含四个严格递增层编号")
        embed_dim = getattr(spatial_encoder, "embed_dim", None)
        patch_size = getattr(spatial_encoder, "patch_size", None)
        blocks = getattr(spatial_encoder, "blocks", None)
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError("HyperSIGMA 空间分支必须公开正整数 embed_dim")
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("HyperSIGMA 空间分支必须公开正整数 patch_size")
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            raise ValueError("HyperSIGMA 空间分支必须公开 blocks")
        self.spatial_encoder = spatial_encoder
        self.spectral_encoder = spectral_encoder
        self.input_adapter = ChannelAdapter(input_channels, pretrained_in_channels)
        self.feature_blocks = feature_blocks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.blocks = blocks
        spectral_tokens = getattr(spectral_encoder, "NUM_TOKENS", None)
        if not isinstance(spectral_tokens, int) or spectral_tokens <= 0:
            raise ValueError("HyperSIGMA 光谱分支必须公开正整数 NUM_TOKENS")
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(spectral_tokens, embed_dim), nn.Sigmoid())
            for _ in feature_blocks
        )

    def forward_intermediates(
        self, inputs: Tensor, *, indices: Sequence[int]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if tuple(indices) != self.feature_blocks:
            raise ValueError("HyperSIGMA 请求的 feature_blocks 与构造配置不一致")
        adapted = self.input_adapter(inputs)
        spatial_features = self.spatial_encoder.forward_features(adapted, self.patch_size)
        if not isinstance(spatial_features, (tuple, list)) or len(spatial_features) != 5:
            raise ValueError("HyperSIGMA 空间分支必须返回原始输入和四层特征")
        raw_spectral = self.spectral_encoder(adapted)
        if not isinstance(raw_spectral, (tuple, list)) or len(raw_spectral) != 1:
            raise ValueError("HyperSIGMA 光谱分支必须返回单个中间特征")
        spectral_feature = raw_spectral[0]
        if spectral_feature.ndim != 3:
            raise ValueError("HyperSIGMA 光谱特征必须为三维张量")
        spectral_summary = spectral_feature.mean(dim=-1)

        outputs: list[Tensor] = []
        for feature, gate in zip(spatial_features[1:], self.gates, strict=True):
            if feature.ndim != 4 or feature.shape[1] != self.embed_dim:
                raise ValueError("HyperSIGMA 空间特征必须是 embed_dim 对齐的 NCHW 张量")
            scale = gate(spectral_summary).unsqueeze(-1).unsqueeze(-1)
            outputs.append(feature * (1.0 + scale))
        return tuple(outputs)  # type: ignore[return-value]


def load_hypersigma_weights(
    bridge: HyperSigmaBridge, spatial_path: Path, spectral_path: Path
) -> None:
    """Strictly load official HyperSIGMA weights into their matching branch."""

    _load_state_dict_strict(bridge.spatial_encoder, spatial_path, "HyperSIGMA 空间分支")
    _load_state_dict_strict(bridge.spectral_encoder, spectral_path, "HyperSIGMA 光谱分支")
