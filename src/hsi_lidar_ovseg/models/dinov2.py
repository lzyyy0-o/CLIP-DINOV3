"""Offline adapter for injected DINOv2-style token backbones."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import FeaturePyramid


def _pair(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, Sequence) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"无法从主干读取 {name}")
    if min(result) <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return result


def _backbone_patch_size(backbone: nn.Module) -> tuple[int, int]:
    value = getattr(backbone, "patch_size", None)
    if value is None and hasattr(backbone, "patch_embed"):
        value = getattr(backbone.patch_embed, "patch_size", None)
    return _pair(value, "patch_size")


def _backbone_feature_dim(backbone: nn.Module) -> int:
    for name in ("embed_dim", "num_features", "feature_dim"):
        value = getattr(backbone, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("主干必须公开正整数 embed_dim、num_features 或 feature_dim")


def _backbone_blocks(backbone: nn.Module) -> tuple[nn.Module, ...]:
    candidates = [
        getattr(backbone, "blocks", None),
        getattr(backbone, "layers", None),
    ]
    encoder = getattr(backbone, "encoder", None)
    if encoder is not None:
        candidates.append(getattr(encoder, "layer", None))
    for candidate in candidates:
        if isinstance(candidate, (nn.ModuleList, nn.Sequential, list, tuple)):
            blocks = tuple(candidate)
            if blocks and all(isinstance(block, nn.Module) for block in blocks):
                return blocks
    raise ValueError("设置 unfreeze_blocks 时主干必须公开 blocks、layers 或 encoder.layer")


class _TokenPyramidAdapter(nn.Module):
    out_strides = (4, 8, 16, 32)

    def __init__(
        self,
        backbone: nn.Module,
        feature_blocks: tuple[int, int, int, int],
        feature_dim: int,
        *,
        frozen: bool,
        unfreeze_blocks: int,
    ) -> None:
        super().__init__()
        if len(feature_blocks) != 4:
            raise ValueError("feature_blocks 必须包含四个层编号")
        if feature_dim <= 0:
            raise ValueError("feature_dim 必须为正整数")
        self.backbone = backbone
        self.feature_blocks = feature_blocks
        self.feature_dim = feature_dim
        self.frozen = frozen
        self.unfreeze_blocks = unfreeze_blocks
        self.patch_size = _backbone_patch_size(backbone)
        input_dim = _backbone_feature_dim(backbone)
        self.out_channels = (feature_dim,) * 4
        self.projections = nn.ModuleList(
            nn.Conv2d(input_dim, feature_dim, 1) for _ in feature_blocks
        )
        self._frozen_blocks: tuple[nn.Module, ...] = ()
        if frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        elif unfreeze_blocks:
            blocks = _backbone_blocks(backbone)
            if unfreeze_blocks > len(blocks):
                raise ValueError(f"unfreeze_blocks={unfreeze_blocks} 超过主干块数 {len(blocks)}")
            self.backbone.requires_grad_(False)
            for block in blocks[-unfreeze_blocks:]:
                block.requires_grad_(True)
            self._frozen_blocks = blocks[:-unfreeze_blocks]

    def _extract(self, inputs: Tensor) -> tuple[Tensor, ...]:
        raise NotImplementedError

    def _tokens_to_grid(self, tokens: Tensor, inputs: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"主干中间特征必须为 NTC 张量, 实际形状为 {tuple(tokens.shape)}")
        grid_height = inputs.shape[-2] // self.patch_size[0]
        grid_width = inputs.shape[-1] // self.patch_size[1]
        expected_tokens = grid_height * grid_width
        if tokens.shape[1] == expected_tokens + 1:
            tokens = tokens[:, 1:]
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"主干返回 {tokens.shape[1]} 个令牌, 无法还原为 {grid_height}x{grid_width} 网格"
            )
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], grid_height, grid_width
        )

    def forward(self, inputs: Tensor) -> FeaturePyramid:
        if inputs.ndim != 4:
            raise ValueError(f"输入必须为 NCHW 张量, 实际形状为 {tuple(inputs.shape)}")
        if inputs.shape[-2] % self.patch_size[0] or inputs.shape[-1] % self.patch_size[1]:
            raise ValueError("输入空间尺寸必须能被主干 patch_size 整除")
        if self.frozen:
            with torch.no_grad():
                raw_features = self._extract(inputs)
        else:
            raw_features = self._extract(inputs)
        if len(raw_features) != 4:
            raise ValueError(f"主干必须返回四层中间特征, 实际返回 {len(raw_features)} 层")

        outputs: list[Tensor] = []
        for raw, projection, stride in zip(
            raw_features, self.projections, self.out_strides, strict=True
        ):
            grid = projection(self._tokens_to_grid(raw, inputs))
            target_size = (max(1, inputs.shape[-2] // stride), max(1, inputs.shape[-1] // stride))
            outputs.append(
                functional.interpolate(grid, size=target_size, mode="bilinear", align_corners=False)
            )
        return tuple(outputs)  # type: ignore[return-value]

    def train(self, mode: bool = True) -> _TokenPyramidAdapter:
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        else:
            for block in self._frozen_blocks:
                block.eval()
        return self


class DinoV2Adapter(_TokenPyramidAdapter):
    """Convert DINOv2 intermediate tokens into a four-level feature pyramid."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_blocks: tuple[int, int, int, int],
        feature_dim: int,
        frozen: bool = True,
        unfreeze_blocks: int = 0,
    ) -> None:
        if frozen and unfreeze_blocks:
            raise ValueError("frozen=true 时 unfreeze_blocks 必须为 0")
        super().__init__(
            backbone,
            feature_blocks,
            feature_dim,
            frozen=frozen,
            unfreeze_blocks=unfreeze_blocks,
        )
        if not callable(getattr(backbone, "get_intermediate_layers", None)):
            raise ValueError("DINOv2 主干必须实现 get_intermediate_layers")

    def _extract(self, inputs: Tensor) -> tuple[Tensor, ...]:
        features = self.backbone.get_intermediate_layers(
            inputs,
            n=self.feature_blocks,
            reshape=False,
            return_class_token=False,
        )
        return tuple(features)
