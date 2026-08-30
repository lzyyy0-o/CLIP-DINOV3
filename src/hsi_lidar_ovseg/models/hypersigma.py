"""Offline adapter for an injected HyperSIGMA backbone."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.models.dinov2 import _TokenPyramidAdapter


class HyperSigmaAdapter(_TokenPyramidAdapter):
    """Convert HyperSIGMA intermediate tokens into a feature pyramid."""

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
        input_adapter = getattr(self.backbone, "input_adapter", None)
        gates = getattr(self.backbone, "gates", None)
        if not frozen:
            if isinstance(input_adapter, nn.Module):
                input_adapter.requires_grad_(True)
            if isinstance(gates, nn.Module):
                gates.requires_grad_(True)
        has_forward = callable(getattr(backbone, "forward_intermediates", None))
        has_get = callable(getattr(backbone, "get_intermediate_layers", None))
        if not has_forward and not has_get:
            raise ValueError(
                "HyperSIGMA 主干必须实现 forward_intermediates 或 get_intermediate_layers"
            )

    def _extract(self, inputs: Tensor) -> tuple[Tensor, ...]:
        if callable(getattr(self.backbone, "forward_intermediates", None)):
            features = self.backbone.forward_intermediates(inputs, indices=self.feature_blocks)
        else:
            features = self.backbone.get_intermediate_layers(
                inputs,
                n=self.feature_blocks,
                reshape=False,
                return_class_token=False,
            )
        return tuple(features)

    def forward(self, inputs: Tensor):  # type: ignore[override]
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
        if all(feature.ndim == 3 for feature in raw_features):
            outputs: list[Tensor] = []
            for raw, projection, stride in zip(
                raw_features, self.projections, self.out_strides, strict=True
            ):
                grid = projection(self._tokens_to_grid(raw, inputs))
                target_size = (
                    max(1, inputs.shape[-2] // stride),
                    max(1, inputs.shape[-1] // stride),
                )
                outputs.append(
                    functional.interpolate(
                        grid, size=target_size, mode="bilinear", align_corners=False
                    )
                )
            return tuple(outputs)

        outputs: list[Tensor] = []
        for raw, projection, stride in zip(
            raw_features, self.projections, self.out_strides, strict=True
        ):
            if raw.ndim != 4:
                raise ValueError("HyperSIGMA 中间特征必须均为 NTC 或 NCHW 张量")
            target_size = (max(1, inputs.shape[-2] // stride), max(1, inputs.shape[-1] // stride))
            outputs.append(
                functional.interpolate(
                    projection(raw), size=target_size, mode="bilinear", align_corners=False
                )
            )
        return tuple(outputs)
