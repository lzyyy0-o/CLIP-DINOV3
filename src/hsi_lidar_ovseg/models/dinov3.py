"""Adapters for locally loaded DINOv3 ViT and ConvNeXt backbones."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from hsi_lidar_ovseg.models.dinov2 import _TokenPyramidAdapter
from hsi_lidar_ovseg.models.protocols import FeaturePyramid


class DinoV3ViTAdapter(_TokenPyramidAdapter):
    """Convert DINOv3 ViT intermediate tokens into a four-level pyramid."""

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
            raise ValueError("DINOv3 ViT 主干必须实现 get_intermediate_layers")

    def _extract(self, inputs: Tensor) -> tuple[Tensor, ...]:
        features = self.backbone.get_intermediate_layers(
            inputs,
            n=self.feature_blocks,
            reshape=False,
            return_class_token=False,
        )
        return tuple(features)


class DinoV3ConvNeXtAdapter(nn.Module):
    """Expose DINOv3 ConvNeXt's native four-stage convolutional pyramid."""

    out_strides = (4, 8, 16, 32)

    def __init__(
        self,
        backbone: nn.Module,
        feature_stages: tuple[int, int, int, int] = (0, 1, 2, 3),
        *,
        frozen: bool = True,
        unfreeze_blocks: int = 0,
    ) -> None:
        super().__init__()
        if feature_stages != (0, 1, 2, 3):
            raise ValueError("DINOv3 ConvNeXt 必须使用四个原生阶段 (0, 1, 2, 3)")
        if frozen and unfreeze_blocks:
            raise ValueError("frozen=true 时 unfreeze_blocks 必须为 0")
        if not callable(getattr(backbone, "get_intermediate_layers", None)):
            raise ValueError("DINOv3 ConvNeXt 主干必须实现 get_intermediate_layers")
        embed_dims = getattr(backbone, "embed_dims", None)
        if not isinstance(embed_dims, Sequence) or len(embed_dims) != 4:
            raise ValueError("DINOv3 ConvNeXt 主干必须公开四层 embed_dims")
        channels = tuple(int(value) for value in embed_dims)
        if min(channels) <= 0:
            raise ValueError("DINOv3 ConvNeXt embed_dims 必须为正整数")

        stages = getattr(backbone, "stages", None)
        downsample_layers = getattr(backbone, "downsample_layers", None)
        if not isinstance(stages, (nn.ModuleList, nn.Sequential)) or len(stages) != 4:
            raise ValueError("DINOv3 ConvNeXt 主干必须公开四个 stages")
        if (
            not isinstance(downsample_layers, (nn.ModuleList, nn.Sequential))
            or len(downsample_layers) != 4
        ):
            raise ValueError("DINOv3 ConvNeXt 主干必须公开四个 downsample_layers")
        if unfreeze_blocks < 0 or unfreeze_blocks > 4:
            raise ValueError("unfreeze_blocks 必须位于 [0, 4]")

        self.backbone = backbone
        self.feature_stages = feature_stages
        self.frozen = frozen
        self.unfreeze_blocks = unfreeze_blocks
        self.out_channels = channels
        self._frozen_modules: tuple[nn.Module, ...] = ()
        if frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        elif unfreeze_blocks:
            self.backbone.requires_grad_(False)
            split = 4 - unfreeze_blocks
            for module in (*stages[split:], *downsample_layers[split:]):
                module.requires_grad_(True)
            self._frozen_modules = tuple((*stages[:split], *downsample_layers[:split]))
        input_adapter = getattr(self.backbone, "input_adapter", None)
        if not frozen and isinstance(input_adapter, nn.Module):
            input_adapter.requires_grad_(True)

    def forward(self, inputs: Tensor) -> FeaturePyramid:
        if inputs.ndim != 4:
            raise ValueError(f"输入必须为 NCHW 张量, 实际形状为 {tuple(inputs.shape)}")
        if inputs.shape[-2] % 32 or inputs.shape[-1] % 32:
            raise ValueError("DINOv3 ConvNeXt 输入空间尺寸必须能被 32 整除")
        if self.frozen:
            with torch.no_grad():
                raw_features = self._extract(inputs)
        else:
            raw_features = self._extract(inputs)
        if len(raw_features) != 4:
            raise ValueError(f"主干必须返回四层中间特征, 实际返回 {len(raw_features)} 层")

        outputs: list[Tensor] = []
        for feature, channels, stride in zip(
            raw_features, self.out_channels, self.out_strides, strict=True
        ):
            expected_size = (inputs.shape[-2] // stride, inputs.shape[-1] // stride)
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError("DINOv3 ConvNeXt 中间特征必须为与 embed_dims 匹配的 NCHW 张量")
            if feature.shape[-2:] != expected_size:
                raise ValueError(
                    f"DINOv3 ConvNeXt 阶段特征尺寸应为 {expected_size}, "
                    f"实际为 {tuple(feature.shape[-2:])}"
                )
            outputs.append(feature)
        return tuple(outputs)  # type: ignore[return-value]

    def _extract(self, inputs: Tensor) -> tuple[Tensor, ...]:
        features = self.backbone.get_intermediate_layers(
            inputs,
            n=self.feature_stages,
            reshape=True,
            return_class_token=False,
        )
        return tuple(features)

    def train(self, mode: bool = True) -> DinoV3ConvNeXtAdapter:
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        else:
            for module in self._frozen_modules:
                module.eval()
        return self
