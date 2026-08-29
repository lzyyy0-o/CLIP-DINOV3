"""Four-scale visual adapter for a locally loaded RemoteCLIP model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import FeaturePyramid


def _visual_width(backbone: nn.Module) -> int:
    conv1 = getattr(backbone, "conv1", None)
    width = getattr(conv1, "out_channels", None)
    if isinstance(width, int) and width > 0:
        return width
    ln_pre = getattr(backbone, "ln_pre", None)
    normalized_shape = getattr(ln_pre, "normalized_shape", None)
    if isinstance(normalized_shape, Sequence) and normalized_shape:
        width = int(normalized_shape[-1])
        if width > 0:
            return width
    raise ValueError("RemoteCLIP 视觉塔必须通过 conv1 或 ln_pre 公开特征宽度")


class RemoteClipVisionAdapter(nn.Module):
    """Convert OpenCLIP visual intermediates into a four-level teacher pyramid."""

    out_strides = (4, 8, 16, 32)

    def __init__(
        self,
        backbone: nn.Module,
        feature_blocks: tuple[int, int, int, int],
        feature_dim: int,
        *,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        if len(feature_blocks) != 4:
            raise ValueError("feature_blocks 必须包含四个层编号")
        if feature_dim <= 0:
            raise ValueError("feature_dim 必须为正整数")
        if not callable(getattr(backbone, "forward_intermediates", None)):
            raise ValueError("RemoteCLIP 视觉塔必须实现 forward_intermediates")
        self.backbone = backbone
        self.feature_blocks = feature_blocks
        self.feature_dim = feature_dim
        self.frozen = frozen
        self.out_channels = (feature_dim,) * 4
        width = _visual_width(backbone)
        self.projections = nn.ModuleList(nn.Conv2d(width, feature_dim, 1) for _ in range(4))
        if frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def _extract(self, inputs: Tensor) -> tuple[Tensor, ...]:
        result = self.backbone.forward_intermediates(
            inputs,
            indices=self.feature_blocks,
            stop_early=True,
            normalize_intermediates=True,
            intermediates_only=True,
            output_fmt="NCHW",
        )
        if not isinstance(result, Mapping):
            raise ValueError("RemoteCLIP forward_intermediates 必须返回映射")
        features = result.get("image_intermediates")
        if not isinstance(features, (list, tuple)):
            raise ValueError("RemoteCLIP 返回值必须包含 image_intermediates 序列")
        return tuple(features)

    def forward(self, inputs: Tensor) -> FeaturePyramid:
        if inputs.ndim != 4:
            raise ValueError(f"输入必须为 NCHW 张量, 实际形状为 {tuple(inputs.shape)}")
        if inputs.shape[-2] % 32 or inputs.shape[-1] % 32:
            raise ValueError("RemoteCLIP 输入空间尺寸必须能被 32 整除")
        if self.frozen:
            with torch.no_grad():
                raw_features = self._extract(inputs)
        else:
            raw_features = self._extract(inputs)
        if len(raw_features) != 4:
            raise ValueError(f"视觉塔必须返回四层中间特征, 实际返回 {len(raw_features)} 层")

        outputs: list[Tensor] = []
        for feature, projection, stride in zip(
            raw_features, self.projections, self.out_strides, strict=True
        ):
            if feature.ndim != 4:
                raise ValueError("RemoteCLIP 中间特征必须为 NCHW 张量")
            projected = projection(feature)
            target_size = (inputs.shape[-2] // stride, inputs.shape[-1] // stride)
            outputs.append(
                functional.interpolate(
                    projected,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return tuple(outputs)  # type: ignore[return-value]

    def train(self, mode: bool = True) -> RemoteClipVisionAdapter:
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self
