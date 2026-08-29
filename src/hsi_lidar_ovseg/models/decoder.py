"""FPN decoder that scores dense pixels against text prototypes."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import FeaturePyramid


def _group_count(channels: int) -> int:
    for groups in range(min(8, max(1, channels // 2)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DenseTextDecoder(nn.Module):
    """Decode a feature pyramid into normalized CLIP-space pixel embeddings."""

    def __init__(self, feature_dim: int, text_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0 or text_dim <= 0:
            raise ValueError("feature_dim 和 text_dim 必须为正整数")
        self.feature_dim = feature_dim
        self.text_dim = text_dim
        self.lateral = nn.ModuleList(nn.Conv2d(feature_dim, feature_dim, 1) for _ in range(4))
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(feature_dim), feature_dim),
            nn.GELU(),
        )
        self.output = nn.Conv2d(feature_dim, text_dim, 1)

    def forward(
        self,
        features: FeaturePyramid,
        text_embeddings: Tensor,
        output_size: tuple[int, int],
        logit_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if len(features) != 4:
            raise ValueError("解码器必须接收四层特征")
        if text_embeddings.ndim != 2 or text_embeddings.shape[1] != self.text_dim:
            raise ValueError(
                f"文本嵌入必须为 [类别数, {self.text_dim}], "
                f"实际形状为 {tuple(text_embeddings.shape)}"
            )
        decoded = self.lateral[-1](features[-1])
        for level in range(2, -1, -1):
            decoded = functional.interpolate(
                decoded, size=features[level].shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = decoded + self.lateral[level](features[level])
        decoded = self.output(self.refine(decoded))
        decoded = functional.interpolate(
            decoded, size=output_size, mode="bilinear", align_corners=False
        )
        pixel_embeddings = functional.normalize(decoded, dim=1)
        text_embeddings = functional.normalize(text_embeddings.float(), dim=-1)
        logits = torch.einsum("ndhw,cd->nchw", pixel_embeddings, text_embeddings)
        return logits * logit_scale, pixel_embeddings
