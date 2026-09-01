"""Class-conditioned fusion of CLIP and multimodal text correlation maps."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.models.correlation_aggregator import CorrelationAggregatorLayer
from hsi_lidar_ovseg.models.protocols import FeaturePyramid


class _CorrelationEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class TextCorrelationDecoder(nn.Module):
    """Decode two text-correlation pyramids with category-shared weights."""

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 64,
        clip_residual_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_dim != 512:
            raise ValueError("文本相关性解码器必须使用 512 维特征")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim 必须为正整数")
        if clip_residual_weight <= 0:
            raise ValueError("clip_residual_weight 必须为正数")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.clip_residual_weight = clip_residual_weight
        self.embeddings = nn.ModuleList(_CorrelationEmbedding(hidden_dim) for _ in range(4))
        self.aggregators = nn.ModuleList(
            CorrelationAggregatorLayer(hidden_dim, feature_dim, num_heads=4, window_size=7)
            for _ in range(2)
        )
        self.refine = nn.ModuleList(
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU())
            for _ in range(3)
        )
        self.head = nn.Conv2d(hidden_dim, 1, 1)

    @staticmethod
    def _validate_features(joint: FeaturePyramid, clip: FeaturePyramid) -> None:
        if len(joint) != 4 or len(clip) != 4:
            raise ValueError("相关性解码器必须接收四尺度特征")
        for joint_level, clip_level in zip(joint, clip, strict=True):
            if joint_level.shape != clip_level.shape:
                raise ValueError("联合模态和 CLIP 特征的形状必须一致")
            if joint_level.ndim != 4 or joint_level.shape[1] != 512:
                raise ValueError("相关性特征必须是 512 通道 NCHW 张量")

    @staticmethod
    def _correlation(features: Tensor, text: Tensor) -> Tensor:
        normalized_features = functional.normalize(features, dim=1)
        return torch.einsum("bchw,npc->bnphw", normalized_features, text).mean(dim=2)

    def forward(
        self,
        joint_features: FeaturePyramid,
        clip_features: FeaturePyramid,
        text_features: Tensor,
        output_size: tuple[int, int],
    ) -> Tensor:
        self._validate_features(joint_features, clip_features)
        if text_features.ndim != 3 or text_features.shape[-1] != self.feature_dim:
            raise ValueError("文本特征必须具有 [类别数,提示词数,512] 形状")
        if text_features.shape[0] <= 0 or text_features.shape[1] <= 0:
            raise ValueError("文本特征必须包含至少一个类别和一个提示词")
        text_features = functional.normalize(text_features, dim=-1)
        batch, classes = joint_features[0].shape[0], text_features.shape[0]
        levels: list[Tensor] = []
        raw_clip_correlations: list[Tensor] = []
        for joint, clip, embedding in zip(
            joint_features, clip_features, self.embeddings, strict=True
        ):
            height, width = joint.shape[-2:]
            joint_correlation = self._correlation(joint, text_features)
            clip_correlation = self._correlation(clip, text_features)
            correlations = (joint_correlation, clip_correlation)
            raw_clip_correlations.append(clip_correlation)
            pair = torch.stack(correlations, dim=2)
            cost = embedding(pair.reshape(batch * classes, 2, height, width))
            cost = cost.reshape(batch, classes, self.hidden_dim, height, width).permute(
                0, 2, 1, 3, 4
            )
            for aggregator in self.aggregators:
                cost = aggregator(cost, text_features)
            levels.append(
                cost.permute(0, 2, 1, 3, 4).reshape(batch * classes, self.hidden_dim, height, width)
            )

        decoded = levels[-1]
        for lateral, refine in zip(reversed(levels[:-1]), self.refine, strict=True):
            decoded = functional.interpolate(
                decoded, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = refine(decoded + lateral)
        logits = self.head(decoded).reshape(batch, classes, *decoded.shape[-2:])
        learned_logits = functional.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )
        residual = torch.stack(
            [
                functional.interpolate(
                    correlation,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for correlation in raw_clip_correlations
            ],
            dim=0,
        ).mean(dim=0)
        return learned_logits + self.clip_residual_weight * residual
