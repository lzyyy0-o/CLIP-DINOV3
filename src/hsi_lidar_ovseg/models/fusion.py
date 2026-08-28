"""Multi-resolution gated fusion for registered HSI and LiDAR features."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import FeaturePyramid


def _group_count(channels: int) -> int:
    for groups in range(min(8, max(1, channels // 2)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _projection(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
    )


class GatedPyramidFusion(nn.Module):
    """Project and spatially gate paired feature pyramids at every scale."""

    def __init__(
        self,
        hsi_channels: tuple[int, int, int, int],
        lidar_channels: tuple[int, int, int, int],
        feature_dim: int,
    ) -> None:
        super().__init__()
        if len(hsi_channels) != 4 or len(lidar_channels) != 4:
            raise ValueError("HSI 和 LiDAR 通道配置都必须包含四层")
        if feature_dim <= 0:
            raise ValueError("feature_dim 必须为正整数")
        self.feature_dim = feature_dim
        self.hsi_projections = nn.ModuleList(
            _projection(channels, feature_dim) for channels in hsi_channels
        )
        self.lidar_projections = nn.ModuleList(
            _projection(channels, feature_dim) for channels in lidar_channels
        )
        self.gates = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(feature_dim, 1, 1),
                nn.Sigmoid(),
            )
            for _ in range(4)
        )

    @staticmethod
    def _validate_pyramids(
        hsi_features: tuple[Tensor, ...], lidar_features: tuple[Tensor, ...]
    ) -> None:
        if len(hsi_features) != 4 or len(lidar_features) != 4:
            raise ValueError("HSI 和 LiDAR 编码器都必须返回四层特征")
        for level, (hsi, lidar) in enumerate(zip(hsi_features, lidar_features, strict=True)):
            if hsi.ndim != 4 or lidar.ndim != 4:
                raise ValueError(f"第 {level} 层特征必须为 NCHW 张量")
            if hsi.shape[0] != lidar.shape[0]:
                raise ValueError(f"第 {level} 层批量大小不一致")
            if hsi.shape[-2:] != lidar.shape[-2:]:
                raise ValueError(f"第 {level} 层空间尺寸不一致")

    def project(
        self, hsi_features: tuple[Tensor, ...], lidar_features: tuple[Tensor, ...]
    ) -> tuple[FeaturePyramid, FeaturePyramid]:
        """Project both modality pyramids to the shared feature dimension."""

        self._validate_pyramids(hsi_features, lidar_features)
        projected_hsi = tuple(
            projection(feature)
            for projection, feature in zip(self.hsi_projections, hsi_features, strict=True)
        )
        projected_lidar = tuple(
            projection(feature)
            for projection, feature in zip(self.lidar_projections, lidar_features, strict=True)
        )
        return projected_hsi, projected_lidar  # type: ignore[return-value]

    def fuse_projected(
        self, hsi_features: FeaturePyramid, lidar_features: FeaturePyramid
    ) -> tuple[FeaturePyramid, FeaturePyramid]:
        """Fuse already projected features and return spatial gate maps."""

        fused: list[Tensor] = []
        gate_maps: list[Tensor] = []
        for hsi, lidar, gate_layer in zip(hsi_features, lidar_features, self.gates, strict=True):
            gate = gate_layer(torch.cat((hsi, lidar), dim=1))
            fused.append(gate * hsi + (1.0 - gate) * lidar)
            gate_maps.append(gate)
        return tuple(fused), tuple(gate_maps)  # type: ignore[return-value]

    def forward(
        self, hsi_features: tuple[Tensor, ...], lidar_features: tuple[Tensor, ...]
    ) -> tuple[FeaturePyramid, FeaturePyramid]:
        projected_hsi, projected_lidar = self.project(hsi_features, lidar_features)
        return self.fuse_projected(projected_hsi, projected_lidar)
