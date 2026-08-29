"""End-to-end HSI-LiDAR open-vocabulary segmentor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hsi_lidar_ovseg.models.decoder import DenseTextDecoder
from hsi_lidar_ovseg.models.fusion import GatedPyramidFusion
from hsi_lidar_ovseg.models.native import NativePyramidEncoder
from hsi_lidar_ovseg.models.protocols import FeaturePyramid


@dataclass
class SegmentationOutput:
    """Dense predictions and intermediate tensors needed by the objective."""

    logits: Tensor
    pixel_embeddings: Tensor
    alignment_features: dict[str, FeaturePyramid]
    gates: FeaturePyramid


class HSILidarOVSegmentor(nn.Module):
    """Fuse paired HSI and LiDAR features under frozen structure and semantic teachers."""

    def __init__(
        self,
        hsi_encoder: nn.Module,
        lidar_encoder: nn.Module,
        structure_teacher_encoder: nn.Module,
        semantic_teacher_encoder: nn.Module,
        feature_dim: int,
        text_dim: int,
        *,
        freeze_teachers: bool = True,
    ) -> None:
        super().__init__()
        self.hsi_encoder = hsi_encoder
        self.lidar_encoder = lidar_encoder
        self.structure_teacher_encoder = structure_teacher_encoder
        self.semantic_teacher_encoder = semantic_teacher_encoder
        self.freeze_teachers = freeze_teachers
        hsi_channels = self._channels(hsi_encoder, "hsi_encoder")
        lidar_channels = self._channels(lidar_encoder, "lidar_encoder")
        structure_teacher_channels = self._channels(
            structure_teacher_encoder, "structure_teacher_encoder"
        )
        semantic_teacher_channels = self._channels(
            semantic_teacher_encoder, "semantic_teacher_encoder"
        )
        self.fusion = GatedPyramidFusion(hsi_channels, lidar_channels, feature_dim)
        self.structure_teacher_projections = nn.ModuleList(
            nn.Conv2d(channels, feature_dim, 1) for channels in structure_teacher_channels
        )
        self.semantic_teacher_projections = nn.ModuleList(
            nn.Conv2d(channels, feature_dim, 1) for channels in semantic_teacher_channels
        )
        self.decoder = DenseTextDecoder(feature_dim, text_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        if freeze_teachers:
            self.structure_teacher_encoder.requires_grad_(False)
            self.semantic_teacher_encoder.requires_grad_(False)
            self.structure_teacher_encoder.eval()
            self.semantic_teacher_encoder.eval()

    @staticmethod
    def _channels(encoder: nn.Module, name: str) -> tuple[int, int, int, int]:
        channels = getattr(encoder, "out_channels", None)
        if not isinstance(channels, tuple) or len(channels) != 4:
            raise ValueError(f"{name} 必须公开四层 out_channels")
        return channels

    def _teacher_features(
        self,
        encoder: nn.Module,
        projections: nn.ModuleList,
        inputs: Tensor,
    ) -> FeaturePyramid:
        if self.freeze_teachers:
            with torch.no_grad():
                features = encoder(inputs)
        else:
            features = encoder(inputs)
        projected = tuple(
            projection(feature) for projection, feature in zip(projections, features, strict=True)
        )
        return projected  # type: ignore[return-value]

    def forward(
        self,
        hsi: Tensor,
        lidar: Tensor,
        pseudo_rgb: Tensor,
        text_embeddings: Tensor,
    ) -> SegmentationOutput:
        if hsi.ndim != 4 or lidar.ndim != 4 or pseudo_rgb.ndim != 4:
            raise ValueError("HSI、LiDAR 和伪 RGB 输入必须为 NCHW 张量")
        if not (hsi.shape[0] == lidar.shape[0] == pseudo_rgb.shape[0]):
            raise ValueError("三种输入的批量大小必须一致")
        if not (hsi.shape[-2:] == lidar.shape[-2:] == pseudo_rgb.shape[-2:]):
            raise ValueError("三种输入的空间尺寸必须一致")

        hsi_raw = self.hsi_encoder(hsi)
        lidar_raw = self.lidar_encoder(lidar)
        hsi_features, lidar_features = self.fusion.project(hsi_raw, lidar_raw)
        fused, gates = self.fusion.fuse_projected(hsi_features, lidar_features)
        structure_teacher_features = self._teacher_features(
            self.structure_teacher_encoder,
            self.structure_teacher_projections,
            pseudo_rgb,
        )
        semantic_teacher_features = self._teacher_features(
            self.semantic_teacher_encoder,
            self.semantic_teacher_projections,
            pseudo_rgb,
        )
        scale = self.logit_scale.clamp(min=0.0, max=math.log(100.0)).exp()
        logits, pixel_embeddings = self.decoder(fused, text_embeddings, hsi.shape[-2:], scale)
        return SegmentationOutput(
            logits=logits,
            pixel_embeddings=pixel_embeddings,
            alignment_features={
                "hsi": hsi_features,
                "lidar": lidar_features,
                "structure_teacher": structure_teacher_features,
                "semantic_teacher": semantic_teacher_features,
                "fused": fused,
            },
            gates=gates,
        )

    def train(self, mode: bool = True) -> HSILidarOVSegmentor:
        super().train(mode)
        if self.freeze_teachers:
            self.structure_teacher_encoder.eval()
            self.semantic_teacher_encoder.eval()
        return self


def make_native_model(
    hsi_bands: int,
    lidar_channels: int,
    feature_dim: int,
    text_dim: int,
    *,
    encoder_channels: tuple[int, int, int, int] = (16, 24, 32, 48),
) -> HSILidarOVSegmentor:
    """Build a fully offline model for smoke tests and native baselines."""

    return HSILidarOVSegmentor(
        hsi_encoder=NativePyramidEncoder(hsi_bands, encoder_channels),
        lidar_encoder=NativePyramidEncoder(lidar_channels, encoder_channels),
        structure_teacher_encoder=NativePyramidEncoder(3, encoder_channels),
        semantic_teacher_encoder=NativePyramidEncoder(3, encoder_channels),
        feature_dim=feature_dim,
        text_dim=text_dim,
        freeze_teachers=True,
    )
