"""End-to-end CLIP-guided Shared Lite-ViT segmentor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import FeaturePyramid


@dataclass(frozen=True)
class ClipGuidedSegmentationOutput:
    """Dense dynamic-class logits emitted by the CLIP-guided model."""

    logits: Tensor
    joint_features: FeaturePyramid
    clip_features: FeaturePyramid


class CLIPGuidedSharedLiteViTSegmentor(nn.Module):
    """Fuse Shared Lite-ViT and CLIP features through text-conditioned correlations."""

    def __init__(
        self,
        shared_encoder: nn.Module,
        mmfb: nn.Module,
        cmfeb: nn.Module,
        joint_projector: nn.Module,
        clip_guidance: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.shared_encoder = shared_encoder
        self.mmfb = mmfb
        self.cmfeb = cmfeb
        self.joint_projector = joint_projector
        self.clip_guidance = clip_guidance
        self.decoder = decoder

    def forward(
        self, hsi: Tensor, lidar: Tensor, pseudo_rgb: Tensor, class_names: Sequence[str]
    ) -> ClipGuidedSegmentationOutput:
        if hsi.ndim != 4 or lidar.ndim != 4 or pseudo_rgb.ndim != 4:
            raise ValueError("HSI、LiDAR 和伪 RGB 输入必须为 NCHW 张量")
        if not (hsi.shape[0] == lidar.shape[0] == pseudo_rgb.shape[0]):
            raise ValueError("三种输入的批量大小必须一致")
        if not (hsi.shape[-2:] == lidar.shape[-2:] == pseudo_rgb.shape[-2:]):
            raise ValueError("三种输入的空间尺寸必须一致")
        if not class_names:
            raise ValueError("class_names 不得为空")
        token_output = self.shared_encoder(hsi, lidar)
        hsi_tokens, lidar_tokens = self.mmfb(token_output.hsi_tokens, token_output.lidar_tokens)
        joint_tokens = self.cmfeb(hsi_tokens, lidar_tokens)
        joint_features = self.joint_projector(joint_tokens, token_output.grid_size, hsi.shape[-2:])
        clip_features = self.clip_guidance.visual_features(pseudo_rgb)
        text_features = self.clip_guidance.text_features(tuple(class_names))
        logits = self.decoder(joint_features, clip_features, text_features, hsi.shape[-2:])
        return ClipGuidedSegmentationOutput(
            logits=logits,
            joint_features=joint_features,
            clip_features=clip_features,
        )
