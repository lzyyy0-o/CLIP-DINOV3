"""Masked multiclass cross-entropy for the CLIP-guided architecture."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import FeaturePyramid


class _LogitOutput(Protocol):
    logits: Tensor


class _ClipGuidedOutput(_LogitOutput, Protocol):
    joint_features: FeaturePyramid
    clip_features: FeaturePyramid


class MaskedCrossEntropyObjective(nn.Module):
    """Supervise only configured seen class pixels with one segmentation loss."""

    def __init__(self, seen_class_ids: tuple[int, ...]) -> None:
        super().__init__()
        if not seen_class_ids or any(class_id <= 0 for class_id in seen_class_ids):
            raise ValueError("seen_class_ids 必须包含正类别编号")
        self.seen_class_ids = seen_class_ids

    def forward(
        self, output: _LogitOutput, labels: Tensor, valid_mask: Tensor
    ) -> dict[str, Tensor]:
        if labels.ndim != 3 or valid_mask.shape != labels.shape:
            raise ValueError("labels 和 valid_mask 必须是形状相同的 NHW 张量")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask 必须是布尔张量")
        logits = output.logits
        if (
            logits.ndim != 4
            or logits.shape[0] != labels.shape[0]
            or logits.shape[-2:] != labels.shape[-2:]
        ):
            raise ValueError("分割 logits 与标签的批量或空间尺寸不一致")
        local_targets = torch.full_like(labels, -1)
        for local_index, class_id in enumerate(self.seen_class_ids):
            local_targets[labels == class_id] = local_index
        supervised_mask = valid_mask & (local_targets >= 0)
        if not supervised_mask.any():
            raise ValueError("监督掩码中没有已见类像素")
        targets = local_targets[supervised_mask]
        if targets.min() < 0 or targets.max() >= logits.shape[1]:
            raise ValueError("监督标签超出模型类别分数范围")
        pixel_logits = logits.permute(0, 2, 3, 1)[supervised_mask]
        segmentation = functional.cross_entropy(pixel_logits, targets)
        return {"total": segmentation, "segmentation": segmentation}


class ClipGuidedAlignmentObjective(MaskedCrossEntropyObjective):
    """Align every joint feature level to its detached CLIP counterpart."""

    def __init__(self, seen_class_ids: tuple[int, ...], clip_alignment_weight: float) -> None:
        super().__init__(seen_class_ids)
        if clip_alignment_weight <= 0:
            raise ValueError("clip_alignment_weight 必须为正数")
        self.clip_alignment_weight = clip_alignment_weight

    @staticmethod
    def _validate_pyramids(
        joint_features: FeaturePyramid, clip_features: FeaturePyramid, batch_size: int
    ) -> None:
        if len(joint_features) != 4 or len(clip_features) != 4:
            raise ValueError("CLIP 对齐必须接收四尺度特征")
        for joint, clip in zip(joint_features, clip_features, strict=True):
            if joint.ndim != 4 or joint.shape[1] != 512:
                raise ValueError("联合特征必须是 512 通道 NCHW 张量")
            if joint.shape != clip.shape:
                raise ValueError("联合特征和 CLIP 特征的形状必须一致")
            if joint.shape[0] != batch_size:
                raise ValueError("对齐特征与标签的批量大小不一致")

    def forward(
        self, output: _ClipGuidedOutput, labels: Tensor, valid_mask: Tensor
    ) -> dict[str, Tensor]:
        losses = super().forward(output, labels, valid_mask)
        self._validate_pyramids(output.joint_features, output.clip_features, labels.shape[0])
        alignment_terms: list[Tensor] = []
        for joint, clip in zip(output.joint_features, output.clip_features, strict=True):
            mask = functional.interpolate(
                valid_mask[:, None].float(), size=joint.shape[-2:], mode="nearest"
            ).squeeze(1).bool()
            if not mask.any():
                raise ValueError("对齐掩码中没有有效像素")
            student = functional.normalize(joint, dim=1)
            teacher = functional.normalize(clip.detach(), dim=1)
            cosine_distance = 1 - (student * teacher).sum(dim=1)
            alignment_terms.append(cosine_distance[mask].mean())
        alignment = torch.stack(alignment_terms).mean()
        return {
            "total": losses["segmentation"] + self.clip_alignment_weight * alignment,
            "segmentation": losses["segmentation"],
            "clip_alignment": alignment,
        }
