"""Combined supervised and multimodal objective."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.config import LossConfig
from hsi_lidar_ovseg.losses.contrastive import LossError, symmetric_info_nce
from hsi_lidar_ovseg.models import SegmentationOutput
from hsi_lidar_ovseg.models.protocols import FeaturePyramid


class OpenVocabularyObjective(nn.Module):
    """Train seen-class segmentation while aligning all valid spatial features."""

    def __init__(
        self,
        config: LossConfig,
        seen_class_ids: tuple[int, ...],
        *,
        max_alignment_tokens: int = 256,
    ) -> None:
        super().__init__()
        if not seen_class_ids or any(class_id <= 0 for class_id in seen_class_ids):
            raise LossError("seen_class_ids 必须包含正类别编号")
        if max_alignment_tokens <= 0:
            raise LossError("max_alignment_tokens 必须为正整数")
        self.config = config
        self.seen_class_ids = seen_class_ids
        self.max_alignment_tokens = max_alignment_tokens

    def _selected_tokens(self, feature: Tensor, valid_mask: Tensor) -> Tensor:
        resized_mask = functional.interpolate(
            valid_mask[:, None].float(), size=feature.shape[-2:], mode="nearest"
        )[:, 0].bool()
        tokens = feature.permute(0, 2, 3, 1)[resized_mask]
        if tokens.shape[0] > self.max_alignment_tokens:
            step = math.ceil(tokens.shape[0] / self.max_alignment_tokens)
            tokens = tokens[::step][: self.max_alignment_tokens]
        return tokens

    def _alignment_loss(
        self, first: FeaturePyramid, second: FeaturePyramid, valid_mask: Tensor
    ) -> Tensor:
        losses: list[Tensor] = []
        for first_level, second_level in zip(first, second, strict=True):
            if second_level.shape[-2:] != first_level.shape[-2:]:
                second_level = functional.interpolate(
                    second_level,
                    size=first_level.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            first_tokens = self._selected_tokens(first_level, valid_mask)
            second_tokens = self._selected_tokens(second_level, valid_mask)
            if first_tokens.shape != second_tokens.shape:
                raise LossError("对齐特征的有效令牌形状不一致")
            if first_tokens.shape[0] > 0:
                losses.append(
                    symmetric_info_nce(
                        first_tokens,
                        second_tokens,
                        temperature=self.config.temperature,
                    )
                )
        if not losses:
            raise LossError("对齐掩码没有覆盖任何特征令牌")
        return torch.stack(losses).mean()

    def _private_loss(
        self,
        hsi: FeaturePyramid,
        lidar: FeaturePyramid,
        fused: FeaturePyramid,
        valid_mask: Tensor,
    ) -> Tensor:
        losses: list[Tensor] = []
        for hsi_level, lidar_level, fused_level in zip(hsi, lidar, fused, strict=True):
            private = self._selected_tokens(hsi_level - lidar_level, valid_mask)
            shared = self._selected_tokens(fused_level, valid_mask)
            if private.shape[0] > 0:
                private = functional.normalize(private, dim=-1)
                shared = functional.normalize(shared, dim=-1)
                losses.append((private * shared).sum(dim=-1).square().mean())
        if not losses:
            raise LossError("私有特征正则项没有有效令牌")
        return torch.stack(losses).mean()

    def forward(
        self, output: SegmentationOutput, labels: Tensor, valid_mask: Tensor
    ) -> dict[str, Tensor]:
        if labels.ndim != 3 or valid_mask.shape != labels.shape:
            raise LossError("labels 和 valid_mask 必须是形状相同的 NHW 张量")
        if valid_mask.dtype != torch.bool:
            raise LossError("valid_mask 必须是布尔张量")
        if (
            output.logits.shape[0] != labels.shape[0]
            or output.logits.shape[-2:] != labels.shape[-2:]
        ):
            raise LossError("分割分数与标签的批量或空间尺寸不一致")
        seen_ids = torch.tensor(self.seen_class_ids, device=labels.device, dtype=labels.dtype)
        supervised_mask = valid_mask & torch.isin(labels, seen_ids)
        if not supervised_mask.any():
            raise LossError("监督掩码中没有已见类像素")
        targets = labels[supervised_mask] - 1
        if targets.min() < 0 or targets.max() >= output.logits.shape[1]:
            raise LossError("监督标签超出模型类别分数范围")
        pixel_logits = output.logits.permute(0, 2, 3, 1)[supervised_mask]
        segmentation = functional.cross_entropy(pixel_logits, targets)

        features = output.alignment_features
        required = {
            "hsi",
            "lidar",
            "structure_teacher",
            "semantic_teacher",
            "fused",
        }
        if set(features) != required:
            raise LossError(f"alignment_features 必须包含 {sorted(required)}")
        hsi_structure = self._alignment_loss(
            features["hsi"], features["structure_teacher"], valid_mask
        )
        lidar_structure = self._alignment_loss(
            features["lidar"], features["structure_teacher"], valid_mask
        )
        fused_semantic = self._alignment_loss(
            features["fused"], features["semantic_teacher"], valid_mask
        )
        hsi_lidar = self._alignment_loss(features["hsi"], features["lidar"], valid_mask)
        gate = torch.stack([(gate_map.mean() - 0.5).square() for gate_map in output.gates]).mean()
        private = self._private_loss(
            features["hsi"], features["lidar"], features["fused"], valid_mask
        )
        total = (
            segmentation
            + self.config.structure_teacher_weight * (hsi_structure + lidar_structure)
            + self.config.semantic_teacher_weight * fused_semantic
            + self.config.cross_weight * hsi_lidar
            + self.config.gate_weight * gate
            + self.config.private_weight * private
        )
        losses = {
            "total": total,
            "segmentation": segmentation,
            "hsi_structure": hsi_structure,
            "lidar_structure": lidar_structure,
            "fused_semantic": fused_semantic,
            "hsi_lidar": hsi_lidar,
            "gate": gate,
            "private": private,
        }
        if not all(torch.isfinite(value) for value in losses.values()):
            raise LossError("损失包含非有限值")
        return losses
