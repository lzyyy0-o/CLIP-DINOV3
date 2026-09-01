"""Masked multiclass cross-entropy for the CLIP-guided architecture."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class _LogitOutput(Protocol):
    logits: Tensor


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
