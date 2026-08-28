"""Contrastive objectives for registered multimodal features."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


class LossError(ValueError):
    """Raised when a loss cannot be computed from the supplied batch."""


def symmetric_info_nce(
    first: Tensor,
    second: Tensor,
    valid_mask: Tensor | None = None,
    temperature: float = 0.1,
) -> Tensor:
    """Compute bidirectional InfoNCE for row-aligned feature pairs."""

    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise LossError(
            "InfoNCE 输入必须是形状相同的二维张量, "
            f"实际为 {tuple(first.shape)} 和 {tuple(second.shape)}"
        )
    if temperature <= 0:
        raise LossError("temperature 必须为正数")
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.ndim != 1:
            raise LossError("valid_mask 必须是一维布尔张量")
        if valid_mask.shape[0] != first.shape[0]:
            raise LossError("valid_mask 长度必须与特征行数一致")
        first = first[valid_mask]
        second = second[valid_mask]
    if first.shape[0] == 0:
        raise LossError("InfoNCE 有效特征对不能为空")

    first = functional.normalize(first.float(), dim=-1)
    second = functional.normalize(second.float(), dim=-1)
    logits = first @ second.transpose(0, 1) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    first_to_second = functional.cross_entropy(logits, targets)
    second_to_first = functional.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (first_to_second + second_to_first)
