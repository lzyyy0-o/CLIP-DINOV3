"""Class vocabularies used to switch between training and evaluation label spaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ClassVocabulary:
    """Map a dynamic local logit axis to global one-based dataset class IDs."""

    class_ids: tuple[int, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.class_ids or len(self.class_ids) != len(self.class_names):
            raise ValueError("类别编号和类别名称必须为等长的非空序列")
        if len(set(self.class_ids)) != len(self.class_ids) or any(
            class_id <= 0 for class_id in self.class_ids
        ):
            raise ValueError("类别编号必须为不重复的正整数")
        if any(not class_name.strip() for class_name in self.class_names):
            raise ValueError("类别名称不得为空")

    @classmethod
    def from_all_class_names(
        cls, all_class_names: Sequence[str], class_ids: Sequence[int]
    ) -> ClassVocabulary:
        names = tuple(all_class_names)
        ids = tuple(int(class_id) for class_id in class_ids)
        if any(class_id > len(names) for class_id in ids):
            raise ValueError("类别编号超出类别名称词表范围")
        return cls(ids, tuple(names[class_id - 1] for class_id in ids))

    def decode_logits(self, logits: Tensor) -> Tensor:
        """Return global class IDs for logits indexed by this vocabulary's local axis."""

        if logits.ndim < 1 or logits.shape[0] != len(self.class_ids):
            raise ValueError("logits 的类别轴必须与当前词表长度一致")
        local_predictions = logits.argmax(dim=0)
        class_ids = torch.tensor(self.class_ids, device=logits.device, dtype=torch.int64)
        return class_ids[local_predictions]

    def prediction_counts(self, predictions: Tensor) -> dict[int, int]:
        """Count predictions for every class in the vocabulary, including absent classes."""

        if predictions.dtype not in (torch.int32, torch.int64):
            raise ValueError("预测类别必须为整数张量")
        return {
            class_id: int((predictions == class_id).sum().item()) for class_id in self.class_ids
        }
