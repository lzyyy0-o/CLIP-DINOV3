"""Streaming metrics for seen and unseen semantic classes."""

from __future__ import annotations

import torch
from torch import Tensor

MetricValue = float | list[float | None]


class SegmentationMetrics:
    """Accumulate a confusion matrix using one-based semantic class IDs."""

    def __init__(
        self,
        num_classes: int,
        seen_ids: tuple[int, ...],
        unseen_ids: tuple[int, ...],
        ignore_index: int = 0,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes 必须为正整数")
        expected = set(range(1, num_classes + 1))
        if set(seen_ids) & set(unseen_ids):
            raise ValueError("seen_ids 与 unseen_ids 不得重叠")
        if set(seen_ids) | set(unseen_ids) != expected:
            raise ValueError("seen_ids 与 unseen_ids 必须覆盖全部正类别")
        self.num_classes = num_classes
        self.seen_ids = seen_ids
        self.unseen_ids = unseen_ids
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self) -> None:
        """Clear all accumulated pixels."""

        self.confusion.zero_()

    def update(self, ground_truth: Tensor, predictions: Tensor) -> None:
        """Accumulate one prediction tensor using one-based class IDs."""

        if ground_truth.shape != predictions.shape:
            raise ValueError("真实标签与预测的形状必须一致")
        ground_truth = ground_truth.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        predictions = predictions.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        valid = ground_truth != self.ignore_index
        ground_truth = ground_truth[valid]
        predictions = predictions[valid]
        if ground_truth.numel() == 0:
            return
        if torch.any((ground_truth < 1) | (ground_truth > self.num_classes)):
            raise ValueError("真实标签包含配置范围外的类别编号")
        if torch.any((predictions < 1) | (predictions > self.num_classes)):
            raise ValueError("预测包含配置范围外的类别编号")
        encoded = (ground_truth - 1) * self.num_classes + (predictions - 1)
        self.confusion += torch.bincount(encoded, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    @staticmethod
    def _mean_present(values: Tensor, present: Tensor) -> float:
        if not present.any():
            return 0.0
        return float(values[present].mean().item())

    def _subset_mean(self, values: Tensor, present: Tensor, class_ids: tuple[int, ...]) -> float:
        indices = torch.tensor([class_id - 1 for class_id in class_ids], dtype=torch.long)
        return self._mean_present(values[indices], present[indices])

    def compute(self) -> dict[str, MetricValue]:
        """Return aggregate and per-class metrics from the confusion matrix."""

        confusion = self.confusion.to(torch.float64)
        true_pixels = confusion.sum(dim=1)
        predicted_pixels = confusion.sum(dim=0)
        intersection = confusion.diag()
        union = true_pixels + predicted_pixels - intersection
        present = true_pixels > 0
        iou = torch.zeros_like(intersection)
        accuracy = torch.zeros_like(intersection)
        iou[union > 0] = intersection[union > 0] / union[union > 0]
        accuracy[present] = intersection[present] / true_pixels[present]
        seen_miou = self._subset_mean(iou, present, self.seen_ids)
        unseen_miou = self._subset_mean(iou, present, self.unseen_ids)
        denominator = seen_miou + unseen_miou
        harmonic = 0.0 if denominator == 0 else 2.0 * seen_miou * unseen_miou / denominator
        total = true_pixels.sum()
        overall_accuracy = 0.0 if total == 0 else float(intersection.sum().item() / total.item())

        per_class_iou = [
            float(iou[index]) if present[index] else None for index in range(self.num_classes)
        ]
        per_class_accuracy = [
            float(accuracy[index]) if present[index] else None for index in range(self.num_classes)
        ]
        return {
            "miou": self._mean_present(iou, present),
            "mean_class_accuracy": self._mean_present(accuracy, present),
            "seen_miou": seen_miou,
            "unseen_miou": unseen_miou,
            "harmonic_miou": harmonic,
            "overall_accuracy": overall_accuracy,
            "per_class_iou": per_class_iou,
            "per_class_accuracy": per_class_accuracy,
        }
