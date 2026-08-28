from __future__ import annotations

import pytest
import torch

from hsi_lidar_ovseg.metrics import SegmentationMetrics


def test_metrics_report_seen_unseen_harmonic_mean() -> None:
    metrics = SegmentationMetrics(3, seen_ids=(1, 2), unseen_ids=(3,))

    metrics.update(torch.tensor([1, 2, 3, 3]), torch.tensor([1, 2, 3, 1]))
    result = metrics.compute()

    assert result["seen_miou"] == pytest.approx(0.75)
    assert result["unseen_miou"] == pytest.approx(0.5)
    assert result["harmonic_miou"] == pytest.approx(0.6)
    assert result["overall_accuracy"] == pytest.approx(0.75)


def test_metrics_ignore_zero_and_absent_ground_truth_classes() -> None:
    metrics = SegmentationMetrics(3, seen_ids=(1, 2), unseen_ids=(3,))

    metrics.update(torch.tensor([0, 1, 1]), torch.tensor([3, 1, 2]))
    result = metrics.compute()

    assert result["miou"] == pytest.approx(0.5)
    assert result["per_class_iou"][2] is None


def test_metrics_reject_invalid_positive_predictions() -> None:
    metrics = SegmentationMetrics(3, seen_ids=(1, 2), unseen_ids=(3,))

    with pytest.raises(ValueError, match="预测"):
        metrics.update(torch.tensor([1]), torch.tensor([4]))
