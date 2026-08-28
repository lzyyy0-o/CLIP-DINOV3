from __future__ import annotations

import pytest
import torch

from hsi_lidar_ovseg.config import LossConfig
from hsi_lidar_ovseg.losses import LossError, OpenVocabularyObjective, symmetric_info_nce
from hsi_lidar_ovseg.models import SegmentationOutput


def _output_fixture() -> SegmentationOutput:
    levels = (
        torch.randn(1, 8, 4, 4, requires_grad=True),
        torch.randn(1, 8, 2, 2, requires_grad=True),
        torch.randn(1, 8, 1, 1, requires_grad=True),
        torch.randn(1, 8, 1, 1, requires_grad=True),
    )
    return SegmentationOutput(
        logits=torch.randn(1, 3, 8, 8, requires_grad=True),
        pixel_embeddings=torch.randn(1, 6, 8, 8, requires_grad=True),
        alignment_features={
            "hsi": levels,
            "lidar": tuple(feature + 0.1 for feature in levels),
            "teacher": tuple(feature.detach() + 0.05 for feature in levels),
            "fused": tuple(feature + 0.02 for feature in levels),
        },
        gates=tuple(torch.sigmoid(feature[:, :1]) for feature in levels),
    )


def test_info_nce_prefers_matching_pairs() -> None:
    aligned = torch.nn.functional.normalize(torch.eye(4), dim=-1)
    shuffled = aligned[[1, 0, 3, 2]]

    assert symmetric_info_nce(aligned, aligned) < symmetric_info_nce(aligned, shuffled)


def test_info_nce_applies_valid_mask() -> None:
    first = torch.eye(4)
    second = first.clone()

    masked = symmetric_info_nce(first, second, valid_mask=torch.tensor([True, False, True, False]))

    assert torch.isfinite(masked)


def test_objective_rejects_empty_supervised_mask() -> None:
    objective = OpenVocabularyObjective(LossConfig(), seen_class_ids=(1, 2))

    with pytest.raises(LossError, match="监督掩码"):
        objective(
            _output_fixture(),
            torch.zeros(1, 8, 8, dtype=torch.long),
            torch.ones(1, 8, 8, dtype=torch.bool),
        )


def test_objective_returns_finite_components_and_gradients() -> None:
    objective = OpenVocabularyObjective(LossConfig(), seen_class_ids=(1, 2))
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[:, :4] = 1
    labels[:, 4:] = 3

    losses = objective(_output_fixture(), labels, torch.ones_like(labels, dtype=torch.bool))

    assert set(losses) == {
        "total",
        "segmentation",
        "hsi_teacher",
        "lidar_teacher",
        "hsi_lidar",
        "gate",
        "private",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
