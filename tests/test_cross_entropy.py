from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hsi_lidar_ovseg.losses.cross_entropy import (
    ClipGuidedAlignmentObjective,
    MaskedCrossEntropyObjective,
)


def test_masked_cross_entropy_ignores_unseen_labels() -> None:
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    output = SimpleNamespace(logits=logits)

    losses = MaskedCrossEntropyObjective((1, 2))(
        output,
        torch.tensor([[[1, 3], [2, 3]]]),
        torch.ones(1, 2, 2, dtype=torch.bool),
    )

    assert set(losses) == {"total", "segmentation"}
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert logits.grad is not None


def test_masked_cross_entropy_rejects_missing_seen_pixels() -> None:
    objective = MaskedCrossEntropyObjective((1, 2))

    with pytest.raises(ValueError, match="已见类"):
        objective(
            SimpleNamespace(logits=torch.randn(1, 3, 2, 2)),
            torch.full((1, 2, 2), 3),
            torch.ones(1, 2, 2, dtype=torch.bool),
        )


def test_masked_cross_entropy_maps_non_contiguous_seen_ids_to_local_logits() -> None:
    logits = torch.tensor([[[[8.0, -8.0]], [[-8.0, 8.0]]]], requires_grad=True)
    losses = MaskedCrossEntropyObjective((2, 5))(
        SimpleNamespace(logits=logits),
        torch.tensor([[[2, 5]]]),
        torch.ones(1, 1, 2, dtype=torch.bool),
    )

    assert losses["total"] < 1e-5


def _alignment_output(joint: torch.Tensor, clip: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        logits=torch.randn(1, 2, 4, 4, requires_grad=True),
        joint_features=(joint, joint, joint, joint),
        clip_features=(clip, clip, clip, clip),
    )


def test_clip_alignment_detaches_identical_teacher_features() -> None:
    joint = torch.randn(1, 512, 4, 4, requires_grad=True)
    teacher = joint.detach().clone().requires_grad_()

    losses = ClipGuidedAlignmentObjective((1, 2), 0.1)(
        _alignment_output(joint, teacher),
        torch.tensor([[[1, 2, 1, 2]] * 4]),
        torch.ones(1, 4, 4, dtype=torch.bool),
    )
    losses["total"].backward()

    assert losses["clip_alignment"] < 1e-5
    assert joint.grad is not None
    assert teacher.grad is None


def test_clip_alignment_uses_only_valid_pixels() -> None:
    joint = torch.zeros(1, 512, 4, 4, requires_grad=True)
    teacher = torch.zeros(1, 512, 4, 4, requires_grad=True)
    with torch.no_grad():
        joint[:, 0] = 1.0
        teacher[:, 0] = 1.0
        teacher[:, 0, 0, 0] = -1.0
    valid_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    valid_mask[:, 0, 0] = False

    losses = ClipGuidedAlignmentObjective((1, 2), 0.1)(
        _alignment_output(joint, teacher),
        torch.ones(1, 4, 4, dtype=torch.long),
        valid_mask,
    )

    assert losses["clip_alignment"] < 1e-5


def test_clip_alignment_penalizes_orthogonal_feature_directions() -> None:
    joint = torch.zeros(1, 512, 4, 4, requires_grad=True)
    teacher = torch.zeros(1, 512, 4, 4, requires_grad=True)
    with torch.no_grad():
        joint[:, 0] = 1.0
        teacher[:, 1] = 1.0

    losses = ClipGuidedAlignmentObjective((1, 2), 0.1)(
        _alignment_output(joint, teacher),
        torch.ones(1, 4, 4, dtype=torch.long),
        torch.ones(1, 4, 4, dtype=torch.bool),
    )

    assert losses["clip_alignment"] > 0.9
