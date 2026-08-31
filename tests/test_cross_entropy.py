from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hsi_lidar_ovseg.losses.cross_entropy import MaskedCrossEntropyObjective


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
