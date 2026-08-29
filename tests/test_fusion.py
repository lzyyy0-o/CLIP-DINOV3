from __future__ import annotations

import pytest
import torch

from hsi_lidar_ovseg.models.fusion import GatedPyramidFusion


def test_gated_fusion_returns_bounded_gates_and_gradients() -> None:
    hsi = tuple(
        torch.randn(2, 8, 8 // 2**level, 8 // 2**level, requires_grad=True) for level in range(4)
    )
    lidar = tuple(torch.randn_like(feature, requires_grad=True) for feature in hsi)

    fused, gates = GatedPyramidFusion((8,) * 4, (8,) * 4, 16)(hsi, lidar)

    assert all(torch.all((gate >= 0) & (gate <= 1)) for gate in gates)
    assert all(feature.shape[1] == 16 for feature in fused)
    sum(feature.mean() for feature in fused).backward()
    assert hsi[0].grad is not None
    assert lidar[0].grad is not None


def test_gated_fusion_rejects_mismatched_spatial_shapes() -> None:
    hsi = tuple(torch.randn(1, 8, 8 // 2**level, 8 // 2**level) for level in range(4))
    lidar = list(torch.randn_like(feature) for feature in hsi)
    lidar[2] = torch.randn(1, 8, 3, 3)

    with pytest.raises(ValueError, match="空间尺寸"):
        GatedPyramidFusion((8,) * 4, (8,) * 4, 16)(hsi, tuple(lidar))
