from __future__ import annotations

import pytest
import torch

from hsi_lidar_ovseg.models.shared_lite_vit import SharedLiteViT


def test_shared_lite_vit_returns_four_token_stages_from_six_shared_blocks() -> None:
    model = SharedLiteViT(hsi_bands=6, lidar_channels=3)

    output = model(torch.randn(1, 6, 32, 32), torch.randn(1, 3, 32, 32))

    assert len(model.blocks) == 6
    assert model.spectral_adapter is not None
    assert model.lidar_patch_embed is not None
    assert output.grid_size == (2, 2)
    assert [item.shape for item in output.hsi_tokens] == [(1, 4, 384)] * 4
    assert [item.shape for item in output.lidar_tokens] == [(1, 4, 384)] * 4


def test_shared_lite_vit_rejects_spatially_unregistered_inputs() -> None:
    model = SharedLiteViT(hsi_bands=4, lidar_channels=1)

    with pytest.raises(ValueError, match="空间尺寸"):
        model(torch.randn(1, 4, 32, 32), torch.randn(1, 1, 16, 32))


def test_shared_lite_vit_rejects_non_patch_aligned_input() -> None:
    model = SharedLiteViT(hsi_bands=4, lidar_channels=1)

    with pytest.raises(ValueError, match="16"):
        model(torch.randn(1, 4, 30, 32), torch.randn(1, 1, 30, 32))
