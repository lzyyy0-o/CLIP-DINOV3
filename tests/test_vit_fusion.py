from __future__ import annotations

import pytest
import torch

from hsi_lidar_ovseg.models.vit_fusion import TokenPyramidProjector, ViTCMFEB, ViTMMFB


def test_vit_mmfb_cmfeb_and_projector_preserve_four_stages() -> None:
    hsi = tuple(torch.randn(1, 4, 384) for _ in range(4))
    lidar = tuple(torch.randn(1, 4, 384) for _ in range(4))
    hsi_updated, lidar_updated = ViTMMFB(embed_dim=384, num_heads=6)(hsi, lidar)
    joint = ViTCMFEB(embed_dim=384, num_heads=6)(hsi_updated, lidar_updated)
    maps = TokenPyramidProjector(384, 512)(joint, grid_size=(2, 2), image_size=(32, 32))

    assert len(joint) == len(maps) == 4
    assert [item.shape for item in maps] == [
        (1, 512, 8, 8),
        (1, 512, 4, 4),
        (1, 512, 2, 2),
        (1, 512, 1, 1),
    ]
    assert not torch.equal(hsi_updated[0], hsi[0])


def test_token_pyramid_projector_rejects_invalid_token_count() -> None:
    projector = TokenPyramidProjector(384, 512)

    with pytest.raises(ValueError, match="token"):
        projector(tuple(torch.randn(1, 3, 384) for _ in range(4)), (2, 2), (32, 32))
