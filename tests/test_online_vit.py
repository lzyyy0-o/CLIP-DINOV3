from __future__ import annotations

import torch

from hsi_lidar_ovseg.models.online_vit import OnlineViTPyramidEncoder


def test_hsi_online_vit_uses_spectral_adapter_and_returns_four_scales() -> None:
    encoder = OnlineViTPyramidEncoder(6, spectral_adapter=True)

    outputs = encoder(torch.randn(1, 6, 32, 32))

    assert encoder.spectral_adapter is not None
    assert [item.shape for item in outputs] == [
        (1, 384, 8, 8),
        (1, 384, 4, 4),
        (1, 384, 2, 2),
        (1, 384, 1, 1),
    ]


def test_lidar_online_vit_has_no_spectral_adapter() -> None:
    encoder = OnlineViTPyramidEncoder(1, spectral_adapter=False)

    outputs = encoder(torch.randn(1, 1, 32, 32))

    assert encoder.spectral_adapter is None
    assert len(outputs) == 4
