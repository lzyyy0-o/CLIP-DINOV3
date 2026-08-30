from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from hsi_lidar_ovseg.config import ConfigError
from hsi_lidar_ovseg.models.hypersigma_bridge import HyperSigmaBridge, load_hypersigma_weights


class _FakeSpatialEncoder(nn.Module):
    patch_size = 4
    embed_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(12))

    def forward_features(self, inputs: Tensor, _: int) -> list[Tensor]:
        batch_size = inputs.shape[0]
        return [inputs] + [torch.ones(batch_size, 8, 8, 8) for _ in range(4)]


class _FakeSpectralEncoder(nn.Module):
    NUM_TOKENS = 4

    def forward(self, inputs: Tensor) -> list[Tensor]:
        return [torch.ones(inputs.shape[0], 4, 3)]


def test_hypersigma_bridge_adapts_bands_and_modulates_four_scales() -> None:
    bridge = HyperSigmaBridge(
        spatial_encoder=_FakeSpatialEncoder(),
        spectral_encoder=_FakeSpectralEncoder(),
        input_channels=6,
        pretrained_in_channels=4,
    )

    outputs = bridge.forward_intermediates(torch.randn(1, 6, 32, 32), indices=(3, 5, 7, 11))

    assert [output.shape for output in outputs] == [(1, 8, 8, 8)] * 4
    assert bridge.input_adapter.projection.in_channels == 6
    assert bridge.input_adapter.projection.out_channels == 4


def test_hypersigma_loader_names_the_failing_branch(tmp_path: Path) -> None:
    bridge = HyperSigmaBridge(
        spatial_encoder=_FakeSpatialEncoder(),
        spectral_encoder=_FakeSpectralEncoder(),
        input_channels=6,
        pretrained_in_channels=4,
    )

    with pytest.raises(ConfigError, match="空间分支"):
        load_hypersigma_weights(bridge, tmp_path / "bad.pt", tmp_path / "spectral.pt")
