from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hsi_lidar_ovseg.models.factories import ExternalSourceError, add_source_path
from hsi_lidar_ovseg.models.input_adapter import ChannelAdapter


def test_channel_adapter_changes_only_channels() -> None:
    adapter = ChannelAdapter(input_channels=1, output_channels=3)

    result = adapter(torch.randn(2, 1, 32, 32))

    assert result.shape == (2, 3, 32, 32)
    assert all(parameter.requires_grad for parameter in adapter.parameters())


def test_add_source_path_requires_expected_official_layout(tmp_path: Path) -> None:
    with pytest.raises(ExternalSourceError, match="dinov3"):
        add_source_path(tmp_path, "dinov3")
