from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from hsi_lidar_ovseg.models.correlation_decoder import TextCorrelationDecoder


def _pyramid() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, 512, 8, 8),
        torch.randn(1, 512, 4, 4),
        torch.randn(1, 512, 2, 2),
        torch.randn(1, 512, 1, 1),
    )


def test_correlation_decoder_supports_dynamic_runtime_class_counts() -> None:
    decoder = TextCorrelationDecoder(feature_dim=512, hidden_dim=32)
    parameters = sum(parameter.numel() for parameter in decoder.parameters())
    two = decoder(
        _pyramid(), _pyramid(), functional.normalize(torch.randn(2, 2, 512), dim=-1), (32, 32)
    )
    seven = decoder(
        _pyramid(), _pyramid(), functional.normalize(torch.randn(7, 2, 512), dim=-1), (32, 32)
    )

    assert two.shape == (1, 2, 32, 32)
    assert seven.shape == (1, 7, 32, 32)
    assert sum(parameter.numel() for parameter in decoder.parameters()) == parameters


def test_correlation_decoder_rejects_invalid_text_dimension() -> None:
    with pytest.raises(ValueError, match="512"):
        TextCorrelationDecoder(512, 32)(
            _pyramid(), _pyramid(), torch.randn(3, 2, 256), (32, 32)
        )
