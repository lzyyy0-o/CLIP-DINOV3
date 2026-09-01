from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from hsi_lidar_ovseg.models.correlation_aggregator import CorrelationAggregatorLayer
from hsi_lidar_ovseg.models.correlation_decoder import TextCorrelationDecoder


def _pyramid() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, 512, 56, 56),
        torch.randn(1, 512, 28, 28),
        torch.randn(1, 512, 14, 14),
        torch.randn(1, 512, 7, 7),
    )


def test_correlation_decoder_supports_dynamic_runtime_class_counts() -> None:
    decoder = TextCorrelationDecoder(feature_dim=512, hidden_dim=64)
    parameters = sum(parameter.numel() for parameter in decoder.parameters())
    ten = decoder(
        _pyramid(), _pyramid(), functional.normalize(torch.randn(10, 2, 512), dim=-1), (224, 224)
    )
    fifteen = decoder(
        _pyramid(), _pyramid(), functional.normalize(torch.randn(15, 2, 512), dim=-1), (224, 224)
    )

    assert ten.shape == (1, 10, 224, 224)
    assert fifteen.shape == (1, 15, 224, 224)
    assert sum(parameter.numel() for parameter in decoder.parameters()) == parameters


def test_correlation_decoder_aggregates_14_by_14_cost_volumes() -> None:
    decoder = TextCorrelationDecoder(feature_dim=512, hidden_dim=64)
    joint = tuple(
        torch.randn(1, 512, size, size, requires_grad=True) for size in (56, 28, 14, 7)
    )
    clip = tuple(
        torch.randn(1, 512, size, size, requires_grad=True) for size in (56, 28, 14, 7)
    )

    logits = decoder(joint, clip, functional.normalize(torch.randn(10, 2, 512), dim=-1), (224, 224))
    logits.mean().backward()

    assert logits.shape == (1, 10, 224, 224)
    assert joint[2].grad is not None
    assert any(parameter.grad is not None for parameter in decoder.parameters())
    assert len(decoder.aggregators) == 2


def test_correlation_decoder_rejects_non_divisible_window_grid() -> None:
    decoder = TextCorrelationDecoder(feature_dim=512, hidden_dim=64)
    invalid = tuple(torch.randn(1, 512, size, size) for size in (56, 28, 14, 8))

    with pytest.raises(ValueError, match="7"):
        decoder(
            invalid,
            invalid,
            functional.normalize(torch.randn(10, 2, 512), dim=-1),
            (224, 224),
        )


def test_correlation_decoder_rejects_invalid_text_dimension() -> None:
    with pytest.raises(ValueError, match="512"):
        TextCorrelationDecoder(512, 32)(
            _pyramid(), _pyramid(), torch.randn(3, 2, 256), (32, 32)
        )


def test_correlation_aggregator_preserves_dynamic_cost_volume_shape() -> None:
    layer = CorrelationAggregatorLayer(hidden_dim=64, text_dim=512, num_heads=4, window_size=7)
    cost = torch.randn(1, 64, 15, 14, 14, requires_grad=True)
    text = functional.normalize(torch.randn(15, 2, 512), dim=-1)

    output = layer(cost, text)
    output.square().mean().backward()

    assert output.shape == cost.shape
    assert cost.grad is not None
