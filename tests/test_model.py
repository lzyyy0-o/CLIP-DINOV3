from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from hsi_lidar_ovseg.models import make_native_model


def test_model_outputs_dense_normalized_embeddings() -> None:
    model = make_native_model(hsi_bands=12, lidar_channels=3, feature_dim=16, text_dim=24)
    text = functional.normalize(torch.randn(5, 24), dim=-1)

    output = model(
        torch.randn(2, 12, 64, 64),
        torch.randn(2, 3, 64, 64),
        torch.randn(2, 3, 64, 64),
        text,
    )

    assert output.logits.shape == (2, 5, 64, 64)
    assert output.pixel_embeddings.shape == (2, 24, 64, 64)
    torch.testing.assert_close(
        output.pixel_embeddings.norm(dim=1),
        torch.ones(2, 64, 64),
        atol=1e-5,
        rtol=1e-5,
    )
    assert set(output.alignment_features) == {"hsi", "lidar", "teacher", "fused"}
    assert len(output.gates) == 4


def test_model_clamps_similarity_scale() -> None:
    model = make_native_model(hsi_bands=4, lidar_channels=3, feature_dim=8, text_dim=6)
    model.logit_scale.data.fill_(math.log(10_000.0))
    text = functional.normalize(torch.randn(3, 6), dim=-1)

    output = model(
        torch.randn(1, 4, 32, 32),
        torch.randn(1, 3, 32, 32),
        torch.randn(1, 3, 32, 32),
        text,
    )

    cosine = torch.einsum("ndhw,cd->nchw", output.pixel_embeddings, text)
    torch.testing.assert_close(output.logits, cosine * 100.0, atol=1e-4, rtol=1e-4)


def test_model_rejects_unregistered_input_shapes() -> None:
    model = make_native_model(hsi_bands=4, lidar_channels=3, feature_dim=8, text_dim=6)

    with pytest.raises(ValueError, match="空间尺寸"):
        model(
            torch.randn(1, 4, 32, 32),
            torch.randn(1, 3, 31, 32),
            torch.randn(1, 3, 32, 32),
            functional.normalize(torch.randn(3, 6), dim=-1),
        )
