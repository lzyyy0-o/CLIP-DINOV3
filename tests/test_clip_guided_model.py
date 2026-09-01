from __future__ import annotations

import torch
from torch import nn

from hsi_lidar_ovseg.models.clip_guided_model import CLIPGuidedSharedLiteViTSegmentor
from hsi_lidar_ovseg.models.shared_lite_vit import SharedTokenOutput


class _Shared(nn.Module):
    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> SharedTokenOutput:
        tokens = tuple(torch.randn(hsi.shape[0], 4, 384) for _ in range(4))
        return SharedTokenOutput(tokens, tokens, (2, 2))  # type: ignore[arg-type]


class _Pair(nn.Module):
    def forward(
        self, hsi: tuple[torch.Tensor, ...], lidar: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        return hsi, lidar


class _Joint(nn.Module):
    def forward(
        self, hsi: tuple[torch.Tensor, ...], _: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return hsi


class _Project(nn.Module):
    def forward(
        self, _: tuple[torch.Tensor, ...], __: tuple[int, int], ___: tuple[int, int]
    ) -> tuple[torch.Tensor, ...]:
        return tuple(torch.randn(1, 512, size, size) for size in (8, 4, 2, 1))


class _Clip(nn.Module):
    def visual_features(self, _: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(torch.randn(1, 512, size, size) for size in (8, 4, 2, 1))

    def text_features(self, names: tuple[str, ...]) -> torch.Tensor:
        return torch.randn(len(names), 2, 512)


class _Decoder(nn.Module):
    def forward(
        self,
        _: tuple[torch.Tensor, ...],
        __: tuple[torch.Tensor, ...],
        text: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        return torch.randn(1, text.shape[0], *size)


def test_clip_guided_model_outputs_logits_for_dynamic_class_names() -> None:
    model = CLIPGuidedSharedLiteViTSegmentor(
        _Shared(), _Pair(), _Joint(), _Project(), _Clip(), _Decoder()
    )

    output = model(
        torch.randn(1, 6, 32, 32),
        torch.randn(1, 3, 32, 32),
        torch.randn(1, 3, 32, 32),
        ("trees", "road"),
    )

    assert output.logits.shape == (1, 2, 32, 32)
    expected = ((1, 512, 8, 8), (1, 512, 4, 4), (1, 512, 2, 2), (1, 512, 1, 1))
    assert tuple(level.shape for level in output.joint_features) == expected
    assert tuple(level.shape for level in output.clip_features) == expected
