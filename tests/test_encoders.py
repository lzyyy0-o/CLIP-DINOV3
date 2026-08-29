from __future__ import annotations

import torch
from torch import nn

from hsi_lidar_ovseg.models import (
    ClipTextEncoder,
    DinoV2Adapter,
    DinoV3ConvNeXtAdapter,
    DinoV3ViTAdapter,
    HyperSigmaAdapter,
    NativePyramidEncoder,
    RemoteClipVisionAdapter,
)


class FakeTokenBackbone(nn.Module):
    def __init__(self, embed_dim: int = 32, patch_size: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.projection = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(nn.Linear(embed_dim, embed_dim) for _ in range(4))

    def get_intermediate_layers(
        self,
        inputs: torch.Tensor,
        n: tuple[int, ...],
        *,
        reshape: bool,
        return_class_token: bool,
    ) -> tuple[torch.Tensor, ...]:
        assert not reshape
        assert not return_class_token
        tokens = self.projection(inputs).flatten(2).transpose(1, 2)
        class_token = torch.zeros(
            inputs.shape[0], 1, self.embed_dim, dtype=tokens.dtype, device=tokens.device
        )
        tokens = torch.cat((class_token, tokens), dim=1)
        return tuple(tokens + float(block) for block in n)


class FakeHyperSigmaBackbone(FakeTokenBackbone):
    def forward_intermediates(
        self, inputs: torch.Tensor, indices: tuple[int, ...]
    ) -> tuple[torch.Tensor, ...]:
        return self.get_intermediate_layers(
            inputs, indices, reshape=False, return_class_token=False
        )


class FakeClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        values = tokens.float()
        return torch.stack((values[:, 0], values[:, 1], values.sum(dim=1)), dim=-1)


class FakeDinoV3ConvNeXt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dims = (8, 16, 32, 64)
        self.downsample_layers = nn.ModuleList(
            (
                nn.Conv2d(3, 8, 4, stride=4),
                nn.Conv2d(8, 16, 2, stride=2),
                nn.Conv2d(16, 32, 2, stride=2),
                nn.Conv2d(32, 64, 2, stride=2),
            )
        )
        self.stages = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, channels, 1)) for channels in self.embed_dims
        )

    def get_intermediate_layers(
        self,
        inputs: torch.Tensor,
        n: tuple[int, ...],
        *,
        reshape: bool,
        return_class_token: bool,
    ) -> tuple[torch.Tensor, ...]:
        assert reshape
        assert not return_class_token
        selected = set(n)
        outputs = []
        features = inputs
        for index, (downsample, stage) in enumerate(
            zip(self.downsample_layers, self.stages, strict=True)
        ):
            features = stage(downsample(features))
            if index in selected:
                outputs.append(features)
        return tuple(outputs)


class FakeRemoteClipVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = (8, 8)
        self.conv1 = nn.Conv2d(3, 32, 8, stride=8)
        self.blocks = nn.ModuleList(nn.Conv2d(32, 32, 1) for _ in range(4))

    def forward_intermediates(
        self,
        inputs: torch.Tensor,
        *,
        indices: tuple[int, ...],
        stop_early: bool,
        normalize_intermediates: bool,
        intermediates_only: bool,
        output_fmt: str,
    ) -> dict[str, list[torch.Tensor]]:
        assert stop_early
        assert normalize_intermediates
        assert intermediates_only
        assert output_fmt == "NCHW"
        selected = set(indices)
        features = self.conv1(inputs)
        outputs = []
        for index, block in enumerate(self.blocks):
            features = block(features)
            if index in selected:
                outputs.append(features)
        return {"image_intermediates": outputs}


def fake_tokenizer(texts: list[str]) -> torch.Tensor:
    return torch.tensor([[len(text), sum(map(ord, text)) % 17] for text in texts])


def test_native_encoder_returns_four_level_pyramid() -> None:
    encoder = NativePyramidEncoder(16, channels=(16, 24, 32, 48))

    outputs = encoder(torch.randn(2, 16, 64, 64))

    assert [tuple(output.shape) for output in outputs] == [
        (2, 16, 16, 16),
        (2, 24, 8, 8),
        (2, 32, 4, 4),
        (2, 48, 2, 2),
    ]
    assert encoder.out_channels == (16, 24, 32, 48)
    assert encoder.out_strides == (4, 8, 16, 32)


def test_dino_adapter_restores_four_resolution_levels() -> None:
    adapter = DinoV2Adapter(
        FakeTokenBackbone(), feature_blocks=(1, 2, 3, 4), feature_dim=16, frozen=False
    )

    outputs = adapter(torch.randn(2, 3, 64, 64))

    assert [tuple(output.shape) for output in outputs] == [
        (2, 16, 16, 16),
        (2, 16, 8, 8),
        (2, 16, 4, 4),
        (2, 16, 2, 2),
    ]


def test_frozen_dino_adapter_keeps_backbone_in_eval() -> None:
    adapter = DinoV2Adapter(
        FakeTokenBackbone(), feature_blocks=(1, 2, 3, 4), feature_dim=16, frozen=True
    )

    adapter.train()

    assert not adapter.backbone.training
    assert not any(parameter.requires_grad for parameter in adapter.backbone.parameters())
    assert adapter.projections.training


def test_dino_adapter_only_unfreezes_requested_last_blocks() -> None:
    adapter = DinoV2Adapter(
        FakeTokenBackbone(),
        feature_blocks=(1, 2, 3, 4),
        feature_dim=16,
        frozen=False,
        unfreeze_blocks=2,
    )

    adapter.train()

    assert not any(
        parameter.requires_grad for parameter in adapter.backbone.blocks[:2].parameters()
    )
    assert all(parameter.requires_grad for parameter in adapter.backbone.blocks[2:].parameters())
    assert not adapter.backbone.blocks[0].training
    assert adapter.backbone.blocks[-1].training


def test_dinov3_vit_adapter_restores_four_resolution_levels() -> None:
    adapter = DinoV3ViTAdapter(
        FakeTokenBackbone(), feature_blocks=(1, 2, 3, 4), feature_dim=16, frozen=False
    )

    outputs = adapter(torch.randn(2, 3, 64, 64))

    assert [tuple(output.shape) for output in outputs] == [
        (2, 16, 16, 16),
        (2, 16, 8, 8),
        (2, 16, 4, 4),
        (2, 16, 2, 2),
    ]


def test_dinov3_convnext_adapter_preserves_native_pyramid() -> None:
    adapter = DinoV3ConvNeXtAdapter(FakeDinoV3ConvNeXt(), feature_stages=(0, 1, 2, 3), frozen=True)

    outputs = adapter(torch.randn(2, 3, 64, 64))

    assert [tuple(output.shape) for output in outputs] == [
        (2, 8, 16, 16),
        (2, 16, 8, 8),
        (2, 32, 4, 4),
        (2, 64, 2, 2),
    ]
    assert adapter.out_channels == (8, 16, 32, 64)


def test_dinov3_convnext_only_unfreezes_requested_last_stage() -> None:
    adapter = DinoV3ConvNeXtAdapter(
        FakeDinoV3ConvNeXt(),
        feature_stages=(0, 1, 2, 3),
        frozen=False,
        unfreeze_blocks=1,
    )

    adapter.train()

    assert not any(parameter.requires_grad for parameter in adapter.backbone.stages[0].parameters())
    assert all(parameter.requires_grad for parameter in adapter.backbone.stages[-1].parameters())
    assert not adapter.backbone.stages[0].training
    assert adapter.backbone.stages[-1].training


def test_remoteclip_visual_adapter_builds_four_scale_pyramid() -> None:
    adapter = RemoteClipVisionAdapter(
        FakeRemoteClipVisual(), feature_blocks=(0, 1, 2, 3), feature_dim=16
    )

    outputs = adapter(torch.randn(2, 3, 64, 64))
    adapter.train()

    assert [tuple(output.shape) for output in outputs] == [
        (2, 16, 16, 16),
        (2, 16, 8, 8),
        (2, 16, 4, 4),
        (2, 16, 2, 2),
    ]
    assert not adapter.backbone.training
    assert not any(parameter.requires_grad for parameter in adapter.backbone.parameters())


def test_hypersigma_adapter_accepts_forward_intermediates_backbone() -> None:
    adapter = HyperSigmaAdapter(
        FakeHyperSigmaBackbone(), feature_blocks=(1, 2, 3, 4), feature_dim=12
    )

    outputs = adapter(torch.randn(1, 3, 64, 64))

    assert len(outputs) == 4
    assert all(output.shape[1] == 12 for output in outputs)


def test_text_encoder_normalizes_prompt_ensemble() -> None:
    encoder = ClipTextEncoder(FakeClip(), fake_tokenizer, templates=("a {}", "satellite {}"))

    embeddings = encoder.encode(("tree", "road"))

    assert embeddings.shape == (2, 3)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2))
    assert not encoder.model.training
    assert not any(parameter.requires_grad for parameter in encoder.model.parameters())
