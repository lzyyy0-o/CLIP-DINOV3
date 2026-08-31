from __future__ import annotations

import torch
from torch import nn

from hsi_lidar_ovseg.models.openai_clip import OpenAIClipGuidance


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + tokens @ self.weight


class _Visual(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, width, 16, 16)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(_Block(width) for _ in range(12))
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, 512))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(image).flatten(2).permute(2, 0, 1)
        for block in self.transformer.resblocks:
            tokens = block(tokens)
        return tokens


class _FakeOpenAIClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _Visual(768)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(_Block(512) for _ in range(12))
        self.ln_final = nn.LayerNorm(512)
        self.text_projection = nn.Parameter(torch.eye(512))
        self.text_seed = nn.Parameter(torch.ones(1, 512))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.text_seed.expand(tokens.shape[0], -1)
        for block in self.transformer.resblocks:
            features = block(features[:, None])[:, 0]
        return self.ln_final(features) @ self.text_projection


def _tokenize(texts: list[str]) -> torch.Tensor:
    return torch.ones(len(texts), 77, dtype=torch.long)


def test_openai_clip_guidance_preserves_prompt_axis_and_partial_gradients() -> None:
    clip = _FakeOpenAIClip()
    guidance = OpenAIClipGuidance(
        clip,
        _tokenize,
        (2, 5, 8, 11),
        ("aerial image of {}", "satellite image of {}"),
    )

    text = guidance.text_features(("trees", "road", "water"))
    maps = guidance.visual_features(torch.randn(1, 3, 224, 224))
    (text.sum() + sum(item.mean() for item in maps)).backward()

    assert text.shape == (3, 2, 512)
    assert [item.shape for item in maps] == [
        (1, 512, 56, 56),
        (1, 512, 28, 28),
        (1, 512, 14, 14),
        (1, 512, 7, 7),
    ]
    assert clip.visual.transformer.resblocks[9].weight.grad is None
    assert clip.visual.transformer.resblocks[10].weight.grad is not None
    assert clip.transformer.resblocks[10].weight.grad is not None
