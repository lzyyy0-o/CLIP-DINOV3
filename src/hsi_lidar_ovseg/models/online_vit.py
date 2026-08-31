"""Online ViT-S/16 students for HSI and LiDAR feature extraction."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import FeaturePyramid


class OnlineViTPyramidEncoder(nn.Module):
    """Train a ViT-S/16 from scratch and expose a four-scale feature pyramid."""

    patch_size = 16
    embed_dim = 384
    out_strides = (4, 8, 16, 32)
    out_channels = (384, 384, 384, 384)
    feature_blocks = (2, 5, 8, 11)

    def __init__(self, in_channels: int, *, spectral_adapter: bool) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("在线 ViT 输入通道数必须为正整数")
        self.spectral_adapter = (
            nn.Conv2d(in_channels, self.embed_dim, kernel_size=1) if spectral_adapter else None
        )
        patch_channels = self.embed_dim if spectral_adapter else in_channels
        self.patch_embed = nn.Conv2d(
            patch_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.position = nn.Parameter(torch.zeros(1, self.embed_dim, 14, 14))
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=6,
                dim_feedforward=self.embed_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(12)
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.pyramid_projections = nn.ModuleList(
            nn.Conv2d(self.embed_dim, channels, kernel_size=1) for channels in self.out_channels
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> FeaturePyramid:
        if inputs.ndim != 4:
            raise ValueError(f"在线 ViT 输入必须为 NCHW 张量, 实际形状为 {tuple(inputs.shape)}")
        if inputs.shape[-2] % self.patch_size or inputs.shape[-1] % self.patch_size:
            raise ValueError("在线 ViT 输入空间尺寸必须能被 16 整除")
        features = self.spectral_adapter(inputs) if self.spectral_adapter is not None else inputs
        tokens_2d = self.patch_embed(features)
        height, width = tokens_2d.shape[-2:]
        position = functional.interpolate(
            self.position, size=(height, width), mode="bicubic", align_corners=False
        )
        tokens = (tokens_2d + position).flatten(2).transpose(1, 2)
        extracted: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in self.feature_blocks:
                extracted.append(
                    self.norm(tokens).transpose(1, 2).reshape(-1, self.embed_dim, height, width)
                )
        outputs: list[Tensor] = []
        for feature, projection, stride in zip(
            extracted, self.pyramid_projections, self.out_strides, strict=True
        ):
            outputs.append(
                functional.interpolate(
                    projection(feature),
                    size=(max(1, inputs.shape[-2] // stride), max(1, inputs.shape[-1] // stride)),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return tuple(outputs)  # type: ignore[return-value]
