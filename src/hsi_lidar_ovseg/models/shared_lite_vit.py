"""Six-layer shared Lite-ViT encoder for paired HSI and LiDAR rasters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True)
class SharedTokenOutput:
    """Four HSI and LiDAR token stages sharing one spatial token grid."""

    hsi_tokens: tuple[Tensor, Tensor, Tensor, Tensor]
    lidar_tokens: tuple[Tensor, Tensor, Tensor, Tensor]
    grid_size: tuple[int, int]


class SharedLiteViT(nn.Module):
    """Encode HSI and LiDAR with modality-specific inputs and shared Transformer blocks."""

    patch_size = 16
    embed_dim = 384
    stage_blocks = (0, 1, 3, 5)

    def __init__(self, hsi_bands: int, lidar_channels: int) -> None:
        super().__init__()
        if hsi_bands <= 0 or lidar_channels <= 0:
            raise ValueError("HSI 波段数和 LiDAR 通道数必须为正整数")
        self.spectral_adapter = nn.Conv2d(hsi_bands, self.embed_dim, kernel_size=1)
        self.hsi_patch_embed = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.lidar_patch_embed = nn.Conv2d(
            lidar_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.hsi_position = nn.Parameter(torch.zeros(1, self.embed_dim, 14, 14))
        self.lidar_position = nn.Parameter(torch.zeros(1, self.embed_dim, 14, 14))
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=6,
                dim_feedforward=self.embed_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(6)
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.hsi_position, std=0.02)
        nn.init.trunc_normal_(self.lidar_position, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _tokens(self, features: Tensor, position: Tensor) -> tuple[Tensor, tuple[int, int]]:
        height, width = features.shape[-2:]
        resized_position = functional.interpolate(
            position, size=(height, width), mode="bicubic", align_corners=False
        )
        tokens = (features + resized_position).flatten(2).transpose(1, 2)
        return tokens, (height, width)

    def forward(self, hsi: Tensor, lidar: Tensor) -> SharedTokenOutput:
        if hsi.ndim != 4 or lidar.ndim != 4:
            raise ValueError("HSI 和 LiDAR 输入必须为 NCHW 张量")
        if hsi.shape[0] != lidar.shape[0]:
            raise ValueError("HSI 和 LiDAR 的批量大小必须一致")
        if hsi.shape[-2:] != lidar.shape[-2:]:
            raise ValueError("HSI 和 LiDAR 的空间尺寸必须一致")
        if hsi.shape[-2] % self.patch_size or hsi.shape[-1] % self.patch_size:
            raise ValueError("Shared Lite-ViT 输入空间尺寸必须能被 16 整除")

        hsi_maps = self.hsi_patch_embed(self.spectral_adapter(hsi))
        lidar_maps = self.lidar_patch_embed(lidar)
        hsi_tokens, grid_size = self._tokens(hsi_maps, self.hsi_position)
        lidar_tokens, lidar_grid_size = self._tokens(lidar_maps, self.lidar_position)
        if grid_size != lidar_grid_size:
            raise ValueError("HSI 和 LiDAR 的 token 网格必须一致")

        hsi_stages: list[Tensor] = []
        lidar_stages: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            hsi_tokens = block(hsi_tokens)
            lidar_tokens = block(lidar_tokens)
            if index in self.stage_blocks:
                hsi_stages.append(self.norm(hsi_tokens))
                lidar_stages.append(self.norm(lidar_tokens))
        return SharedTokenOutput(
            hsi_tokens=tuple(hsi_stages),  # type: ignore[arg-type]
            lidar_tokens=tuple(lidar_stages),  # type: ignore[arg-type]
            grid_size=grid_size,
        )
