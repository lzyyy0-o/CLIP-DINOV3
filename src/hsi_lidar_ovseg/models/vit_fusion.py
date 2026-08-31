"""Transformer cross-modal fusion and token-to-pyramid projection."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import FeaturePyramid

TokenPyramid = tuple[Tensor, Tensor, Tensor, Tensor]


def _validate_pair(first: tuple[Tensor, ...], second: tuple[Tensor, ...]) -> None:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("跨模态融合必须接收四个 token 阶段")
    if any(left.shape != right.shape for left, right in zip(first, second, strict=True)):
        raise ValueError("HSI 和 LiDAR token 阶段形状必须一致")


class _CrossModalBlock(nn.Module):
    """Update both modalities with bidirectional cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hsi_norm = nn.LayerNorm(embed_dim)
        self.lidar_norm = nn.LayerNorm(embed_dim)
        self.hsi_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.lidar_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.hsi_ffn_norm = nn.LayerNorm(embed_dim)
        self.lidar_ffn_norm = nn.LayerNorm(embed_dim)
        self.hsi_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )
        self.lidar_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, hsi: Tensor, lidar: Tensor) -> tuple[Tensor, Tensor]:
        normalized_hsi = self.hsi_norm(hsi)
        normalized_lidar = self.lidar_norm(lidar)
        hsi = hsi + self.hsi_attention(normalized_hsi, normalized_lidar, normalized_lidar)[0]
        lidar = lidar + self.lidar_attention(normalized_lidar, normalized_hsi, normalized_hsi)[0]
        hsi = hsi + self.hsi_mlp(self.hsi_ffn_norm(hsi))
        lidar = lidar + self.lidar_mlp(self.lidar_ffn_norm(lidar))
        return hsi, lidar


class ViTMMFB(nn.Module):
    """Apply an independent bidirectional fusion block to every token stage."""

    def __init__(self, embed_dim: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_CrossModalBlock(embed_dim, num_heads) for _ in range(4))

    def forward(
        self, hsi_tokens: tuple[Tensor, ...], lidar_tokens: tuple[Tensor, ...]
    ) -> tuple[TokenPyramid, TokenPyramid]:
        _validate_pair(hsi_tokens, lidar_tokens)
        hsi_outputs: list[Tensor] = []
        lidar_outputs: list[Tensor] = []
        for block, hsi, lidar in zip(self.blocks, hsi_tokens, lidar_tokens, strict=True):
            hsi_output, lidar_output = block(hsi, lidar)
            hsi_outputs.append(hsi_output)
            lidar_outputs.append(lidar_output)
        return tuple(hsi_outputs), tuple(lidar_outputs)  # type: ignore[return-value]


class _ComplementaryFusionBlock(nn.Module):
    """Combine two updated modalities into a complementary joint token stream."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.merge = nn.Linear(embed_dim * 2, embed_dim)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, hsi: Tensor, lidar: Tensor) -> Tensor:
        merged = self.merge(torch.cat((hsi, lidar), dim=-1))
        normalized = self.attention_norm(merged)
        merged = merged + self.attention(normalized, normalized, normalized)[0]
        return merged + self.mlp(self.ffn_norm(merged))


class ViTCMFEB(nn.Module):
    """Apply complementary feature enhancement at all four stages."""

    def __init__(self, embed_dim: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _ComplementaryFusionBlock(embed_dim, num_heads) for _ in range(4)
        )

    def forward(
        self, hsi_tokens: tuple[Tensor, ...], lidar_tokens: tuple[Tensor, ...]
    ) -> TokenPyramid:
        _validate_pair(hsi_tokens, lidar_tokens)
        outputs = [
            block(hsi, lidar)
            for block, hsi, lidar in zip(self.blocks, hsi_tokens, lidar_tokens, strict=True)
        ]
        return tuple(outputs)  # type: ignore[return-value]


class TokenPyramidProjector(nn.Module):
    """Project four token stages into 1/4, 1/8, 1/16 and 1/32 feature maps."""

    out_strides = (4, 8, 16, 32)

    def __init__(self, input_dim: int = 384, output_dim: int = 512) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.projections = nn.ModuleList(nn.Conv2d(input_dim, output_dim, 1) for _ in range(4))

    def forward(
        self,
        tokens: tuple[Tensor, ...],
        grid_size: tuple[int, int],
        image_size: tuple[int, int],
    ) -> FeaturePyramid:
        if len(tokens) != 4:
            raise ValueError("token 金字塔必须包含四个阶段")
        grid_height, grid_width = grid_size
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("token 网格尺寸必须为正整数")
        expected_tokens = grid_height * grid_width
        outputs: list[Tensor] = []
        for stage, projection, stride in zip(
            tokens, self.projections, self.out_strides, strict=True
        ):
            if stage.ndim != 3 or stage.shape[1] != expected_tokens:
                raise ValueError("token 数量必须与 token 网格尺寸一致")
            feature = stage.transpose(1, 2).reshape(
                stage.shape[0], stage.shape[-1], grid_height, grid_width
            )
            output_size = (max(1, image_size[0] // stride), max(1, image_size[1] // stride))
            outputs.append(
                functional.interpolate(
                    projection(feature), size=output_size, mode="bilinear", align_corners=False
                )
            )
        return tuple(outputs)  # type: ignore[return-value]
