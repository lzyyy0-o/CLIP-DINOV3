"""Category-shared spatial and text-guided cost-volume aggregation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _WindowAttentionBlock(nn.Module):
    """Apply window attention to `[B, N, H, W, C]` cost tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, window_size: int, shift_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _attention_mask(self, height: int, width: int, inputs: Tensor) -> Tensor | None:
        if self.shift_size == 0:
            return None
        window = self.window_size
        region = torch.zeros((1, height, width, 1), device=inputs.device, dtype=inputs.dtype)
        slices = (
            slice(0, -window),
            slice(-window, -self.shift_size),
            slice(-self.shift_size, None),
        )
        identifier = 0
        for height_slice in slices:
            for width_slice in slices:
                region[:, height_slice, width_slice] = identifier
                identifier += 1
        window_regions = (
            region.reshape(1, height // window, window, width // window, window, 1)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window * window)
        )
        differences = window_regions[:, None, :] - window_regions[:, :, None]
        return differences.masked_fill(differences != 0, float("-inf")).masked_fill(
            differences == 0, 0.0
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch, classes, height, width, channels = inputs.shape
        window = self.window_size
        if height % window or width % window:
            raise ValueError(f"相关性网格高宽必须能被窗口大小 {window} 整除")
        shifted = (
            torch.roll(inputs, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            if self.shift_size
            else inputs
        )
        windows = (
            shifted.reshape(
                batch, classes, height // window, window, width // window, window, channels
            )
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(
                batch * classes * (height // window) * (width // window),
                window * window,
                channels,
            )
        )
        mask = self._attention_mask(height, width, inputs)
        if mask is not None:
            mask = mask.repeat(batch * classes, 1, 1).repeat_interleave(
                self.attention.num_heads, dim=0
            )
        normalized = self.norm1(windows)
        attended = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )[0]
        windows = windows + attended
        windows = windows + self.mlp(self.norm2(windows))
        restored = (
            windows.reshape(
                batch, classes, height // window, width // window, window, window, channels
            )
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(batch, classes, height, width, channels)
        )
        return (
            torch.roll(restored, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            if self.shift_size
            else restored
        )


class SpatialWindowAggregator(nn.Module):
    """Aggregate each category map through regular and shifted windows."""

    def __init__(self, hidden_dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        if window_size <= 1 or window_size % 2 == 0:
            raise ValueError("window_size 必须是大于 1 的奇数")
        self.regular = _WindowAttentionBlock(hidden_dim, num_heads, window_size, shift_size=0)
        self.shifted = _WindowAttentionBlock(
            hidden_dim, num_heads, window_size, shift_size=window_size // 2
        )

    def forward(self, cost: Tensor) -> Tensor:
        if cost.ndim != 5:
            raise ValueError("成本体必须是 [B,C,N,H,W] 张量")
        spatial = cost.permute(0, 2, 3, 4, 1)
        spatial = self.regular(spatial)
        spatial = self.shifted(spatial)
        return spatial.permute(0, 4, 1, 2, 3)


class TextGuidedClassAggregator(nn.Module):
    """Exchange information across dynamic classes at every spatial position."""

    def __init__(self, hidden_dim: int, text_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, cost: Tensor, text: Tensor) -> Tensor:
        if text.ndim != 3 or text.shape[-1] != self.text_dim:
            raise ValueError(f"文本特征必须是 [N,P,{self.text_dim}] 张量")
        batch, channels, classes, height, width = cost.shape
        if channels != self.hidden_dim:
            raise ValueError(f"成本体通道必须为 {self.hidden_dim}")
        if text.shape[0] != classes or text.shape[1] == 0:
            raise ValueError("文本类别数与成本体类别数必须一致, 且提示词数必须为正")
        tokens = cost.permute(0, 3, 4, 2, 1).reshape(batch * height * width, classes, channels)
        guidance = self.text_projection(text.mean(dim=1)).unsqueeze(0).expand(
            tokens.shape[0], -1, -1
        )
        normalized = self.norm1(tokens)
        query = self.query(torch.cat((normalized, guidance), dim=-1))
        key = self.key(torch.cat((normalized, guidance), dim=-1))
        attended = self.attention(query, key, normalized, need_weights=False)[0]
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.reshape(batch, height, width, classes, channels).permute(0, 4, 3, 1, 2)


class CorrelationAggregatorLayer(nn.Module):
    """One spatial-window plus text-guided class aggregation layer."""

    def __init__(self, hidden_dim: int, text_dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or text_dim <= 0:
            raise ValueError("hidden_dim 和 text_dim 必须为正整数")
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("num_heads 必须整除 hidden_dim")
        self.spatial = SpatialWindowAggregator(hidden_dim, num_heads, window_size)
        self.classes = TextGuidedClassAggregator(hidden_dim, text_dim, num_heads)

    def forward(self, cost: Tensor, text: Tensor) -> Tensor:
        return self.classes(self.spatial(cost), text)
