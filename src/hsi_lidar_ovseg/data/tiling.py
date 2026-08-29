"""Deterministic tile coverage and weighted sliding-window accumulation."""

from __future__ import annotations

import torch

from hsi_lidar_ovseg.data.io import DataError


def _axis_origins(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    last = length - tile_size
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return tuple(origins)


def tile_origins(
    height: int, width: int, tile_size: int, overlap: int
) -> tuple[tuple[int, int], ...]:
    """Return top-left positions that cover a scene including its far edges."""

    if height <= 0 or width <= 0:
        raise DataError("height 和 width 必须为正整数")
    if tile_size <= 0:
        raise DataError("tile_size 必须为正整数")
    if not 0 <= overlap < tile_size:
        raise DataError("overlap 必须满足 0 <= overlap < tile_size")
    rows = _axis_origins(height, tile_size, overlap)
    columns = _axis_origins(width, tile_size, overlap)
    return tuple((top, left) for top in rows for left in columns)


def _blend_window(tile_size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    one_dimensional = torch.hann_window(
        tile_size, periodic=False, dtype=dtype, device=device
    ).clamp_min(1e-3)
    return one_dimensional[:, None] * one_dimensional[None, :]


class SlidingWindowAccumulator:
    """Blend overlapping dense logits and crop padded tiles to the scene."""

    def __init__(self, num_classes: int, height: int, width: int, tile_size: int) -> None:
        if num_classes <= 0 or height <= 0 or width <= 0 or tile_size <= 0:
            raise DataError("类别数、空间尺寸和 tile_size 必须为正整数")
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.tile_size = tile_size
        self._scores: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None

    def add(self, logits: torch.Tensor, top: int, left: int) -> None:
        """Accumulate one `[classes, tile, tile]` prediction at an origin."""

        expected = (self.num_classes, self.tile_size, self.tile_size)
        if tuple(logits.shape) != expected:
            raise DataError(f"logits 形状必须为 {expected}, 实际为 {tuple(logits.shape)}")
        if top < 0 or left < 0 or top >= self.height or left >= self.width:
            raise DataError(f"图块原点超出场景范围: {(top, left)}")
        if self._scores is None:
            self._scores = torch.zeros(
                (self.num_classes, self.height, self.width),
                dtype=logits.dtype,
                device=logits.device,
            )
            self._weights = torch.zeros(
                (self.height, self.width), dtype=logits.dtype, device=logits.device
            )
        elif logits.device != self._scores.device or logits.dtype != self._scores.dtype:
            raise DataError("所有图块必须使用相同的设备和数据类型")

        crop_height = min(self.tile_size, self.height - top)
        crop_width = min(self.tile_size, self.width - left)
        window = _blend_window(self.tile_size, dtype=logits.dtype, device=logits.device)[
            :crop_height, :crop_width
        ]
        assert self._scores is not None and self._weights is not None
        self._scores[:, top : top + crop_height, left : left + crop_width] += (
            logits[:, :crop_height, :crop_width] * window
        )
        self._weights[top : top + crop_height, left : left + crop_width] += window

    def finalize(self) -> torch.Tensor:
        """Return normalized logits after verifying complete scene coverage."""

        if self._scores is None or self._weights is None:
            raise DataError("没有可用于重建的图块")
        if torch.any(self._weights <= 0):
            uncovered = int(torch.count_nonzero(self._weights <= 0).item())
            raise DataError(f"滑窗预测存在 {uncovered} 个未覆盖像素")
        return self._scores / self._weights.unsqueeze(0)
