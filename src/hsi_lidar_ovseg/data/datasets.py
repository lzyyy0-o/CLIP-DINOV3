"""Paired HSI-LiDAR tile datasets with synchronized spatial augmentation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from hsi_lidar_ovseg.data.io import DataError, SceneArrays
from hsi_lidar_ovseg.data.preprocessing import (
    NormalizationStats,
    normalize_scene,
    pseudo_rgb,
    terrain_channels,
)
from hsi_lidar_ovseg.data.tiling import tile_origins


def _crop_and_pad(
    array: np.ndarray,
    top: int,
    left: int,
    tile_size: int,
    *,
    constant: float | int | bool | None = None,
) -> np.ndarray:
    bottom = min(top + tile_size, array.shape[0])
    right = min(left + tile_size, array.shape[1])
    cropped = array[top:bottom, left:right]
    pad_height = tile_size - cropped.shape[0]
    pad_width = tile_size - cropped.shape[1]
    if pad_height == 0 and pad_width == 0:
        return np.ascontiguousarray(cropped)
    padding = ((0, pad_height), (0, pad_width))
    if cropped.ndim == 3:
        padding += ((0, 0),)
    if constant is not None:
        return np.pad(cropped, padding, mode="constant", constant_values=constant)
    mode = "reflect" if cropped.shape[0] > 1 and cropped.shape[1] > 1 else "edge"
    return np.pad(cropped, padding, mode=mode)


def _spatial_transform(
    array: np.ndarray, rotations: int, flip_vertical: bool, flip_horizontal: bool
) -> np.ndarray:
    transformed = np.rot90(array, k=rotations, axes=(0, 1))
    if flip_vertical:
        transformed = np.flip(transformed, axis=0)
    if flip_horizontal:
        transformed = np.flip(transformed, axis=1)
    return np.ascontiguousarray(transformed)


def _float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(array, -1, 0))).float()


class PairedTileDataset(Dataset[Mapping[str, torch.Tensor]]):
    """Yield aligned multimodal tiles with deterministic per-index augmentation."""

    def __init__(
        self,
        scene: SceneArrays,
        stats: NormalizationStats,
        *,
        pseudo_rgb_indices: tuple[int, int, int],
        tile_size: int,
        min_seen_pixels: int,
        seen_ids: tuple[int, ...],
        training: bool,
        seed: int,
        terrain_window: int = 9,
    ) -> None:
        if tile_size <= 0 or min_seen_pixels <= 0:
            raise DataError("tile_size 和 min_seen_pixels 必须为正整数")
        if not seen_ids:
            raise DataError("seen_ids 不得为空")
        self.tile_size = tile_size
        self.training = training
        self.seed = seed

        split_mask = scene.train_mask if training else scene.test_mask
        supervised = split_mask & np.isin(scene.labels, seen_ids)
        overlap = tile_size // 2 if tile_size > 1 else 0
        candidates: list[tuple[int, int]] = []
        for top, left in tile_origins(*scene.spatial_shape, tile_size, overlap):
            bottom = min(top + tile_size, scene.spatial_shape[0])
            right = min(left + tile_size, scene.spatial_shape[1])
            if int(np.count_nonzero(supervised[top:bottom, left:right])) >= min_seen_pixels:
                candidates.append((top, left))
        if not candidates:
            raise DataError(
                "场景中没有满足 min_seen_pixels 的配对图块; "
                f"阈值={min_seen_pixels}, seen_ids={seen_ids}"
            )
        self.origins = tuple(candidates)

        normalized = normalize_scene(scene, stats)
        self.hsi = normalized.hsi
        self.lidar = terrain_channels(normalized.lidar, window_size=terrain_window)
        self.rgb = pseudo_rgb(scene.hsi, pseudo_rgb_indices, scene.train_mask)
        self.labels = scene.labels
        self.valid_mask = supervised if training else split_mask

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        top, left = self.origins[index]
        hsi = _crop_and_pad(self.hsi, top, left, self.tile_size)
        lidar = _crop_and_pad(self.lidar, top, left, self.tile_size)
        rgb = _crop_and_pad(self.rgb, top, left, self.tile_size)
        labels = _crop_and_pad(self.labels, top, left, self.tile_size, constant=0).astype(
            np.int64, copy=False
        )
        valid_mask = _crop_and_pad(
            self.valid_mask, top, left, self.tile_size, constant=False
        ).astype(np.bool_, copy=False)

        if self.training:
            generator = np.random.default_rng(np.random.SeedSequence((self.seed, index)))
            rotations = int(generator.integers(0, 4))
            flip_vertical = bool(generator.integers(0, 2))
            flip_horizontal = bool(generator.integers(0, 2))
            hsi = _spatial_transform(hsi, rotations, flip_vertical, flip_horizontal)
            lidar = _spatial_transform(lidar, rotations, flip_vertical, flip_horizontal)
            rgb = _spatial_transform(rgb, rotations, flip_vertical, flip_horizontal)
            labels = _spatial_transform(labels, rotations, flip_vertical, flip_horizontal)
            valid_mask = _spatial_transform(valid_mask, rotations, flip_vertical, flip_horizontal)

        return {
            "hsi": _float_tensor(hsi),
            "lidar": _float_tensor(lidar),
            "pseudo_rgb": _float_tensor(rgb),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)).long(),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask)).bool(),
            "origin": torch.tensor((top, left), dtype=torch.int64),
        }
