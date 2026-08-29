"""Leakage-free normalization and modality-specific derived channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from hsi_lidar_ovseg.data.io import DataError, SceneArrays


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel location and positive scale."""

    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class NormalizationStats:
    """HSI standard and LiDAR robust normalization statistics."""

    hsi: ChannelStats
    lidar: ChannelStats


def _training_values(array: np.ndarray, train_mask: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 3 or train_mask.shape != array.shape[:2]:
        raise DataError(f"{name} 与 train_mask 形状不兼容: {array.shape} 和 {train_mask.shape}")
    if not np.any(train_mask):
        raise DataError("train_mask 不包含任何像素")
    values = array[train_mask]
    if not np.isfinite(values).all():
        raise DataError(f"{name} 训练像素包含非有限值")
    return values.astype(np.float64, copy=False)


def fit_hsi_stats(
    hsi: np.ndarray, train_mask: np.ndarray, *, epsilon: float = 1e-6
) -> ChannelStats:
    """Fit per-band mean and standard deviation using training pixels only."""

    values = _training_values(hsi, train_mask, "hsi")
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), epsilon)
    return ChannelStats(mean.astype(np.float32), scale.astype(np.float32))


def fit_lidar_stats(
    lidar: np.ndarray, train_mask: np.ndarray, *, epsilon: float = 1e-6
) -> ChannelStats:
    """Fit per-channel median and interquartile range on training pixels."""

    values = _training_values(lidar, train_mask, "lidar")
    median = np.median(values, axis=0)
    lower, upper = np.percentile(values, (25.0, 75.0), axis=0)
    scale = np.maximum(upper - lower, epsilon)
    return ChannelStats(median.astype(np.float32), scale.astype(np.float32))


def fit_normalization(scene: SceneArrays) -> NormalizationStats:
    """Fit all modality statistics without using validation or test pixels."""

    return NormalizationStats(
        hsi=fit_hsi_stats(scene.hsi, scene.train_mask),
        lidar=fit_lidar_stats(scene.lidar, scene.train_mask),
    )


def _normalize(array: np.ndarray, stats: ChannelStats, name: str) -> np.ndarray:
    if stats.mean.shape != (array.shape[-1],) or stats.scale.shape != (array.shape[-1],):
        raise DataError(f"{name} 统计量通道数与数组不匹配")
    if np.any(stats.scale <= 0) or not np.isfinite(stats.scale).all():
        raise DataError(f"{name} 归一化尺度必须为有限正数")
    return np.asarray((array - stats.mean) / stats.scale, dtype=np.float32)


def normalize_scene(scene: SceneArrays, stats: NormalizationStats) -> SceneArrays:
    """Apply fitted statistics while preserving labels and split masks."""

    return SceneArrays(
        hsi=np.ascontiguousarray(_normalize(scene.hsi, stats.hsi, "hsi")),
        lidar=np.ascontiguousarray(_normalize(scene.lidar, stats.lidar, "lidar")),
        labels=scene.labels,
        train_mask=scene.train_mask,
        test_mask=scene.test_mask,
    )


def terrain_channels(lidar: np.ndarray, window_size: int = 9) -> np.ndarray:
    """Build normalized height, local relative height, and slope magnitude."""

    if lidar.ndim != 3 or lidar.shape[-1] < 1:
        raise DataError(f"lidar 必须是至少单通道的 HWC 数组, 实际形状为 {lidar.shape}")
    if window_size <= 0 or window_size % 2 == 0:
        raise DataError("window_size 必须为正奇数")
    height = np.asarray(lidar[..., 0], dtype=np.float32)
    local_mean = uniform_filter(height, size=window_size, mode="reflect")
    relative = height - local_mean
    padded = np.pad(height, 1, mode="edge")
    gradient_y = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    gradient_x = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    slope = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    result = np.stack((height, relative, slope), axis=-1)
    if not np.isfinite(result).all():
        raise DataError("地形通道包含非有限值")
    return np.ascontiguousarray(result, dtype=np.float32)


def pseudo_rgb(
    hsi: np.ndarray,
    indices: tuple[int, int, int],
    train_mask: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Create a three-channel visible proxy with train-only percentile stretching."""

    if hsi.ndim != 3 or len(indices) != 3:
        raise DataError("hsi 必须是 HWC 数组, indices 必须包含三个索引")
    if min(indices) < 0 or max(indices) >= hsi.shape[-1]:
        raise DataError(f"indices 超过 HSI 波段范围 0..{hsi.shape[-1] - 1}: {indices}")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise DataError("百分位必须满足 0 <= lower < upper <= 100")
    selected = np.asarray(hsi[..., list(indices)], dtype=np.float32)
    values = _training_values(selected, train_mask, "pseudo_rgb")
    lower = np.percentile(values, lower_percentile, axis=0)
    upper = np.percentile(values, upper_percentile, axis=0)
    scale = np.maximum(upper - lower, epsilon)
    stretched = np.clip((selected - lower) / scale, 0.0, 1.0)
    return np.ascontiguousarray(stretched, dtype=np.float32)
