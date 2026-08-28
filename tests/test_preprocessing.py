from __future__ import annotations

import numpy as np

from hsi_lidar_ovseg.data.io import SceneArrays
from hsi_lidar_ovseg.data.preprocessing import (
    fit_hsi_stats,
    fit_normalization,
    normalize_scene,
    pseudo_rgb,
    terrain_channels,
)


def test_hsi_statistics_use_training_pixels_only() -> None:
    hsi = np.array([[[1.0], [3.0]], [[1000.0], [2000.0]]], dtype=np.float32)
    train_mask = np.array([[True, True], [False, False]])

    stats = fit_hsi_stats(hsi, train_mask)

    np.testing.assert_allclose(stats.mean, [2.0])
    np.testing.assert_allclose(stats.scale, [1.0])


def test_normalize_scene_uses_fitted_training_statistics() -> None:
    hsi = np.array([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)
    lidar = np.array([[[10.0], [20.0]], [[1000.0], [2000.0]]], dtype=np.float32)
    train_mask = np.array([[True, True], [False, False]])
    scene = SceneArrays(
        hsi=hsi,
        lidar=lidar,
        labels=np.array([[1, 1], [0, 0]], dtype=np.int64),
        train_mask=train_mask,
        test_mask=~train_mask,
    )

    normalized = normalize_scene(scene, fit_normalization(scene))

    np.testing.assert_allclose(normalized.hsi[0, :, 0], [-1.0, 1.0])
    np.testing.assert_allclose(normalized.lidar[0, :, 0], [-1.0, 1.0])


def test_terrain_channels_are_finite_for_constant_height() -> None:
    result = terrain_channels(np.ones((9, 9, 1), dtype=np.float32), window_size=5)

    assert result.shape == (9, 9, 3)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[..., 1:], 0.0)


def test_pseudo_rgb_uses_training_percentiles_and_clips_test_outlier() -> None:
    hsi = np.zeros((2, 2, 3), dtype=np.float32)
    hsi[0, 0] = [0.0, 10.0, 20.0]
    hsi[0, 1] = [10.0, 20.0, 30.0]
    hsi[1, 0] = [1000.0, 1000.0, 1000.0]
    train_mask = np.array([[True, True], [False, False]])

    rgb = pseudo_rgb(hsi, indices=(0, 1, 2), train_mask=train_mask)

    assert rgb.shape == (2, 2, 3)
    assert np.all((rgb >= 0.0) & (rgb <= 1.0))
    np.testing.assert_allclose(rgb[1, 0], 1.0)
