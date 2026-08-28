from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from hsi_lidar_ovseg.config import DataConfig
from hsi_lidar_ovseg.data.io import DataError, load_array, load_scene


def _save_array(path: Path, array: np.ndarray, key: str = "cube") -> None:
    if path.suffix == ".npy":
        np.save(path, array)
    elif path.suffix == ".npz":
        np.savez(path, **{key: array})
    elif path.suffix == ".mat":
        savemat(path, {key: array})
    else:  # pragma: no cover - test helper guard
        raise AssertionError(path.suffix)


@pytest.mark.parametrize("suffix", [".npy", ".npz", ".mat"])
def test_load_array_supports_declared_formats(tmp_path: Path, suffix: str) -> None:
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / f"array{suffix}"
    _save_array(path, expected)

    actual = load_array(path, key=None if suffix == ".npy" else "cube")

    np.testing.assert_array_equal(actual, expected)


def test_load_array_reports_available_keys(tmp_path: Path) -> None:
    path = tmp_path / "arrays.npz"
    np.savez(path, hsi=np.ones((2, 2)), labels=np.zeros((2, 2)))

    with pytest.raises(DataError, match=r"hsi.*labels"):
        load_array(path, key="missing")


def _data_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        name="demo",
        hsi_path=tmp_path / "hsi.npy",
        lidar_path=tmp_path / "lidar.npy",
        labels_path=tmp_path / "labels.npy",
        train_mask_path=tmp_path / "train.npy",
        test_mask_path=tmp_path / "test.npy",
        hsi_key=None,
        lidar_key=None,
        labels_key=None,
        train_mask_key=None,
        test_mask_key=None,
        class_names=("one", "two"),
        seen_class_ids=(1,),
        unseen_class_ids=(2,),
        pseudo_rgb_indices=(0, 1, 2),
    )


def _write_valid_scene(config: DataConfig) -> None:
    labels = np.array([[0, 1, 1, 0], [2, 2, 0, 0], [0, 1, 2, 0]], dtype=np.int64)
    hsi = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
    lidar = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    train = labels == 1
    test = labels == 2
    np.save(config.hsi_path, hsi)
    np.save(config.lidar_path, lidar)
    np.save(config.labels_path, labels)
    np.save(config.train_mask_path, train)
    np.save(config.test_mask_path, test)


def test_load_scene_normalizes_arrays_to_channel_last(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    _write_valid_scene(config)

    scene = load_scene(config)

    assert scene.hsi.shape == (3, 4, 5)
    assert scene.lidar.shape == (3, 4, 1)
    assert scene.labels.dtype == np.int64
    assert scene.train_mask.dtype == np.bool_
    assert scene.test_mask.dtype == np.bool_


def test_load_scene_rejects_unpaired_spatial_shapes(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    _write_valid_scene(config)
    np.save(config.lidar_path, np.zeros((7, 8), dtype=np.float32))

    with pytest.raises(DataError, match="空间尺寸"):
        load_scene(config)


def test_load_scene_rejects_non_finite_hsi(tmp_path: Path) -> None:
    config = _data_config(tmp_path)
    _write_valid_scene(config)
    hsi = np.load(config.hsi_path)
    hsi[0, 0, 0] = np.nan
    np.save(config.hsi_path, hsi)

    with pytest.raises(DataError, match="非有限"):
        load_scene(config)
