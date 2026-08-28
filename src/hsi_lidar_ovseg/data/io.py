"""Load co-registered HSI, LiDAR, label, and split arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from hsi_lidar_ovseg.config import DataConfig


class DataError(ValueError):
    """Raised when scene data violates the configured contract."""


@dataclass(frozen=True)
class SceneArrays:
    """Canonical channel-last arrays for one co-registered scene."""

    hsi: np.ndarray
    lidar: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return scene height and width."""

        return self.labels.shape


def _available_keys(keys: list[str]) -> str:
    visible = sorted(key for key in keys if not key.startswith("__"))
    return ", ".join(visible) if visible else "<none>"


def _select_array(arrays: dict[str, np.ndarray], key: str | None, path: Path) -> np.ndarray:
    visible = {name: value for name, value in arrays.items() if not name.startswith("__")}
    if key is None:
        if len(visible) != 1:
            raise DataError(
                f"{path} 包含多个数组, 必须配置数组键; 可用键: {_available_keys(list(visible))}"
            )
        return np.asarray(next(iter(visible.values())))
    if key not in visible:
        raise DataError(f"{path} 缺少数组键 {key!r}; 可用键: {_available_keys(list(visible))}")
    return np.asarray(visible[key])


def _load_hdf5_mat(path: Path, key: str | None) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        if key is None:
            if len(keys) != 1:
                raise DataError(
                    f"{path} 包含多个数组, 必须配置数组键; 可用键: {_available_keys(keys)}"
                )
            key = keys[0]
        if key not in handle:
            raise DataError(f"{path} 缺少数组键 {key!r}; 可用键: {_available_keys(keys)}")
        array = np.asarray(handle[key])
    if array.ndim >= 2:
        array = array.transpose(tuple(reversed(range(array.ndim))))
    return array


def load_array(path: Path, key: str | None) -> np.ndarray:
    """Load one NumPy or MATLAB array without allowing pickled objects."""

    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            if key is not None:
                raise DataError(f".npy 文件不使用数组键: {path}")
            return np.asarray(np.load(path, allow_pickle=False))
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            return _select_array(arrays, key, path)
        if suffix == ".mat":
            try:
                arrays = {name: np.asarray(value) for name, value in loadmat(path).items()}
                return _select_array(arrays, key, path)
            except NotImplementedError:
                return _load_hdf5_mat(path, key)
    except OSError as error:
        raise DataError(f"无法读取数组文件 {path}: {error}") from error
    raise DataError(f"不支持的数组格式 {suffix!r}: {path}")


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise DataError(f"{name} 必须是二维数组, 实际形状为 {array.shape}")
    return squeezed


def _as_channel_last(array: np.ndarray, spatial_shape: tuple[int, int], name: str) -> np.ndarray:
    if array.ndim == 2:
        if array.shape != spatial_shape:
            raise DataError(f"{name} 空间尺寸 {array.shape} 与标签 {spatial_shape} 不一致")
        return array[..., None]
    if array.ndim != 3:
        raise DataError(f"{name} 必须是二维或三维数组, 实际形状为 {array.shape}")
    if array.shape[:2] == spatial_shape:
        return array
    if array.shape[-2:] == spatial_shape:
        return np.moveaxis(array, 0, -1)
    raise DataError(f"{name} 空间尺寸无法与标签 {spatial_shape} 配准, 实际形状为 {array.shape}")


def _as_labels(array: np.ndarray) -> np.ndarray:
    labels = _as_2d(array, "labels")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise DataError("labels 必须包含有限整数")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0):
        raise DataError("labels 不得包含负数")
    return labels


def _as_mask(array: np.ndarray, name: str) -> np.ndarray:
    mask = _as_2d(array, name)
    if not np.isin(mask, (0, 1, False, True)).all():
        raise DataError(f"{name} 只能包含 0/1 或布尔值")
    return mask.astype(np.bool_, copy=False)


def load_scene(config: DataConfig) -> SceneArrays:
    """Load and validate one scene in canonical channel-last layout."""

    labels = _as_labels(load_array(config.labels_path, config.labels_key))
    spatial_shape = labels.shape
    hsi = _as_channel_last(load_array(config.hsi_path, config.hsi_key), spatial_shape, "hsi")
    lidar = _as_channel_last(
        load_array(config.lidar_path, config.lidar_key), spatial_shape, "lidar"
    )
    train_mask = _as_mask(load_array(config.train_mask_path, config.train_mask_key), "train_mask")
    test_mask = _as_mask(load_array(config.test_mask_path, config.test_mask_key), "test_mask")

    for name, array in (("hsi", hsi), ("lidar", lidar)):
        if not np.isfinite(array).all():
            raise DataError(f"{name} 包含非有限值")
    for name, mask in (("train_mask", train_mask), ("test_mask", test_mask)):
        if mask.shape != spatial_shape:
            raise DataError(f"{name} 空间尺寸 {mask.shape} 与标签 {spatial_shape} 不一致")
    if np.any(train_mask & test_mask):
        raise DataError("train_mask 与 test_mask 不得重叠")
    if labels.max(initial=0) > config.num_classes:
        raise DataError(
            f"labels 最大类别编号 {labels.max()} 超过 class_names 数量 {config.num_classes}"
        )
    if max(config.pseudo_rgb_indices) >= hsi.shape[-1]:
        raise DataError(
            f"pseudo_rgb_indices 超过 HSI 波段范围 0..{hsi.shape[-1] - 1}: "
            f"{config.pseudo_rgb_indices}"
        )

    return SceneArrays(
        hsi=np.ascontiguousarray(hsi, dtype=np.float32),
        lidar=np.ascontiguousarray(lidar, dtype=np.float32),
        labels=np.ascontiguousarray(labels),
        train_mask=np.ascontiguousarray(train_mask),
        test_mask=np.ascontiguousarray(test_mask),
    )
