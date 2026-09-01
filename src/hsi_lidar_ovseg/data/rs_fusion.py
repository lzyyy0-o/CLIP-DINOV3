"""Adapt rs-fusion-datasets fetchers to the project's dense scene contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from hsi_lidar_ovseg.config import DataConfig
from hsi_lidar_ovseg.data.io import DataError, SceneArrays, _validate_scene_arrays


def _import_rs_fusion_datasets() -> Any:
    try:
        import rs_fusion_datasets
    except ImportError as error:
        raise DataError(
            "缺少 rs-fusion-datasets; 请执行 python -m pip install -r requirements.txt"
        ) from error
    return rs_fusion_datasets


def _dense_labels(values: object, name: str) -> np.ndarray:
    toarray = getattr(values, "toarray", None)
    array = np.asarray(toarray() if callable(toarray) else values)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise DataError(f"rs-fusion {name} 必须是二维标签图, 实际形状为 {array.shape}")
    return array


def _fetch(module: Any, function_name: str, data_home: object) -> tuple[Any, ...]:
    function = getattr(module, function_name, None)
    if not callable(function):
        raise DataError(f"rs-fusion-datasets 缺少 {function_name}; 请升级该依赖")
    try:
        result = function(data_home=data_home)
    except Exception as error:
        raise DataError(f"rs-fusion 加载失败 ({function_name}): {error}") from error
    if not isinstance(result, tuple):
        raise DataError(f"rs-fusion {function_name} 返回值必须是元组")
    return result


def stratified_rs_split(
    labels: np.ndarray, samples_per_class: int | float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic, non-overlapping masks for datasets without official splits."""

    labels = _dense_labels(labels, "labels")
    train_mask = np.zeros(labels.shape, dtype=np.bool_)
    for class_id in np.unique(labels):
        if class_id <= 0:
            continue
        positions = np.flatnonzero(labels.ravel() == class_id)
        if positions.size < 2:
            raise DataError(f"类别 {class_id} 少于两个像素, 无法生成训练/测试划分")
        if isinstance(samples_per_class, int):
            selected_count = min(samples_per_class, positions.size - 1)
        else:
            selected_count = min(
                max(1, round(positions.size * samples_per_class)), positions.size - 1
            )
        random = np.random.default_rng(np.random.SeedSequence((seed, int(class_id))))
        train_mask.ravel()[random.choice(positions, size=selected_count, replace=False)] = True
    test_mask = (labels > 0) & ~train_mask
    return train_mask, test_mask


def load_rs_fusion_scene(config: DataConfig, *, split_seed: int) -> SceneArrays:
    """Fetch a registered HSI-LiDAR scene and produce canonical dense arrays."""

    if config.source != "rs_fusion_datasets":
        raise DataError("load_rs_fusion_scene 仅适用于 source=rs_fusion_datasets")
    assert config.rs_dataset is not None
    assert config.rs_data_home is not None
    module = _import_rs_fusion_datasets()

    if config.rs_dataset == "houston2013":
        hsi, lidar, train_labels, test_labels, _info = _fetch(
            module, "fetch_houston2013", config.rs_data_home
        )
        train_values = _dense_labels(train_labels, "houston2013_train_labels")
        test_values = _dense_labels(test_labels, "houston2013_test_labels")
        labels = np.maximum(train_values, test_values)
        train_mask, test_mask = train_values > 0, test_values > 0
    elif config.rs_dataset == "houston2018_ouc":
        hsi, lidar, train_labels, test_labels, labels, _info = _fetch(
            module, "fetch_houston2018_ouc", config.rs_data_home
        )
        labels = _dense_labels(labels, "houston2018_labels")
        train_mask = _dense_labels(train_labels, "houston2018_train_labels") > 0
        test_mask = _dense_labels(test_labels, "houston2018_test_labels") > 0
    elif config.rs_dataset in {"trento", "muufl"}:
        function_name = "fetch_trento" if config.rs_dataset == "trento" else "fetch_muufl"
        hsi, lidar, labels, _info = _fetch(module, function_name, config.rs_data_home)
        labels = _dense_labels(labels, f"{config.rs_dataset}_labels")
        assert config.rs_train_samples_per_class is not None
        train_mask, test_mask = stratified_rs_split(
            labels, config.rs_train_samples_per_class, split_seed
        )
    else:  # pragma: no cover - DataConfig rejects this before loading.
        raise DataError(f"不支持的 rs_dataset: {config.rs_dataset}")

    return _validate_scene_arrays(
        config,
        np.asarray(hsi),
        np.asarray(lidar),
        np.asarray(labels),
        np.asarray(train_mask),
        np.asarray(test_mask),
    )
