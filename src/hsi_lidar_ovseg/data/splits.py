"""Deterministic train/validation masks for seen-class supervision."""

from __future__ import annotations

import numpy as np

from hsi_lidar_ovseg.data.io import DataError


def split_training_mask(
    labels: np.ndarray,
    train_mask: np.ndarray,
    seen_ids: tuple[int, ...],
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split seen-class training pixels into reproducible train and validation masks."""

    if labels.ndim != 2 or train_mask.shape != labels.shape:
        raise DataError("labels 与 train_mask 必须是形状相同的二维数组")
    if not seen_ids or any(class_id <= 0 for class_id in seen_ids):
        raise DataError("seen_ids 必须包含正类别编号")
    if not 0.0 < validation_fraction < 1.0:
        raise DataError("validation_fraction 必须位于 (0, 1)")
    if seed < 0:
        raise DataError("seed 不得为负数")

    training = np.asarray(train_mask, dtype=np.bool_).copy()
    validation = np.zeros_like(training)
    for class_id in seen_ids:
        coordinates = np.argwhere(training & (labels == class_id))
        count = len(coordinates)
        if count < 2:
            continue
        validation_count = min(count - 1, max(1, round(count * validation_fraction)))
        generator = np.random.default_rng(np.random.SeedSequence((seed, class_id)))
        selected = coordinates[generator.choice(count, size=validation_count, replace=False)]
        validation[selected[:, 0], selected[:, 1]] = True
    training[validation] = False
    return training, validation
