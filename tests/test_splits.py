from __future__ import annotations

import numpy as np

from hsi_lidar_ovseg.data.splits import split_training_mask


def test_split_training_mask_is_reproducible_and_stratified() -> None:
    labels = np.array([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=np.int64)
    original = np.ones_like(labels, dtype=np.bool_)

    train_a, validation_a = split_training_mask(labels, original, (1, 2), 0.25, 7)
    train_b, validation_b = split_training_mask(labels, original, (1, 2), 0.25, 7)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert not np.any(train_a & validation_a)
    assert [int(np.sum(validation_a & (labels == class_id))) for class_id in (1, 2)] == [1, 1]
    assert [int(np.sum(train_a & (labels == class_id))) for class_id in (1, 2)] == [3, 3]


def test_split_training_mask_keeps_singleton_and_unseen_pixels_in_training() -> None:
    labels = np.array([[1, 1, 2, 3]], dtype=np.int64)
    original = np.ones_like(labels, dtype=np.bool_)

    training, validation = split_training_mask(labels, original, (1, 2), 0.5, 19)

    assert bool(training[0, 2])
    assert not bool(validation[0, 2])
    assert bool(training[0, 3])
    assert not bool(validation[0, 3])
