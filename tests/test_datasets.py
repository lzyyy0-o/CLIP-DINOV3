from __future__ import annotations

import numpy as np
import pytest
import torch

from hsi_lidar_ovseg.data.datasets import PairedTileDataset
from hsi_lidar_ovseg.data.io import DataError, SceneArrays
from hsi_lidar_ovseg.data.preprocessing import ChannelStats, NormalizationStats


def _paired_scene() -> SceneArrays:
    labels = np.indices((6, 6)).sum(axis=0) % 2 + 1
    shared = labels.astype(np.float32)
    hsi = np.repeat(shared[..., None], 3, axis=-1)
    lidar = shared[..., None]
    mask = np.ones((6, 6), dtype=np.bool_)
    return SceneArrays(
        hsi=hsi,
        lidar=lidar,
        labels=labels.astype(np.int64),
        train_mask=mask,
        test_mask=mask.copy(),
    )


def _identity_stats() -> NormalizationStats:
    return NormalizationStats(
        hsi=ChannelStats(
            mean=np.zeros(3, dtype=np.float32),
            scale=np.ones(3, dtype=np.float32),
        ),
        lidar=ChannelStats(
            mean=np.zeros(1, dtype=np.float32),
            scale=np.ones(1, dtype=np.float32),
        ),
    )


def _imbalanced_scene() -> SceneArrays:
    labels = np.ones((16, 16), dtype=np.int64)
    labels[14, 14] = 2
    shared = np.indices((16, 16)).sum(axis=0).astype(np.float32)
    mask = np.ones((16, 16), dtype=np.bool_)
    return SceneArrays(
        hsi=np.repeat(shared[..., None], 3, axis=-1),
        lidar=shared[..., None],
        labels=labels,
        train_mask=mask,
        test_mask=mask.copy(),
    )


def _sparse_rare_scene() -> SceneArrays:
    labels = np.zeros((16, 16), dtype=np.int64)
    labels[:4, :4] = 1
    labels[14, 14] = 2
    shared = np.indices((16, 16)).sum(axis=0).astype(np.float32)
    mask = labels > 0
    return SceneArrays(
        hsi=np.repeat(shared[..., None], 3, axis=-1),
        lidar=shared[..., None],
        labels=labels,
        train_mask=mask,
        test_mask=mask.copy(),
    )


def test_spatial_transform_is_shared_across_modalities_and_labels() -> None:
    dataset = PairedTileDataset(
        _paired_scene(),
        _identity_stats(),
        pseudo_rgb_indices=(0, 1, 2),
        tile_size=4,
        min_seen_pixels=1,
        seen_ids=(1, 2),
        training=True,
        seed=7,
    )

    sample = dataset[0]

    torch.testing.assert_close(sample["hsi"][0], sample["lidar"][0])
    torch.testing.assert_close(sample["hsi"][0], sample["labels"].float())
    assert sample["hsi"].shape == (3, 4, 4)
    assert sample["lidar"].shape == (3, 4, 4)
    assert sample["pseudo_rgb"].shape == (3, 4, 4)
    assert sample["valid_mask"].dtype == torch.bool


def test_dataset_is_deterministic_for_seed_and_index() -> None:
    arguments = {
        "scene": _paired_scene(),
        "stats": _identity_stats(),
        "pseudo_rgb_indices": (0, 1, 2),
        "tile_size": 4,
        "min_seen_pixels": 1,
        "seen_ids": (1, 2),
        "training": True,
        "seed": 19,
    }
    first = PairedTileDataset(**arguments)[1]
    second = PairedTileDataset(**arguments)[1]

    for key in ("hsi", "lidar", "pseudo_rgb", "labels", "valid_mask", "origin"):
        torch.testing.assert_close(first[key], second[key])


def test_training_dataset_repeatedly_samples_rare_seen_class() -> None:
    dataset = PairedTileDataset(
        _imbalanced_scene(),
        _identity_stats(),
        pseudo_rgb_indices=(0, 1, 2),
        tile_size=4,
        min_seen_pixels=1,
        seen_ids=(1, 2),
        training=True,
        seed=23,
    )

    rare_class_tiles = sum(
        bool(((sample["labels"] == 2) & sample["valid_mask"]).any()) for sample in dataset
    )

    assert rare_class_tiles >= 10


def test_training_dataset_can_disable_class_aware_sampling() -> None:
    dataset = PairedTileDataset(
        _imbalanced_scene(),
        _identity_stats(),
        pseudo_rgb_indices=(0, 1, 2),
        tile_size=4,
        min_seen_pixels=1,
        seen_ids=(1, 2),
        training=True,
        seed=23,
        class_aware_sampling=False,
        class_aware_fraction=1.0,
    )

    rare_class_tiles = sum(
        bool(((sample["labels"] == 2) & sample["valid_mask"]).any()) for sample in dataset
    )

    assert rare_class_tiles == 1


def test_training_dataset_honors_zero_class_aware_fraction() -> None:
    dataset = PairedTileDataset(
        _imbalanced_scene(),
        _identity_stats(),
        pseudo_rgb_indices=(0, 1, 2),
        tile_size=4,
        min_seen_pixels=1,
        seen_ids=(1, 2),
        training=True,
        seed=23,
        class_aware_sampling=True,
        class_aware_fraction=0.0,
    )

    rare_class_tiles = sum(
        bool(((sample["labels"] == 2) & sample["valid_mask"]).any()) for sample in dataset
    )

    assert rare_class_tiles == 1


def test_dataset_epoch_changes_sampling_but_remains_reproducible() -> None:
    dataset = PairedTileDataset(
        _imbalanced_scene(),
        _identity_stats(),
        pseudo_rgb_indices=(0, 1, 2),
        tile_size=4,
        min_seen_pixels=1,
        seen_ids=(1, 2),
        training=True,
        seed=29,
    )

    epoch_zero = [dataset[index]["origin"].clone() for index in range(len(dataset))]
    dataset.set_epoch(1)
    epoch_one = [dataset[index]["origin"].clone() for index in range(len(dataset))]
    dataset.set_epoch(0)
    replay = [dataset[index]["origin"].clone() for index in range(len(dataset))]

    assert any(
        not torch.equal(first, second) for first, second in zip(epoch_zero, epoch_one, strict=True)
    )
    for first, second in zip(epoch_zero, replay, strict=True):
        torch.testing.assert_close(first, second)


def test_class_aware_sampling_preserves_min_seen_pixels() -> None:
    dataset = PairedTileDataset(
        _sparse_rare_scene(),
        _identity_stats(),
        pseudo_rgb_indices=(0, 1, 2),
        tile_size=4,
        min_seen_pixels=4,
        seen_ids=(1, 2),
        training=True,
        seed=31,
        class_aware_sampling=True,
        class_aware_fraction=1.0,
    )

    for epoch in range(12):
        dataset.set_epoch(epoch)
        for sample in dataset:
            assert int(sample["valid_mask"].sum()) >= 4


def test_dataset_rejects_scene_without_eligible_training_tile() -> None:
    scene = _paired_scene()
    scene = SceneArrays(
        hsi=scene.hsi,
        lidar=scene.lidar,
        labels=scene.labels,
        train_mask=np.zeros_like(scene.train_mask),
        test_mask=scene.test_mask,
    )

    with pytest.raises(DataError, match="满足"):
        PairedTileDataset(
            scene,
            _identity_stats(),
            pseudo_rgb_indices=(0, 1, 2),
            tile_size=4,
            min_seen_pixels=1,
            seen_ids=(1, 2),
            training=True,
            seed=0,
        )
