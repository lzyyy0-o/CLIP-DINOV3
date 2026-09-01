from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_array

from hsi_lidar_ovseg.config import DataConfig
from hsi_lidar_ovseg.data.io import DataError, load_scene
from hsi_lidar_ovseg.data.rs_fusion import _import_rs_fusion_datasets


def _config(
    tmp_path: Path,
    dataset: str,
    samples_per_class: int | float | None = None,
) -> DataConfig:
    return DataConfig(
        name=dataset,
        hsi_path=None,
        lidar_path=None,
        labels_path=None,
        train_mask_path=None,
        test_mask_path=None,
        hsi_key=None,
        lidar_key=None,
        labels_key=None,
        train_mask_key=None,
        test_mask_key=None,
        class_names=("one", "two", "three"),
        seen_class_ids=(1, 2),
        unseen_class_ids=(3,),
        pseudo_rgb_indices=(0, 1, 2),
        source="rs_fusion_datasets",
        rs_dataset=dataset,  # type: ignore[arg-type]
        rs_data_home=tmp_path,
        rs_train_samples_per_class=samples_per_class,
    )


def test_houston2013_adapter_converts_chw_and_official_masks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = SimpleNamespace(
        fetch_houston2013=lambda *, data_home: (
            np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4),
            np.ones((1, 2, 4), dtype=np.float32),
            coo_array(np.array([[1, 0, 0, 0], [0, 2, 0, 0]], dtype=np.int64)),
            coo_array(np.array([[0, 0, 3, 0], [0, 0, 0, 1]], dtype=np.int64)),
            {},
        )
    )
    monkeypatch.setitem(sys.modules, "rs_fusion_datasets", fake)

    scene = load_scene(_config(tmp_path, "houston2013"), split_seed=13)

    assert scene.hsi.shape == (2, 4, 3)
    assert scene.lidar.shape == (2, 4, 1)
    assert scene.labels.tolist() == [[1, 0, 3, 0], [0, 2, 0, 1]]
    assert scene.train_mask.sum() == scene.test_mask.sum() == 2


def test_trento_adapter_samples_each_class_deterministically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    labels = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int64)
    fake = SimpleNamespace(
        fetch_trento=lambda *, data_home: (
            np.ones((3, 2, 3), dtype=np.float32),
            np.ones((1, 2, 3), dtype=np.float32),
            labels,
            {},
        )
    )
    monkeypatch.setitem(sys.modules, "rs_fusion_datasets", fake)
    config = _config(tmp_path, "trento", samples_per_class=2)

    first = load_scene(config, split_seed=7)
    second = load_scene(config, split_seed=7)

    np.testing.assert_array_equal(first.train_mask, second.train_mask)
    assert first.train_mask.sum() == 4
    assert first.test_mask.sum() == 2
    assert not np.any(first.train_mask & first.test_mask)


def test_rs_fusion_missing_dependency_has_install_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    del tmp_path
    real_import = builtins.__import__

    def missing_rs_fusion(name: str, *args: object, **kwargs: object) -> object:
        if name == "rs_fusion_datasets":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "rs_fusion_datasets", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_rs_fusion)

    with pytest.raises(DataError, match="rs-fusion-datasets"):
        _import_rs_fusion_datasets()
