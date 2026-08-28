from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hsi_lidar_ovseg.config import ConfigError, DataConfig, EncoderConfig, load_config


def _valid_data_config(**overrides: object) -> DataConfig:
    values: dict[str, object] = {
        "name": "demo",
        "hsi_path": Path("hsi.npy"),
        "lidar_path": Path("lidar.npy"),
        "labels_path": Path("labels.npy"),
        "train_mask_path": Path("train.npy"),
        "test_mask_path": Path("test.npy"),
        "hsi_key": None,
        "lidar_key": None,
        "labels_key": None,
        "train_mask_key": None,
        "test_mask_key": None,
        "class_names": ("tree", "road", "water"),
        "seen_class_ids": (1, 2),
        "unseen_class_ids": (3,),
        "pseudo_rgb_indices": (0, 1, 2),
    }
    values.update(overrides)
    return DataConfig(**values)  # type: ignore[arg-type]


def _valid_config_dict() -> dict[str, object]:
    return {
        "name": "demo",
        "seed": 7,
        "output_dir": "outputs/demo",
        "data": {
            "name": "demo",
            "hsi_path": "hsi.npy",
            "lidar_path": "lidar.npy",
            "labels_path": "labels.npy",
            "train_mask_path": "train.npy",
            "test_mask_path": "test.npy",
            "hsi_key": None,
            "lidar_key": None,
            "labels_key": None,
            "train_mask_key": None,
            "test_mask_key": None,
            "class_names": ["tree", "road", "water"],
            "seen_class_ids": [1, 2],
            "unseen_class_ids": [3],
            "pseudo_rgb_indices": [0, 1, 2],
        },
        "model": {
            "hsi_encoder": {"kind": "native", "checkpoint": None},
            "lidar_encoder": {"kind": "native", "checkpoint": None},
            "teacher_encoder": {"kind": "native", "checkpoint": None},
            "clip_checkpoint": None,
            "feature_dim": 64,
            "text_dim": 32,
            "terrain_window": 5,
        },
        "loss": {
            "teacher_weight": 1.0,
            "cross_weight": 0.5,
            "gate_weight": 0.01,
            "private_weight": 0.01,
            "temperature": 0.1,
        },
        "train": {
            "tile_size": 224,
            "overlap": 56,
            "min_seen_pixels": 1,
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.0001,
            "backbone_learning_rate": 0.00001,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "amp": True,
            "device": "cuda",
        },
    }


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    values = _valid_config_dict()
    values["unknown"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown"):
        load_config(path, check_files=False)


def test_data_config_rejects_overlapping_seen_and_unseen() -> None:
    with pytest.raises(ConfigError, match="重叠"):
        _valid_data_config(unseen_class_ids=(2, 3))


def test_data_config_rejects_non_contiguous_class_ids() -> None:
    with pytest.raises(ConfigError, match="连续"):
        _valid_data_config(seen_class_ids=(1,), unseen_class_ids=(3,))


def test_load_config_converts_sequences_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "valid.yaml"
    path.write_text(yaml.safe_dump(_valid_config_dict()), encoding="utf-8")

    config = load_config(path, check_files=False)

    assert config.data.class_names == ("tree", "road", "water")
    assert config.data.hsi_path == Path("hsi.npy")
    assert config.model.feature_dim == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("terrain_window", 4, "奇数"),
        ("feature_dim", 0, "feature_dim"),
    ],
)
def test_model_config_rejects_invalid_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    values = _valid_config_dict()
    model = values["model"]
    assert isinstance(model, dict)
    model[field] = value
    path = tmp_path / "invalid-model.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path, check_files=False)


def test_train_config_rejects_overlap_not_smaller_than_tile(tmp_path: Path) -> None:
    values = _valid_config_dict()
    train = values["train"]
    assert isinstance(train, dict)
    train["overlap"] = 224
    path = tmp_path / "invalid-train.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ConfigError, match="overlap"):
        load_config(path, check_files=False)


def test_encoder_config_rejects_frozen_partial_unfreezing() -> None:
    with pytest.raises(ConfigError, match="unfreeze_blocks"):
        EncoderConfig(
            kind="dinov2",
            checkpoint=Path("weights.pt"),
            factory="package:create_model",
            frozen=True,
            unfreeze_blocks=2,
        )


def test_model_config_rejects_hugging_face_hub_clip_name(tmp_path: Path) -> None:
    values = _valid_config_dict()
    model = values["model"]
    assert isinstance(model, dict)
    model["clip_model_name"] = "hf-hub:organization/model"
    path = tmp_path / "remote-clip.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ConfigError, match="hf-hub"):
        load_config(path, check_files=False)
