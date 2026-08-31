from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hsi_lidar_ovseg.config import (
    ConfigError,
    DataConfig,
    EncoderConfig,
    TrainConfig,
    load_config,
)


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
            "structure_teacher_encoder": {"kind": "native", "checkpoint": None},
            "semantic_teacher_encoder": {"kind": "native", "checkpoint": None},
            "clip_checkpoint": None,
            "feature_dim": 64,
            "text_dim": 32,
            "terrain_window": 5,
        },
        "loss": {
            "structure_teacher_weight": 1.0,
            "semantic_teacher_weight": 1.0,
            "cross_weight": 0.5,
            "gate_weight": 0.01,
            "private_weight": 0.01,
            "temperature": 0.1,
        },
        "train": {
            "tile_size": 224,
            "overlap": 56,
            "min_seen_pixels": 1,
            "class_aware_sampling": True,
            "class_aware_fraction": 0.7,
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


def _clip_guided_config_dict(
    *, tile_size: int = 224, include_teacher: bool = False
) -> dict[str, object]:
    values = _valid_config_dict()
    model: dict[str, object] = {
        "architecture": "clip_guided_shared_lite_vit",
        "shared_lite_vit": {
            "patch_size": 16,
            "embed_dim": 384,
            "depths": [1, 1, 2, 2],
            "num_heads": 6,
            "mlp_ratio": 2.0,
        },
        "clip": {
            "checkpoint": "weights/openai_clip/ViT-B-16.pt",
            "model_name": "ViT-B/16",
            "feature_blocks": [2, 5, 8, 11],
            "unfreeze_blocks": 2,
        },
        "prompt_templates": [
            "aerial image of {}",
            "satellite image of {}",
            "top-down view of {}",
            "remote sensing image of {}",
        ],
        "feature_dim": 512,
        "text_dim": 512,
        "terrain_window": 5,
    }
    if include_teacher:
        model["hsi_encoder"] = {"kind": "native"}
    values["model"] = model
    values["loss"] = {"kind": "masked_cross_entropy"}
    train = values["train"]
    assert isinstance(train, dict)
    train["tile_size"] = tile_size
    return values


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
    assert config.train.class_aware_sampling
    assert config.train.class_aware_fraction == 0.7


def test_clip_guided_config_accepts_dynamic_vocabulary_without_teachers(tmp_path: Path) -> None:
    path = tmp_path / "clip-guided.yaml"
    path.write_text(yaml.safe_dump(_clip_guided_config_dict()), encoding="utf-8")

    config = load_config(path, check_files=False)

    assert config.model.architecture == "clip_guided_shared_lite_vit"
    assert config.model.shared_lite_vit is not None
    assert config.model.shared_lite_vit.depths == (1, 1, 2, 2)
    assert config.model.clip is not None
    assert config.model.clip.model_name == "ViT-B/16"
    assert config.loss.kind == "masked_cross_entropy"


@pytest.mark.parametrize(
    ("tile_size", "include_teacher", "message"),
    [(192, False, "224"), (224, True, "教师")],
)
def test_clip_guided_config_rejects_incompatible_fields(
    tmp_path: Path, tile_size: int, include_teacher: bool, message: str
) -> None:
    path = tmp_path / "bad-clip-guided.yaml"
    path.write_text(
        yaml.safe_dump(
            _clip_guided_config_dict(tile_size=tile_size, include_teacher=include_teacher)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        load_config(path, check_files=False)


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


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_train_config_rejects_invalid_class_aware_fraction(value: float) -> None:
    with pytest.raises(ConfigError, match="class_aware_fraction"):
        TrainConfig(class_aware_fraction=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_fraction", 0.0),
        ("validation_fraction", 1.0),
        ("early_stopping_patience", 0),
        ("early_stopping_min_delta", -0.1),
        ("cosine_eta_min", 1e-5),
    ],
)
def test_train_config_rejects_invalid_validation_controls(field: str, value: float | int) -> None:
    with pytest.raises(ConfigError):
        TrainConfig(**{field: value})


def test_encoder_config_rejects_frozen_partial_unfreezing() -> None:
    with pytest.raises(ConfigError, match="unfreeze_blocks"):
        EncoderConfig(
            kind="dinov2",
            checkpoint=Path("weights.pt"),
            factory="package:create_model",
            frozen=True,
            unfreeze_blocks=2,
        )


@pytest.mark.parametrize("kind", ["hypersigma", "dinov2", "dinov3_vit", "dinov3_convnext"])
def test_external_encoder_kinds_require_factory(kind: str) -> None:
    with pytest.raises(ConfigError, match="factory"):
        EncoderConfig(kind=kind, checkpoint=Path("weights.pt"))


def test_hypersigma_requires_both_component_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="spatial_checkpoint 与 spectral_checkpoint"):
        EncoderConfig(
            kind="hypersigma",
            factory="hsi_lidar_ovseg.models.factories:create_hypersigma",
            source_dir=tmp_path,
            spatial_checkpoint=tmp_path / "spatial.pt",
            pretrained_in_channels=100,
        )


def test_dinov3_requires_source_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="source_dir"):
        EncoderConfig(
            kind="dinov3_convnext",
            checkpoint=tmp_path / "dino.pt",
            factory="hsi_lidar_ovseg.models.factories:create_dinov3",
        )


def test_remoteclip_encoder_does_not_require_factory() -> None:
    config = EncoderConfig(
        kind="remoteclip",
        checkpoint=Path("remoteclip.pt"),
        model_name="ViT-L-14",
        frozen=True,
    )

    assert config.factory is None


def test_remoteclip_teacher_must_share_text_tower_checkpoint(tmp_path: Path) -> None:
    values = _valid_config_dict()
    model = values["model"]
    assert isinstance(model, dict)
    model["clip_checkpoint"] = "text.pt"
    model["semantic_teacher_encoder"] = {
        "kind": "remoteclip",
        "checkpoint": "vision.pt",
        "model_name": "ViT-B-32",
        "frozen": True,
    }
    path = tmp_path / "mismatched-remoteclip.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"同一.*checkpoint"):
        load_config(path, check_files=False)


def test_remoteclip_teacher_must_share_text_tower_model_name(tmp_path: Path) -> None:
    values = _valid_config_dict()
    model = values["model"]
    assert isinstance(model, dict)
    model["clip_checkpoint"] = "remoteclip.pt"
    model["clip_model_name"] = "ViT-B-32"
    model["semantic_teacher_encoder"] = {
        "kind": "remoteclip",
        "checkpoint": "remoteclip.pt",
        "model_name": "ViT-L-14",
        "frozen": True,
    }
    path = tmp_path / "mismatched-model-name.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ConfigError, match="model_name"):
        load_config(path, check_files=False)


def test_model_config_rejects_hugging_face_hub_clip_name(tmp_path: Path) -> None:
    values = _valid_config_dict()
    model = values["model"]
    assert isinstance(model, dict)
    model["clip_model_name"] = "hf-hub:organization/model"
    path = tmp_path / "remote-clip.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ConfigError, match="hf-hub"):
        load_config(path, check_files=False)


def test_pretrained_template_has_only_local_external_resources() -> None:
    config = load_config(Path("configs/pretrained.yaml"), check_files=False)

    assert config.model.hsi_encoder.spatial_checkpoint is not None
    assert config.model.hsi_encoder.spectral_checkpoint is not None
    assert config.model.lidar_encoder.source_dir == Path("third_party/dinov3")
    assert config.model.semantic_teacher_encoder.kind == "remoteclip"
