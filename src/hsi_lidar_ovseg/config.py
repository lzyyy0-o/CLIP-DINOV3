"""Strict, typed experiment configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


_ARRAY_SUFFIXES = {".mat", ".npy", ".npz"}
_ConfigT = TypeVar("_ConfigT")


@dataclass(frozen=True)
class DataConfig:
    """Paths, array keys, and class protocol for one registered scene."""

    name: str
    hsi_path: Path
    lidar_path: Path
    labels_path: Path
    train_mask_path: Path
    test_mask_path: Path
    hsi_key: str | None
    lidar_key: str | None
    labels_key: str | None
    train_mask_key: str | None
    test_mask_key: str | None
    class_names: tuple[str, ...]
    seen_class_ids: tuple[int, ...]
    unseen_class_ids: tuple[int, ...]
    pseudo_rgb_indices: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("data.name 不能为空")
        if not self.class_names or any(not name.strip() for name in self.class_names):
            raise ConfigError("class_names 必须包含非空类别名称")
        if len(set(self.class_names)) != len(self.class_names):
            raise ConfigError("class_names 不得重复")

        seen = set(self.seen_class_ids)
        unseen = set(self.unseen_class_ids)
        overlap = seen & unseen
        if overlap:
            raise ConfigError(f"seen_class_ids 与 unseen_class_ids 重叠: {sorted(overlap)}")
        expected = set(range(1, len(self.class_names) + 1))
        observed = seen | unseen
        if observed != expected:
            raise ConfigError(
                "类别编号必须从 1 开始连续覆盖全部 class_names; "
                f"期望 {sorted(expected)}, 实际 {sorted(observed)}"
            )
        if len(self.pseudo_rgb_indices) != 3 or any(index < 0 for index in self.pseudo_rgb_indices):
            raise ConfigError("pseudo_rgb_indices 必须包含三个非负索引")
        for field_name in (
            "hsi_path",
            "lidar_path",
            "labels_path",
            "train_mask_path",
            "test_mask_path",
        ):
            path = getattr(self, field_name)
            if path.suffix.lower() not in _ARRAY_SUFFIXES:
                raise ConfigError(f"{field_name} 仅支持 .mat、.npy 或 .npz: {path}")

    @property
    def num_classes(self) -> int:
        """Return the number of positive semantic classes."""

        return len(self.class_names)

    def validate_files(self) -> None:
        """Require every configured scene file to exist locally."""

        for field_name in (
            "hsi_path",
            "lidar_path",
            "labels_path",
            "train_mask_path",
            "test_mask_path",
        ):
            path = getattr(self, field_name)
            if not path.is_file():
                raise ConfigError(f"{field_name} 文件不存在: {path}")


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration shared by native and external visual encoders."""

    kind: str
    checkpoint: Path | None = None
    factory: str | None = None
    model_name: str | None = None
    feature_blocks: tuple[int, ...] = (2, 5, 8, 11)
    frozen: bool = False
    unfreeze_blocks: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"native", "hypersigma", "dinov2"}:
            raise ConfigError(f"不支持的 encoder.kind: {self.kind}")
        if (
            len(self.feature_blocks) != 4
            or tuple(sorted(self.feature_blocks)) != self.feature_blocks
        ):
            raise ConfigError("feature_blocks 必须包含四个严格递增的层编号")
        if self.unfreeze_blocks < 0:
            raise ConfigError("unfreeze_blocks 不得为负数")
        if self.kind != "native" and self.checkpoint is None:
            raise ConfigError(f"encoder.kind={self.kind} 时必须提供 checkpoint")
        if self.kind != "native" and not self.factory:
            raise ConfigError(f"encoder.kind={self.kind} 时必须提供 factory")

    def validate_files(self) -> None:
        """Require external checkpoints to exist."""

        if self.checkpoint is not None and not self.checkpoint.is_file():
            raise ConfigError(f"编码器检查点不存在: {self.checkpoint}")


@dataclass(frozen=True)
class ModelConfig:
    """Multimodal segmentor dimensions and backbone choices."""

    hsi_encoder: EncoderConfig
    lidar_encoder: EncoderConfig
    teacher_encoder: EncoderConfig
    clip_checkpoint: Path | None
    clip_model_name: str = "ViT-B-32"
    prompt_templates: tuple[str, ...] = (
        "a remote sensing image of {}",
        "an aerial view of {}",
    )
    feature_dim: int = 256
    text_dim: int = 512
    terrain_window: int = 9

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ConfigError("feature_dim 必须为正整数")
        if self.text_dim <= 0:
            raise ConfigError("text_dim 必须为正整数")
        if not self.clip_model_name.strip():
            raise ConfigError("clip_model_name 不能为空")
        if not self.prompt_templates or any(
            template.count("{}") != 1 for template in self.prompt_templates
        ):
            raise ConfigError("每个 prompt_templates 项必须且只能包含一个 {} 占位符")
        if self.terrain_window <= 0 or self.terrain_window % 2 == 0:
            raise ConfigError("terrain_window 必须为正奇数")

    def validate_files(self) -> None:
        """Validate every configured local model artifact."""

        self.hsi_encoder.validate_files()
        self.lidar_encoder.validate_files()
        self.teacher_encoder.validate_files()
        if self.clip_checkpoint is not None and not self.clip_checkpoint.is_file():
            raise ConfigError(f"CLIP 检查点不存在: {self.clip_checkpoint}")


@dataclass(frozen=True)
class LossConfig:
    """Weights for supervised, alignment, and regularization terms."""

    teacher_weight: float = 1.0
    cross_weight: float = 0.5
    gate_weight: float = 0.01
    private_weight: float = 0.01
    temperature: float = 0.1

    def __post_init__(self) -> None:
        weights = {
            "teacher_weight": self.teacher_weight,
            "cross_weight": self.cross_weight,
            "gate_weight": self.gate_weight,
            "private_weight": self.private_weight,
        }
        invalid = [name for name, value in weights.items() if value < 0]
        if invalid:
            raise ConfigError(f"损失权重不得为负数: {', '.join(invalid)}")
        if self.temperature <= 0:
            raise ConfigError("temperature 必须为正数")


@dataclass(frozen=True)
class TrainConfig:
    """Optimization and tiling parameters."""

    tile_size: int = 224
    overlap: int = 56
    min_seen_pixels: int = 1
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    amp: bool = True
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ConfigError("tile_size 必须为正整数")
        if not 0 <= self.overlap < self.tile_size:
            raise ConfigError("overlap 必须满足 0 <= overlap < tile_size")
        if self.min_seen_pixels <= 0:
            raise ConfigError("min_seen_pixels 必须为正整数")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ConfigError("batch_size 和 epochs 必须为正整数")
        if self.learning_rate <= 0 or self.backbone_learning_rate <= 0:
            raise ConfigError("学习率必须为正数")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ConfigError("weight_decay 不得为负; gradient_clip 必须为正")
        if self.device not in {"cpu", "cuda"}:
            raise ConfigError("device 必须是 cpu 或 cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete reproducible experiment configuration."""

    name: str
    seed: int
    output_dir: Path
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig

    def validate(self, *, check_files: bool) -> None:
        """Validate cross-component constraints and optional local artifacts."""

        if not self.name.strip():
            raise ConfigError("name 不能为空")
        if self.seed < 0:
            raise ConfigError("seed 不得为负数")
        if check_files:
            self.data.validate_files()
            self.model.validate_files()


def _reject_unknown(raw: Mapping[str, Any], cls: type[object], context: str) -> None:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{context} 包含 unknown 配置键: {', '.join(unknown)}")


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} 必须是映射")
    return value


def _path(value: object, context: str, *, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"{context} 必须是路径字符串")
    return Path(value)


def _tuple(value: object, context: str, item_type: type[Any]) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{context} 必须是序列")
    if any(not isinstance(item, item_type) for item in value):
        raise ConfigError(f"{context} 包含错误的元素类型")
    return tuple(value)


def _decode_encoder(raw: object, context: str) -> EncoderConfig:
    mapping = _require_mapping(raw, context)
    _reject_unknown(mapping, EncoderConfig, context)
    values = dict(mapping)
    values["checkpoint"] = _path(values.get("checkpoint"), f"{context}.checkpoint", optional=True)
    if "feature_blocks" in values:
        values["feature_blocks"] = _tuple(
            values["feature_blocks"], f"{context}.feature_blocks", int
        )
    try:
        return EncoderConfig(**values)
    except TypeError as error:
        raise ConfigError(f"{context} 缺少或包含非法字段: {error}") from error


def _decode_data(raw: object) -> DataConfig:
    mapping = _require_mapping(raw, "data")
    _reject_unknown(mapping, DataConfig, "data")
    values = dict(mapping)
    for field_name in (
        "hsi_path",
        "lidar_path",
        "labels_path",
        "train_mask_path",
        "test_mask_path",
    ):
        values[field_name] = _path(values.get(field_name), f"data.{field_name}")
    values["class_names"] = _tuple(values.get("class_names"), "data.class_names", str)
    values["seen_class_ids"] = _tuple(values.get("seen_class_ids"), "data.seen_class_ids", int)
    values["unseen_class_ids"] = _tuple(
        values.get("unseen_class_ids"), "data.unseen_class_ids", int
    )
    rgb = _tuple(values.get("pseudo_rgb_indices"), "data.pseudo_rgb_indices", int)
    values["pseudo_rgb_indices"] = rgb
    try:
        return DataConfig(**values)
    except TypeError as error:
        raise ConfigError(f"data 缺少或包含非法字段: {error}") from error


def _decode_model(raw: object) -> ModelConfig:
    mapping = _require_mapping(raw, "model")
    _reject_unknown(mapping, ModelConfig, "model")
    values = dict(mapping)
    for field_name in ("hsi_encoder", "lidar_encoder", "teacher_encoder"):
        values[field_name] = _decode_encoder(values.get(field_name), f"model.{field_name}")
    values["clip_checkpoint"] = _path(
        values.get("clip_checkpoint"), "model.clip_checkpoint", optional=True
    )
    if "prompt_templates" in values:
        values["prompt_templates"] = _tuple(
            values["prompt_templates"], "model.prompt_templates", str
        )
    try:
        return ModelConfig(**values)
    except TypeError as error:
        raise ConfigError(f"model 缺少或包含非法字段: {error}") from error


def _decode_simple(cls: type[_ConfigT], raw: object, context: str) -> _ConfigT:
    mapping = _require_mapping(raw, context)
    _reject_unknown(mapping, cls, context)
    try:
        return cls(**mapping)
    except TypeError as error:
        raise ConfigError(f"{context} 缺少或包含非法字段: {error}") from error


def load_config(path: Path, *, check_files: bool = True) -> ExperimentConfig:
    """Load a YAML experiment with strict key and value validation."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"无法读取配置文件 {path}: {error}") from error
    mapping = _require_mapping(raw, "root")
    _reject_unknown(mapping, ExperimentConfig, "root")
    values = dict(mapping)
    values["output_dir"] = _path(values.get("output_dir"), "output_dir")
    values["data"] = _decode_data(values.get("data"))
    values["model"] = _decode_model(values.get("model"))
    values["loss"] = _decode_simple(LossConfig, values.get("loss"), "loss")
    values["train"] = _decode_simple(TrainConfig, values.get("train"), "train")
    try:
        config = ExperimentConfig(**values)
    except TypeError as error:
        raise ConfigError(f"root 缺少或包含非法字段: {error}") from error
    config.validate(check_files=check_files)
    return config
