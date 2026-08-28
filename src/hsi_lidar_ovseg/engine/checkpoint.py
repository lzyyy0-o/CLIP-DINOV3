"""Atomic, identity-checked training checkpoints."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is unreadable or incompatible."""


@dataclass(frozen=True)
class CheckpointIdentity:
    """Fields that must match before training state can be restored."""

    class_names: tuple[str, ...]
    seen_class_ids: tuple[int, ...]
    unseen_class_ids: tuple[int, ...]
    hsi_bands: int
    lidar_channels: int
    feature_dim: int
    text_dim: int


@dataclass
class TrainingState:
    """Serializable state required for deterministic experiment recovery."""

    identity: CheckpointIdentity
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
    epoch: int
    global_step: int
    normalization: dict[str, Tensor]
    config: dict[str, Any]


def _identity_payload(identity: CheckpointIdentity) -> dict[str, Any]:
    return {field.name: getattr(identity, field.name) for field in fields(identity)}


def _state_payload(state: TrainingState) -> dict[str, Any]:
    return {
        "identity": _identity_payload(state.identity),
        "model_state": state.model_state,
        "optimizer_state": state.optimizer_state,
        "scheduler_state": state.scheduler_state,
        "scaler_state": state.scaler_state,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "normalization": state.normalization,
        "config": state.config,
    }


def save_checkpoint(path: Path, state: TrainingState) -> None:
    """Atomically serialize a training state in the target directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(_state_payload(state), temporary_path)
        temporary_path.replace(path)
    except (OSError, RuntimeError) as error:
        raise CheckpointError(f"无法保存检查点 {path}: {error}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _decode_state(payload: object, path: Path) -> TrainingState:
    if not isinstance(payload, dict):
        raise CheckpointError(f"检查点根节点必须是字典: {path}")
    required = {
        "identity",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "epoch",
        "global_step",
        "normalization",
        "config",
    }
    missing = required - set(payload)
    if missing:
        raise CheckpointError(f"检查点缺少字段: {', '.join(sorted(missing))}")
    try:
        identity = CheckpointIdentity(**payload["identity"])
        return TrainingState(
            identity=identity,
            model_state=payload["model_state"],
            optimizer_state=payload["optimizer_state"],
            scheduler_state=payload["scheduler_state"],
            scaler_state=payload["scaler_state"],
            epoch=int(payload["epoch"]),
            global_step=int(payload["global_step"]),
            normalization=payload["normalization"],
            config=payload["config"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(f"检查点字段类型无效: {error}") from error


def _check_identity(actual: CheckpointIdentity, expected: CheckpointIdentity) -> None:
    conflicts = [
        field.name
        for field in fields(CheckpointIdentity)
        if getattr(actual, field.name) != getattr(expected, field.name)
    ]
    if conflicts:
        raise CheckpointError(f"检查点身份不兼容: {', '.join(conflicts)}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    expected: CheckpointIdentity,
    *,
    scheduler: LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> TrainingState:
    """Validate identity, then restore model and optimizer states."""

    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise CheckpointError(f"无法读取检查点 {path}: {error}") from error
    state = _decode_state(payload, path)
    _check_identity(state.identity, expected)
    try:
        model.load_state_dict(state.model_state, strict=True)
        optimizer.load_state_dict(state.optimizer_state)
        if scheduler is not None and state.scheduler_state is not None:
            scheduler.load_state_dict(state.scheduler_state)
        if scaler is not None and state.scaler_state is not None:
            scaler.load_state_dict(state.scaler_state)
    except (RuntimeError, ValueError, KeyError) as error:
        raise CheckpointError(f"无法恢复训练状态: {error}") from error
    return state
