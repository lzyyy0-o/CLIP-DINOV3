from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from hsi_lidar_ovseg.engine.checkpoint import (
    CheckpointError,
    CheckpointIdentity,
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)


def _identity(**overrides: object) -> CheckpointIdentity:
    values: dict[str, object] = {
        "class_names": ("tree", "road"),
        "seen_class_ids": (1,),
        "unseen_class_ids": (2,),
        "hsi_bands": 8,
        "lidar_channels": 3,
        "feature_dim": 16,
        "text_dim": 12,
    }
    values.update(overrides)
    return CheckpointIdentity(**values)  # type: ignore[arg-type]


def _state(model: nn.Module, optimizer: torch.optim.Optimizer) -> TrainingState:
    return TrainingState(
        identity=_identity(),
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        scaler_state=None,
        epoch=2,
        global_step=17,
        normalization={"hsi_mean": torch.zeros(8), "hsi_scale": torch.ones(8)},
        config={"name": "test"},
    )


def test_checkpoint_rejects_class_identity_mismatch(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(path, _state(model, optimizer))

    with pytest.raises(CheckpointError, match="class_names"):
        load_checkpoint(
            path,
            model,
            optimizer,
            _identity(class_names=("road", "tree")),
        )


def test_checkpoint_round_trip_restores_model_and_counters(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    original = model.weight.detach().clone()
    save_checkpoint(path, _state(model, optimizer))
    model.weight.data.zero_()

    restored = load_checkpoint(path, model, optimizer, _identity())

    torch.testing.assert_close(model.weight, original)
    assert restored.epoch == 2
    assert restored.global_step == 17
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_round_trip_preserves_selection_state(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    state = _state(model, optimizer)
    state.selection_state = {"best_score": 0.4, "epochs_without_improvement": 3}
    save_checkpoint(path, state)

    restored = load_checkpoint(path, model, optimizer, _identity())

    assert restored.selection_state == state.selection_state


def test_checkpoint_without_selection_state_remains_loadable(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(path, _state(model, optimizer))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.pop("selection_state", None)
    torch.save(payload, path)

    restored = load_checkpoint(path, model, optimizer, _identity())

    assert restored.selection_state is None
