from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch import nn

from hsi_lidar_ovseg import cli
from hsi_lidar_ovseg.cli import _build_model_and_text, deterministic_text_embeddings
from hsi_lidar_ovseg.config import (
    DataConfig,
    EncoderConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
)

ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + existing if existing else "")
    return environment


def test_cli_help_lists_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hsi_lidar_ovseg.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_environment(),
    )

    assert result.returncode == 0
    assert {"train", "evaluate", "validate-config"} <= set(result.stdout.split())


def test_cli_validates_all_example_configs_without_local_files() -> None:
    for name in ("houston2013.yaml", "trento.yaml", "muufl.yaml", "pretrained.yaml"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hsi_lidar_ovseg.cli",
                "validate-config",
                str(ROOT / "configs" / name),
                "--skip-file-checks",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=_environment(),
        )

        assert result.returncode == 0, result.stderr


def test_deterministic_text_embeddings_are_normalized_and_repeatable() -> None:
    first = deterministic_text_embeddings(("tree", "road"), text_dim=12)
    second = deterministic_text_embeddings(("tree", "road"), text_dim=12)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.norm(dim=-1), torch.ones(2))


class _FakeRemoteClipVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 8, stride=8)

    def forward_intermediates(self, inputs: torch.Tensor, **_: object) -> object:
        raise AssertionError("构建模型时不应运行视觉前向")


class _FakeRemoteClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _FakeRemoteClipVisual()
        self.text_anchor = nn.Parameter(torch.ones(()))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        values = tokens.float()
        return torch.stack((values[:, 0], values[:, 1], values.sum(dim=1)), dim=-1)


def _fake_tokenizer(texts: list[str]) -> torch.Tensor:
    return torch.tensor([[len(text), sum(map(ord, text)) % 17] for text in texts])


def test_remoteclip_model_is_loaded_once_for_text_and_visual_towers(
    tmp_path: Path, monkeypatch: object
) -> None:
    checkpoint = tmp_path / "remoteclip.pt"
    remoteclip = _FakeRemoteClip()
    calls: list[tuple[str, object]] = []

    def create_model(model_name: str, *, pretrained: object) -> nn.Module:
        calls.append((model_name, pretrained))
        return remoteclip

    fake_open_clip = SimpleNamespace(
        create_model=create_model, get_tokenizer=lambda _: _fake_tokenizer
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "_load_local_weights", lambda _module, _path: None)  # type: ignore[attr-defined]
    data = DataConfig(
        name="demo",
        hsi_path=Path("hsi.npy"),
        lidar_path=Path("lidar.npy"),
        labels_path=Path("labels.npy"),
        train_mask_path=Path("train.npy"),
        test_mask_path=Path("test.npy"),
        hsi_key=None,
        lidar_key=None,
        labels_key=None,
        train_mask_key=None,
        test_mask_key=None,
        class_names=("tree", "road"),
        seen_class_ids=(1,),
        unseen_class_ids=(2,),
        pseudo_rgb_indices=(0, 1, 2),
    )
    config = ExperimentConfig(
        name="remoteclip",
        seed=7,
        output_dir=tmp_path / "outputs",
        data=data,
        model=ModelConfig(
            hsi_encoder=EncoderConfig(kind="native"),
            lidar_encoder=EncoderConfig(kind="native"),
            structure_teacher_encoder=EncoderConfig(kind="native", frozen=True),
            semantic_teacher_encoder=EncoderConfig(
                kind="remoteclip",
                checkpoint=checkpoint,
                model_name="ViT-B-32",
                feature_blocks=(0, 1, 2, 3),
                frozen=True,
            ),
            clip_checkpoint=checkpoint,
            clip_model_name="ViT-B-32",
            feature_dim=8,
            text_dim=3,
        ),
        loss=LossConfig(),
        train=TrainConfig(device="cpu"),
    )

    model, embeddings = _build_model_and_text(config, hsi_bands=6)

    assert calls == [("ViT-B-32", None)]
    assert model.semantic_teacher_encoder.backbone is remoteclip.visual
    assert embeddings.shape == (2, 3)


def test_cli_selects_on_validation_and_writes_final_test_metrics(tmp_path: Path) -> None:
    generator = np.random.default_rng(9)
    height = width = 32
    paths = {
        "hsi_path": tmp_path / "hsi.npy",
        "lidar_path": tmp_path / "lidar.npy",
        "labels_path": tmp_path / "labels.npy",
        "train_mask_path": tmp_path / "train_mask.npy",
        "test_mask_path": tmp_path / "test_mask.npy",
    }
    np.save(paths["hsi_path"], generator.normal(size=(height, width, 6)).astype(np.float32))
    np.save(paths["lidar_path"], generator.normal(size=(height, width, 1)).astype(np.float32))
    labels = np.ones((height, width), dtype=np.int64)
    labels[:, width // 2 :] = 2
    np.save(paths["labels_path"], labels)
    train_mask = np.zeros((height, width), dtype=np.bool_)
    train_mask[: height // 2] = True
    test_mask = ~train_mask
    np.save(paths["train_mask_path"], train_mask)
    np.save(paths["test_mask_path"], test_mask)
    output_dir = tmp_path / "outputs"
    config = {
        "name": "cli-smoke",
        "seed": 5,
        "output_dir": str(output_dir),
        "data": {
            "name": "synthetic",
            **{name: str(path) for name, path in paths.items()},
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
            "hsi_encoder": {"kind": "native"},
            "lidar_encoder": {"kind": "native"},
            "structure_teacher_encoder": {"kind": "native", "frozen": True},
            "semantic_teacher_encoder": {"kind": "native", "frozen": True},
            "clip_checkpoint": None,
            "feature_dim": 8,
            "text_dim": 10,
            "terrain_window": 3,
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
            "tile_size": 32,
            "overlap": 8,
            "min_seen_pixels": 1,
            "class_aware_sampling": True,
            "class_aware_fraction": 0.7,
            "validation_fraction": 0.25,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
            "cosine_eta_min": 0.000001,
            "batch_size": 1,
            "epochs": 3,
            "learning_rate": 0.001,
            "backbone_learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "amp": False,
            "device": "cpu",
        },
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "hsi_lidar_ovseg.cli", "train", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_environment(),
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "best.pt").is_file()
    metrics = json.loads((output_dir / "test_metrics.json").read_text(encoding="utf-8"))
    state = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=True)

    assert "miou" in metrics
    assert state["scheduler_state"] is not None
    assert state["selection_state"] is not None
