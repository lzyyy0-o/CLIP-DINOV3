from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from hsi_lidar_ovseg.cli import deterministic_text_embeddings

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
    for name in ("houston2013.yaml", "trento.yaml", "muufl.yaml"):
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


def test_cli_runs_one_offline_cpu_training_epoch(tmp_path: Path) -> None:
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
            "teacher_encoder": {"kind": "native", "frozen": True},
            "clip_checkpoint": None,
            "feature_dim": 8,
            "text_dim": 10,
            "terrain_window": 3,
        },
        "loss": {
            "teacher_weight": 1.0,
            "cross_weight": 0.5,
            "gate_weight": 0.01,
            "private_weight": 0.01,
            "temperature": 0.1,
        },
        "train": {
            "tile_size": 32,
            "overlap": 8,
            "min_seen_pixels": 1,
            "batch_size": 1,
            "epochs": 1,
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
