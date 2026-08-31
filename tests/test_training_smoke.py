from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hsi_lidar_ovseg.cli import _cosine_scheduler
from hsi_lidar_ovseg.config import LossConfig, TrainConfig
from hsi_lidar_ovseg.data import SceneArrays
from hsi_lidar_ovseg.engine import Trainer, sliding_window_predict
from hsi_lidar_ovseg.losses import MaskedCrossEntropyObjective, OpenVocabularyObjective
from hsi_lidar_ovseg.models import (
    ClipGuidedSegmentationOutput,
    SegmentationOutput,
    make_native_model,
)


class ConstantSegmentor(nn.Module):
    def forward(
        self,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
        pseudo_rgb: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> SegmentationOutput:
        del lidar, pseudo_rgb
        batch, _, height, width = hsi.shape
        logits = torch.ones(batch, text_embeddings.shape[0], height, width, device=hsi.device)
        embeddings = torch.ones(batch, text_embeddings.shape[1], height, width, device=hsi.device)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return SegmentationOutput(logits, embeddings, {}, ())


class ClassNameSegmentor(nn.Module):
    def forward(
        self,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
        pseudo_rgb: torch.Tensor,
        class_names: tuple[str, ...],
    ) -> SegmentationOutput:
        del lidar, pseudo_rgb
        batch, _, height, width = hsi.shape
        logits = torch.ones(batch, len(class_names), height, width, device=hsi.device)
        embeddings = torch.ones(batch, 4, height, width, device=hsi.device)
        return SegmentationOutput(logits, embeddings, {}, ())


class TextRecomputingSegmentor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.text_forward_calls = 0

    def forward(
        self,
        hsi: torch.Tensor,
        lidar: torch.Tensor,
        pseudo_rgb: torch.Tensor,
        class_names: tuple[str, ...],
    ) -> ClipGuidedSegmentationOutput:
        del lidar, pseudo_rgb
        self.text_forward_calls += 1
        batch, _, height, width = hsi.shape
        logits = self.scale * torch.ones(
            batch, len(class_names), height, width, device=hsi.device
        )
        return ClipGuidedSegmentationOutput(logits)


def test_cosine_scheduler_uses_total_training_steps() -> None:
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)

    scheduler = _cosine_scheduler(
        optimizer,
        TrainConfig(epochs=3, cosine_eta_min=1e-6),
        steps_per_epoch=5,
    )

    assert scheduler.T_max == 15
    assert scheduler.eta_min == 1e-6


def test_cpu_training_step_updates_trainable_parameter() -> None:
    model = make_native_model(hsi_bands=6, lidar_channels=3, feature_dim=8, text_dim=10)
    objective = OpenVocabularyObjective(LossConfig(), seen_class_ids=(1, 2))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3
    )
    trainer = Trainer(
        model,
        objective,
        optimizer,
        torch.nn.functional.normalize(torch.randn(3, 10), dim=-1),
        device=torch.device("cpu"),
        gradient_clip=1.0,
        amp=False,
    )
    labels = torch.ones(1, 32, 32, dtype=torch.long)
    labels[:, :, 16:] = 2
    batch = {
        "hsi": torch.randn(1, 6, 32, 32),
        "lidar": torch.randn(1, 3, 32, 32),
        "pseudo_rgb": torch.rand(1, 3, 32, 32),
        "labels": labels,
        "valid_mask": torch.ones(1, 32, 32, dtype=torch.bool),
    }
    before = model.decoder.output.weight.detach().clone()

    losses = trainer.train_step(batch)

    assert np.isfinite(losses["total"])
    assert not torch.equal(before, model.decoder.output.weight.detach())


def test_sliding_window_predict_reconstructs_full_scene() -> None:
    generator = np.random.default_rng(3)
    scene = SceneArrays(
        hsi=generator.normal(size=(13, 15, 5)).astype(np.float32),
        lidar=generator.normal(size=(13, 15, 1)).astype(np.float32),
        labels=np.ones((13, 15), dtype=np.int64),
        train_mask=np.ones((13, 15), dtype=np.bool_),
        test_mask=np.ones((13, 15), dtype=np.bool_),
    )

    logits = sliding_window_predict(
        ConstantSegmentor(),
        scene,
        torch.randn(3, 7),
        tile_size=8,
        overlap=2,
        device=torch.device("cpu"),
        pseudo_rgb_indices=(0, 1, 2),
        terrain_window=3,
    )

    assert logits.shape == (3, 13, 15)
    torch.testing.assert_close(logits, torch.ones_like(logits))


def test_sliding_window_predict_accepts_dynamic_class_names() -> None:
    scene = SceneArrays(
        hsi=np.ones((8, 8, 3), dtype=np.float32),
        lidar=np.ones((8, 8, 1), dtype=np.float32),
        labels=np.ones((8, 8), dtype=np.int64),
        train_mask=np.ones((8, 8), dtype=np.bool_),
        test_mask=np.ones((8, 8), dtype=np.bool_),
    )

    logits = sliding_window_predict(
        ClassNameSegmentor(),
        scene,
        ("trees", "roads"),
        tile_size=8,
        overlap=2,
        device=torch.device("cpu"),
        pseudo_rgb_indices=(0, 1, 2),
        terrain_window=3,
    )

    assert logits.shape == (2, 8, 8)


def test_trainer_recomputes_dynamic_text_conditioning_every_step() -> None:
    model = TextRecomputingSegmentor()
    trainer = Trainer(
        model,
        MaskedCrossEntropyObjective((1, 2)),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        ("trees", "roads", "water"),
        device=torch.device("cpu"),
        gradient_clip=1.0,
        amp=False,
    )
    batch = {
        "hsi": torch.randn(1, 6, 8, 8),
        "lidar": torch.randn(1, 3, 8, 8),
        "pseudo_rgb": torch.rand(1, 3, 8, 8),
        "labels": torch.ones(1, 8, 8, dtype=torch.long),
        "valid_mask": torch.ones(1, 8, 8, dtype=torch.bool),
    }

    trainer.train_step(batch)
    trainer.train_step(batch)

    assert model.text_forward_calls == 2
