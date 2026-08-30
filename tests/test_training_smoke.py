from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hsi_lidar_ovseg.cli import _cosine_scheduler
from hsi_lidar_ovseg.config import LossConfig, TrainConfig
from hsi_lidar_ovseg.data import SceneArrays
from hsi_lidar_ovseg.engine import Trainer, sliding_window_predict
from hsi_lidar_ovseg.losses import OpenVocabularyObjective
from hsi_lidar_ovseg.models import SegmentationOutput, make_native_model


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
