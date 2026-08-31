"""Single-device training operations."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hsi_lidar_ovseg.losses import LossError


class Trainer:
    """Own optimization mechanics while leaving epoch orchestration to the CLI."""

    def __init__(
        self,
        model: nn.Module,
        objective: nn.Module,
        optimizer: Optimizer,
        text_embeddings: Tensor | tuple[str, ...],
        *,
        device: torch.device,
        gradient_clip: float,
        amp: bool,
        scheduler: LRScheduler | None = None,
    ) -> None:
        if gradient_clip <= 0:
            raise ValueError("gradient_clip 必须为正数")
        if isinstance(text_embeddings, Tensor) and text_embeddings.ndim != 2:
            raise ValueError("text_embeddings 必须为二维张量")
        self.model = model.to(device)
        self.objective = objective.to(device)
        self.optimizer = optimizer
        self.text_embeddings = (
            text_embeddings.detach().to(device)
            if isinstance(text_embeddings, Tensor)
            else text_embeddings
        )
        self.device = device
        self.gradient_clip = gradient_clip
        self.scheduler = scheduler
        self.use_amp = bool(amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)

    def _to_device(self, batch: Mapping[str, Tensor], key: str) -> Tensor:
        if key not in batch:
            raise KeyError(f"训练批次缺少字段: {key}")
        return batch[key].to(self.device, non_blocking=self.device.type == "cuda")

    def train_step(self, batch: Mapping[str, Tensor]) -> dict[str, float]:
        """Run one optimized batch and return detached scalar loss components."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        hsi = self._to_device(batch, "hsi")
        lidar = self._to_device(batch, "lidar")
        pseudo_rgb = self._to_device(batch, "pseudo_rgb")
        labels = self._to_device(batch, "labels")
        valid_mask = self._to_device(batch, "valid_mask")
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            output = self.model(hsi, lidar, pseudo_rgb, self.text_embeddings)
            losses = self.objective(output, labels, valid_mask)
            total = losses["total"]
        if not torch.isfinite(total):
            raise LossError("训练总损失为非有限值")
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return {name: float(value.detach().cpu()) for name, value in losses.items()}
