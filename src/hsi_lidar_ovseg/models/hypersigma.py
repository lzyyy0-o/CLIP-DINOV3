"""Offline adapter for an injected HyperSIGMA backbone."""

from __future__ import annotations

from torch import Tensor, nn

from hsi_lidar_ovseg.models.dinov2 import _TokenPyramidAdapter


class HyperSigmaAdapter(_TokenPyramidAdapter):
    """Convert HyperSIGMA intermediate tokens into a feature pyramid."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_blocks: tuple[int, int, int, int],
        feature_dim: int,
        frozen: bool = True,
        unfreeze_blocks: int = 0,
    ) -> None:
        if frozen and unfreeze_blocks:
            raise ValueError("frozen=true 时 unfreeze_blocks 必须为 0")
        super().__init__(
            backbone,
            feature_blocks,
            feature_dim,
            frozen=frozen,
            unfreeze_blocks=unfreeze_blocks,
        )
        has_forward = callable(getattr(backbone, "forward_intermediates", None))
        has_get = callable(getattr(backbone, "get_intermediate_layers", None))
        if not has_forward and not has_get:
            raise ValueError(
                "HyperSIGMA 主干必须实现 forward_intermediates 或 get_intermediate_layers"
            )

    def _extract(self, inputs: Tensor) -> tuple[Tensor, ...]:
        if callable(getattr(self.backbone, "forward_intermediates", None)):
            features = self.backbone.forward_intermediates(inputs, indices=self.feature_blocks)
        else:
            features = self.backbone.get_intermediate_layers(
                inputs,
                n=self.feature_blocks,
                reshape=False,
                return_class_token=False,
            )
        return tuple(features)
