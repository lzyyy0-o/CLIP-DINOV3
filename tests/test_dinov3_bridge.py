from __future__ import annotations

import torch
from torch import Tensor, nn

from hsi_lidar_ovseg.models.dinov3_bridge import DinoV3InputBridge


class _FakeConvNeXt(nn.Module):
    embed_dims = (4, 8, 12, 16)

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList(nn.Identity() for _ in range(4))
        self.downsample_layers = nn.ModuleList(nn.Identity() for _ in range(4))

    def get_intermediate_layers(
        self, inputs: Tensor, *, n: tuple[int, ...], **_: object
    ) -> tuple[Tensor, ...]:
        assert inputs.shape[1] == 3
        return tuple(torch.ones(inputs.shape[0], self.embed_dims[index], 2, 2) for index in n)


def test_dinov3_input_bridge_keeps_backbone_rgb_and_adapts_lidar() -> None:
    backbone = _FakeConvNeXt()
    bridge = DinoV3InputBridge(backbone, input_channels=1)

    outputs = bridge.get_intermediate_layers(torch.randn(2, 1, 64, 64), n=(0, 1, 2, 3))

    assert bridge.input_adapter.projection.out_channels == 3
    assert len(outputs) == 4
    assert bridge.embed_dims == backbone.embed_dims
