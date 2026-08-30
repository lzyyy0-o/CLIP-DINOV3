"""Input adaptation for three-channel DINOv3 backbones."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from hsi_lidar_ovseg.models.input_adapter import ChannelAdapter


class DinoV3InputBridge(nn.Module):
    """Keep an official RGB DINOv3 backbone unchanged behind a trainable channel adapter."""

    def __init__(self, backbone: nn.Module, input_channels: int) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("DINOv3 输入通道数必须为正整数")
        self.backbone = backbone
        self.input_adapter = ChannelAdapter(input_channels, 3)

    @property
    def embed_dims(self) -> Sequence[int]:
        value = getattr(self.backbone, "embed_dims", None)
        if not isinstance(value, Sequence):
            raise ValueError("DINOv3 ConvNeXt 主干必须公开 embed_dims")
        return value

    @property
    def stages(self) -> nn.ModuleList | nn.Sequential:
        value = getattr(self.backbone, "stages", None)
        if not isinstance(value, (nn.ModuleList, nn.Sequential)):
            raise ValueError("DINOv3 ConvNeXt 主干必须公开 stages")
        return value

    @property
    def downsample_layers(self) -> nn.ModuleList | nn.Sequential:
        value = getattr(self.backbone, "downsample_layers", None)
        if not isinstance(value, (nn.ModuleList, nn.Sequential)):
            raise ValueError("DINOv3 ConvNeXt 主干必须公开 downsample_layers")
        return value

    def get_intermediate_layers(self, inputs: Tensor, **kwargs: object) -> tuple[Tensor, ...]:
        method = getattr(self.backbone, "get_intermediate_layers", None)
        if not callable(method):
            raise ValueError("DINOv3 主干必须实现 get_intermediate_layers")
        result = method(self.input_adapter(inputs), **kwargs)
        if not isinstance(result, (tuple, list)):
            raise ValueError("DINOv3 必须返回中间特征序列")
        return tuple(result)
