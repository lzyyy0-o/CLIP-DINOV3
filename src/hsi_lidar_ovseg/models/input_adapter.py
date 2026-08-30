"""Trainable input-channel bridges for fixed-channel pretrained visual backbones."""

from __future__ import annotations

from torch import Tensor, nn


class ChannelAdapter(nn.Module):
    """Map an NCHW raster from dataset channels to backbone input channels."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("输入与输出通道数必须为正整数")
        self.projection = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"通道适配器输入必须为 NCHW 张量, 实际形状为 {tuple(inputs.shape)}")
        if inputs.shape[1] != self.projection.in_channels:
            raise ValueError(
                f"通道适配器期望 {self.projection.in_channels} 个输入通道, 实际为 {inputs.shape[1]}"
            )
        return self.projection(inputs)
