"""Small trainable convolutional encoders for HSI and LiDAR inputs."""

from __future__ import annotations

from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import FeaturePyramid


def _group_count(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.block(inputs))


class _EncoderStage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        kernel_size = 7 if stride == 4 else 3
        padding = 3 if stride == 4 else 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            _ResidualDepthwiseBlock(out_channels),
        )


class NativePyramidEncoder(nn.Module):
    """Four-stage encoder with no external weights or network dependency."""

    out_strides = (4, 8, 16, 32)

    def __init__(
        self, in_channels: int, channels: tuple[int, int, int, int] = (64, 96, 160, 256)
    ) -> None:
        super().__init__()
        if in_channels <= 0 or any(channel <= 0 for channel in channels):
            raise ValueError("编码器通道数必须为正整数")
        self.out_channels = channels
        input_channels = (in_channels, *channels[:-1])
        strides = (4, 2, 2, 2)
        self.stages = nn.ModuleList(
            _EncoderStage(stage_in, stage_out, stride)
            for stage_in, stage_out, stride in zip(input_channels, channels, strides, strict=True)
        )

    def forward(self, inputs: Tensor) -> FeaturePyramid:
        if inputs.ndim != 4:
            raise ValueError(f"输入必须为 NCHW 张量, 实际形状为 {tuple(inputs.shape)}")
        features: list[Tensor] = []
        output = inputs
        for stage in self.stages:
            output = stage(output)
            features.append(output)
        return tuple(features)  # type: ignore[return-value]
