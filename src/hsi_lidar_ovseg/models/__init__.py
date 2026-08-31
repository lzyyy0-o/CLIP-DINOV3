"""Model components for multimodal open-vocabulary segmentation."""

from hsi_lidar_ovseg.models.clip_text import ClipTextEncoder
from hsi_lidar_ovseg.models.dinov2 import DinoV2Adapter
from hsi_lidar_ovseg.models.dinov3 import DinoV3ConvNeXtAdapter, DinoV3ViTAdapter
from hsi_lidar_ovseg.models.hypersigma import HyperSigmaAdapter
from hsi_lidar_ovseg.models.input_adapter import ChannelAdapter
from hsi_lidar_ovseg.models.model import (
    HSILidarOVSegmentor,
    SegmentationOutput,
    make_native_model,
)
from hsi_lidar_ovseg.models.native import NativePyramidEncoder
from hsi_lidar_ovseg.models.online_vit import OnlineViTPyramidEncoder
from hsi_lidar_ovseg.models.protocols import FeaturePyramid, PyramidEncoder, TextEncoder
from hsi_lidar_ovseg.models.remoteclip import RemoteClipVisionAdapter

__all__ = [
    "ChannelAdapter",
    "ClipTextEncoder",
    "DinoV2Adapter",
    "DinoV3ConvNeXtAdapter",
    "DinoV3ViTAdapter",
    "FeaturePyramid",
    "HSILidarOVSegmentor",
    "HyperSigmaAdapter",
    "NativePyramidEncoder",
    "OnlineViTPyramidEncoder",
    "PyramidEncoder",
    "RemoteClipVisionAdapter",
    "SegmentationOutput",
    "TextEncoder",
    "make_native_model",
]
