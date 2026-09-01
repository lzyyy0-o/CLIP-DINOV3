"""Scene loading, preprocessing, and paired raster datasets."""

from hsi_lidar_ovseg.data.io import DataError, SceneArrays, load_scene
from hsi_lidar_ovseg.data.preprocessing import NormalizationStats, fit_normalization
from hsi_lidar_ovseg.data.rs_fusion import load_rs_fusion_scene, stratified_rs_split
from hsi_lidar_ovseg.data.splits import split_training_mask
from hsi_lidar_ovseg.data.tiling import SlidingWindowAccumulator, tile_origins

__all__ = [
    "DataError",
    "NormalizationStats",
    "SceneArrays",
    "SlidingWindowAccumulator",
    "fit_normalization",
    "load_rs_fusion_scene",
    "load_scene",
    "split_training_mask",
    "stratified_rs_split",
    "tile_origins",
]
