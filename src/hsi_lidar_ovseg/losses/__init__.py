"""Training losses for HSI-LiDAR open-vocabulary segmentation."""

from hsi_lidar_ovseg.losses.contrastive import LossError, symmetric_info_nce
from hsi_lidar_ovseg.losses.cross_entropy import (
    ClipGuidedAlignmentObjective,
    MaskedCrossEntropyObjective,
)
from hsi_lidar_ovseg.losses.objective import OpenVocabularyObjective

__all__ = [
    "ClipGuidedAlignmentObjective",
    "LossError",
    "MaskedCrossEntropyObjective",
    "OpenVocabularyObjective",
    "symmetric_info_nce",
]
