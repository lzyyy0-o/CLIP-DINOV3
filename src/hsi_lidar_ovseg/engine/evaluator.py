"""Memory-bounded sliding-window scene prediction."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from hsi_lidar_ovseg.data import (
    NormalizationStats,
    SceneArrays,
    SlidingWindowAccumulator,
    fit_normalization,
    tile_origins,
)
from hsi_lidar_ovseg.data.preprocessing import normalize_scene, pseudo_rgb, terrain_channels


def _crop_and_pad(array: np.ndarray, top: int, left: int, tile_size: int) -> Tensor:
    cropped = array[top : top + tile_size, left : left + tile_size]
    pad_height = tile_size - cropped.shape[0]
    pad_width = tile_size - cropped.shape[1]
    if pad_height or pad_width:
        padding = ((0, pad_height), (0, pad_width), (0, 0))
        mode = "reflect" if cropped.shape[0] > 1 and cropped.shape[1] > 1 else "edge"
        cropped = np.pad(cropped, padding, mode=mode)
    channels_first = np.ascontiguousarray(np.moveaxis(cropped, -1, 0))
    return torch.from_numpy(channels_first).float().unsqueeze(0)


def sliding_window_predict(
    model: nn.Module,
    scene: SceneArrays,
    text_embeddings: Tensor | Sequence[str],
    tile_size: int,
    overlap: int,
    device: torch.device,
    *,
    pseudo_rgb_indices: tuple[int, int, int],
    terrain_window: int = 9,
    stats: NormalizationStats | None = None,
) -> Tensor:
    """Predict a complete registered scene and return CPU logits."""

    if isinstance(text_embeddings, Tensor):
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] <= 0:
            raise ValueError("text_embeddings 必须是非空二维张量")
        class_count = text_embeddings.shape[0]
    else:
        if not text_embeddings:
            raise ValueError("类别名称不得为空")
        class_count = len(text_embeddings)
    if stats is None:
        stats = fit_normalization(scene)
    normalized = normalize_scene(scene, stats)
    hsi = normalized.hsi
    lidar = terrain_channels(normalized.lidar, window_size=terrain_window)
    rgb = pseudo_rgb(scene.hsi, pseudo_rgb_indices, scene.train_mask)
    accumulator = SlidingWindowAccumulator(class_count, *scene.spatial_shape, tile_size)
    if isinstance(text_embeddings, Tensor):
        conditioning: Tensor | tuple[str, ...] = text_embeddings.to(device)
    else:
        conditioning = tuple(text_embeddings)
    was_training = model.training
    model.to(device).eval()
    cache_text = getattr(getattr(model, "clip_guidance", None), "cache_text_features", None)
    clear_text_cache = getattr(getattr(model, "clip_guidance", None), "clear_text_cache", None)
    try:
        with torch.inference_mode():
            if not isinstance(conditioning, Tensor) and callable(cache_text):
                cache_text(conditioning)
            for top, left in tile_origins(*scene.spatial_shape, tile_size, overlap):
                output = model(
                    _crop_and_pad(hsi, top, left, tile_size).to(device),
                    _crop_and_pad(lidar, top, left, tile_size).to(device),
                    _crop_and_pad(rgb, top, left, tile_size).to(device),
                    conditioning,
                )
                accumulator.add(output.logits[0].float().cpu(), top, left)
    finally:
        if callable(clear_text_cache):
            clear_text_cache()
        model.train(was_training)
    return accumulator.finalize()
