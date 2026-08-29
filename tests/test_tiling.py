from __future__ import annotations

import pytest
import torch

from hsi_lidar_ovseg.data.io import DataError
from hsi_lidar_ovseg.data.tiling import SlidingWindowAccumulator, tile_origins


def test_tile_origins_cover_bottom_and_right_edges() -> None:
    origins = tile_origins(13, 17, tile_size=8, overlap=2)

    assert (0, 0) in origins
    assert (5, 9) in origins
    assert max(top for top, _ in origins) + 8 == 13
    assert max(left for _, left in origins) + 8 == 17


def test_tile_origins_use_one_padded_tile_for_small_scene() -> None:
    assert tile_origins(5, 7, tile_size=8, overlap=2) == ((0, 0),)


def test_tile_origins_reject_invalid_overlap() -> None:
    with pytest.raises(DataError, match="overlap"):
        tile_origins(8, 8, tile_size=8, overlap=8)


def test_accumulator_reconstructs_constant_logits() -> None:
    accumulator = SlidingWindowAccumulator(num_classes=2, height=10, width=11, tile_size=8)
    for top, left in tile_origins(10, 11, tile_size=8, overlap=2):
        accumulator.add(torch.ones(2, 8, 8), top, left)

    torch.testing.assert_close(accumulator.finalize(), torch.ones(2, 10, 11))


def test_accumulator_crops_tile_for_small_scene() -> None:
    accumulator = SlidingWindowAccumulator(num_classes=1, height=3, width=5, tile_size=8)
    accumulator.add(torch.full((1, 8, 8), 4.0), top=0, left=0)

    assert accumulator.finalize().shape == (1, 3, 5)
    torch.testing.assert_close(accumulator.finalize(), torch.full((1, 3, 5), 4.0))


def test_accumulator_rejects_uncovered_pixels() -> None:
    accumulator = SlidingWindowAccumulator(num_classes=1, height=10, width=10, tile_size=4)
    accumulator.add(torch.ones(1, 4, 4), top=0, left=0)

    with pytest.raises(DataError, match="未覆盖"):
        accumulator.finalize()
