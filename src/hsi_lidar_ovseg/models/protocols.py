"""Structural interfaces shared by model components."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeAlias, runtime_checkable

from torch import Tensor

FeaturePyramid: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]


@runtime_checkable
class PyramidEncoder(Protocol):
    """Encoder that returns four feature maps at strides 4, 8, 16, and 32."""

    @property
    def out_channels(self) -> tuple[int, int, int, int]: ...

    @property
    def out_strides(self) -> tuple[int, int, int, int]: ...

    def __call__(self, inputs: Tensor) -> FeaturePyramid: ...


@runtime_checkable
class TextEncoder(Protocol):
    """Text prototype encoder used by the dense open-vocabulary decoder."""

    def encode(self, class_names: Sequence[str]) -> Tensor: ...


Tokenizer: TypeAlias = Callable[[list[str]], Tensor]
