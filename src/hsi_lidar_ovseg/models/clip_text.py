"""Prompt-ensemble wrapper for an injected CLIP text tower."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.models.protocols import Tokenizer


class ClipTextEncoder(nn.Module):
    """Build normalized class prototypes without loading or downloading a model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        templates: tuple[str, ...] = ("a remote sensing image of {}",),
    ) -> None:
        super().__init__()
        if not templates or any(template.count("{}") != 1 for template in templates):
            raise ValueError("每个文本模板必须且只能包含一个 {} 占位符")
        if not callable(getattr(model, "encode_text", None)):
            raise ValueError("CLIP 模型必须实现 encode_text")
        self.model = model
        self.tokenizer = tokenizer
        self.templates = templates
        self.model.requires_grad_(False)
        self.model.eval()

    def encode(self, class_names: Sequence[str]) -> Tensor:
        if not class_names or any(not name.strip() for name in class_names):
            raise ValueError("class_names 必须包含非空类别名称")
        texts = [template.format(name) for template in self.templates for name in class_names]
        tokens = self.tokenizer(texts)
        if not isinstance(tokens, Tensor):
            raise TypeError("tokenizer 必须返回 torch.Tensor")
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            tokens = tokens.to(parameter.device)
        with torch.no_grad():
            encoded = self.model.encode_text(tokens).float()
        if encoded.ndim != 2 or encoded.shape[0] != len(texts):
            raise ValueError("CLIP encode_text 返回了不兼容的形状")
        prototypes = encoded.reshape(len(self.templates), len(class_names), -1).mean(dim=0)
        return functional.normalize(prototypes, dim=-1)

    def train(self, mode: bool = True) -> ClipTextEncoder:
        super().train(mode)
        self.model.eval()
        return self
