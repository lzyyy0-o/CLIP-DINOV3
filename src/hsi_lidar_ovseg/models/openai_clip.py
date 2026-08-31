"""Local OpenAI CLIP ViT-B/16 guidance with trainable terminal blocks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch.nn.functional as functional
from torch import Tensor, nn

from hsi_lidar_ovseg.config import ConfigError
from hsi_lidar_ovseg.models.protocols import FeaturePyramid, Tokenizer


def load_openai_clip(checkpoint: Path) -> tuple[nn.Module, Tokenizer]:
    """Load an official CLIP checkpoint locally without downloading weights."""

    if not checkpoint.is_file():
        raise ConfigError(f"OpenAI CLIP checkpoint 不存在: {checkpoint}")
    try:
        import clip
    except ImportError as error:
        raise ConfigError("使用 OpenAI CLIP 必须安装官方 clip 包") from error
    model, _ = clip.load(str(checkpoint), device="cpu", jit=False)
    if not isinstance(model, nn.Module):
        raise ConfigError("OpenAI CLIP 必须返回 torch.nn.Module")
    return model.float(), clip.tokenize


def _resblocks(module: object, name: str) -> nn.ModuleList:
    transformer = getattr(module, "transformer", None)
    blocks = getattr(transformer, "resblocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 12:
        raise ValueError(f"{name} 必须公开 12 个 Transformer resblocks")
    return blocks


class OpenAIClipGuidance(nn.Module):
    """Provide CLIP visual pyramids and differentiable prompt-level text features."""

    out_strides = (4, 8, 16, 32)
    text_dim = 512

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        feature_blocks: tuple[int, int, int, int],
        templates: tuple[str, ...],
    ) -> None:
        super().__init__()
        if feature_blocks != (2, 5, 8, 11):
            raise ValueError("OpenAI CLIP 必须提取 block [2, 5, 8, 11]")
        if not templates or any(template.count("{}") != 1 for template in templates):
            raise ValueError("每个文本模板必须且只能包含一个 {} 占位符")
        visual = getattr(model, "visual", None)
        if not isinstance(visual, nn.Module):
            raise ValueError("OpenAI CLIP 必须公开 visual 模块")
        visual_blocks = _resblocks(visual, "视觉塔")
        _resblocks(model, "文本塔")
        width = getattr(getattr(visual, "ln_post", None), "normalized_shape", (None,))[-1]
        if not isinstance(width, int) or width <= 0:
            raise ValueError("OpenAI CLIP 视觉塔必须公开 ln_post.normalized_shape")

        self.model = model
        self.tokenizer = tokenizer
        self.templates = templates
        self.feature_blocks = feature_blocks
        self.projections = nn.ModuleList(nn.Conv2d(width, self.text_dim, 1) for _ in range(4))
        self._captured: list[Tensor] = []
        self._hooks = [
            visual_blocks[index].register_forward_hook(self._capture) for index in feature_blocks
        ]
        self.configure_partial_finetune()

    def _capture(self, _: nn.Module, __: tuple[Tensor, ...], output: Tensor) -> None:
        if not isinstance(output, Tensor):
            raise ValueError("OpenAI CLIP Transformer block 必须返回 Tensor")
        self._captured.append(output)

    def configure_partial_finetune(self) -> None:
        """Freeze CLIP then unfreeze only its terminal visual and text components."""

        self.model.requires_grad_(False)
        visual = self.model.visual
        assert isinstance(visual, nn.Module)
        for block in _resblocks(visual, "视觉塔")[-2:]:
            block.requires_grad_(True)
        for block in _resblocks(self.model, "文本塔")[-2:]:
            block.requires_grad_(True)
        for name in ("ln_post", "proj"):
            module = getattr(visual, name, None)
            if isinstance(module, nn.Module):
                module.requires_grad_(True)
            elif isinstance(module, nn.Parameter):
                module.requires_grad_(True)
        for name in ("ln_final", "text_projection"):
            module = getattr(self.model, name, None)
            if isinstance(module, nn.Module):
                module.requires_grad_(True)
            elif isinstance(module, nn.Parameter):
                module.requires_grad_(True)

    @staticmethod
    def _spatial_tokens(tokens: Tensor) -> tuple[Tensor, int]:
        if tokens.ndim != 3:
            raise ValueError("OpenAI CLIP 中间特征必须为 [L,B,C] 张量")
        token_count, batch, channels = tokens.shape
        candidate_count = token_count - 1
        spatial_count = (
            candidate_count if math.isqrt(candidate_count) ** 2 == candidate_count else token_count
        )
        side = math.isqrt(spatial_count)
        if side * side != spatial_count:
            raise ValueError("OpenAI CLIP 中间 token 数必须对应平方网格")
        start = token_count - spatial_count
        return tokens[start:].permute(1, 2, 0).reshape(batch, channels, side, side), side

    def visual_features(self, pseudo_rgb: Tensor) -> FeaturePyramid:
        if pseudo_rgb.ndim != 4 or pseudo_rgb.shape[1] != 3:
            raise ValueError("OpenAI CLIP 伪 RGB 输入必须是三通道 NCHW 张量")
        if pseudo_rgb.shape[-2:] != (224, 224):
            raise ValueError("OpenAI CLIP ViT-B/16 输入必须为 224x224")
        self._captured.clear()
        self.model.encode_image(pseudo_rgb)
        if len(self._captured) != 4:
            raise ValueError("OpenAI CLIP 未捕获四个视觉中间特征")
        outputs: list[Tensor] = []
        for captured, projection, stride in zip(
            self._captured, self.projections, self.out_strides, strict=True
        ):
            feature, _ = self._spatial_tokens(captured)
            outputs.append(
                functional.interpolate(
                    projection(feature),
                    size=(224 // stride, 224 // stride),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return tuple(outputs)  # type: ignore[return-value]

    def text_features(self, class_names: Sequence[str]) -> Tensor:
        if not class_names or any(not name.strip() for name in class_names):
            raise ValueError("class_names 必须包含非空类别名称")
        texts = [template.format(name) for template in self.templates for name in class_names]
        tokens = self.tokenizer(texts)
        if not isinstance(tokens, Tensor):
            raise TypeError("OpenAI CLIP tokenizer 必须返回 Tensor")
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            tokens = tokens.to(parameter.device)
        encoded = self.model.encode_text(tokens).float()
        if encoded.ndim != 2 or encoded.shape != (len(texts), self.text_dim):
            raise ValueError("OpenAI CLIP 文本特征必须具有 [提示词数x类别数,512] 形状")
        return functional.normalize(
            encoded.reshape(len(self.templates), len(class_names), self.text_dim).permute(1, 0, 2),
            dim=-1,
        )
