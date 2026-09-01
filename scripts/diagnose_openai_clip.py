"""Verify that the local OpenAI CLIP weight matches this project's ViT-B/16 contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hsi_lidar_ovseg.models.openai_clip import OpenAIClipGuidance, load_openai_clip


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 OpenAI CLIP ViT-B/16 权重与项目接口")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("weights/openai_clip/ViT-B-16.pt"),
        help="本地 OpenAI CLIP 权重路径",
    )
    args = parser.parse_args()

    model, tokenizer = load_openai_clip(args.checkpoint)
    visual_blocks = getattr(getattr(model.visual, "transformer", None), "resblocks", None)
    text_blocks = getattr(getattr(model, "transformer", None), "resblocks", None)

    print(f"项目代码: {__import__('hsi_lidar_ovseg').__file__}")
    print(f"权重路径: {args.checkpoint.resolve()}")
    print(f"视觉 block 类型: {type(visual_blocks)}")
    print(f"视觉 block 为 ModuleList: {isinstance(visual_blocks, torch.nn.ModuleList)}")
    print(f"视觉 block 数: {len(visual_blocks) if visual_blocks is not None else '缺失'}")
    print(f"文本 block 数: {len(text_blocks) if text_blocks is not None else '缺失'}")
    print(
        "视觉 patch 大小: "
        f"{getattr(getattr(model.visual, 'conv1', None), 'kernel_size', '缺失')}"
    )

    OpenAIClipGuidance(
        model,
        tokenizer,
        (2, 5, 8, 11),
        ("aerial image of {}",),
    )
    print("OpenAIClipGuidance 初始化成功")


if __name__ == "__main__":
    main()
