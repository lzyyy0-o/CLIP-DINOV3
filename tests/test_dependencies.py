from __future__ import annotations

from pathlib import Path


def test_server_requirements_include_training_and_openai_clip_dependencies() -> None:
    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    )

    for dependency in (
        "torch>=2.1",
        "torchvision>=0.16",
        "numpy>=1.26",
        "scipy>=1.11",
        "PyYAML>=6.0",
        "clip @ git+https://github.com/openai/CLIP.git",
        "rs-fusion-datasets",
        "timm>=1.0",
    ):
        assert dependency in requirements
