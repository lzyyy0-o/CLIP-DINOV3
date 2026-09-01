"""Chinese command-line interface for training and evaluating experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from hsi_lidar_ovseg.config import (
    ConfigError,
    DataConfig,
    EncoderConfig,
    ExperimentConfig,
    TrainConfig,
    load_config,
)
from hsi_lidar_ovseg.data import (
    DataError,
    NormalizationStats,
    SceneArrays,
    fit_normalization,
    load_scene,
    split_training_mask,
)
from hsi_lidar_ovseg.data.datasets import PairedTileDataset
from hsi_lidar_ovseg.data.preprocessing import ChannelStats
from hsi_lidar_ovseg.engine import (
    CheckpointError,
    CheckpointIdentity,
    Trainer,
    TrainingState,
    load_checkpoint,
    save_checkpoint,
    sliding_window_predict,
)
from hsi_lidar_ovseg.losses import (
    ClipGuidedAlignmentObjective,
    LossError,
    MaskedCrossEntropyObjective,
    OpenVocabularyObjective,
)
from hsi_lidar_ovseg.metrics import SegmentationMetrics
from hsi_lidar_ovseg.models import (
    CLIPGuidedSharedLiteViTSegmentor,
    ClipTextEncoder,
    DinoV2Adapter,
    DinoV3ConvNeXtAdapter,
    DinoV3ViTAdapter,
    HSILidarOVSegmentor,
    HyperSigmaAdapter,
    NativePyramidEncoder,
    OnlineViTPyramidEncoder,
    OpenAIClipGuidance,
    RemoteClipVisionAdapter,
    SharedLiteViT,
    TextCorrelationDecoder,
    TokenPyramidProjector,
    ViTCMFEB,
    ViTMMFB,
    load_openai_clip,
)
from hsi_lidar_ovseg.models.dinov3_bridge import DinoV3InputBridge
from hsi_lidar_ovseg.models.hypersigma_bridge import HyperSigmaBridge, load_hypersigma_weights
from hsi_lidar_ovseg.vocabulary import ClassVocabulary

LOGGER = logging.getLogger(__name__)


def deterministic_text_embeddings(class_names: Sequence[str], text_dim: int) -> torch.Tensor:
    """Create repeatable normalized prototypes for explicit offline smoke experiments."""

    if not class_names or text_dim <= 0:
        raise ValueError("class_names 不得为空且 text_dim 必须为正整数")
    embeddings: list[torch.Tensor] = []
    for class_name in class_names:
        digest = hashlib.sha256(class_name.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        embeddings.append(torch.randn(text_dim, generator=generator))
    return functional.normalize(torch.stack(embeddings), dim=-1)


def _clip_vocabularies(data: DataConfig) -> tuple[ClassVocabulary, ClassVocabulary]:
    """Build MM-OVSeg-style seen training and complete evaluation vocabularies."""

    train_vocabulary = ClassVocabulary.from_all_class_names(
        data.class_names, data.seen_class_ids
    )
    test_vocabulary = ClassVocabulary.from_all_class_names(
        data.class_names, tuple(range(1, data.num_classes + 1))
    )
    return train_vocabulary, test_vocabulary


def _load_local_weights(module: nn.Module, path: Path) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise ConfigError(f"无法读取本地权重 {path}: {error}") from error
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "model_state"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping):
        raise ConfigError(f"权重文件不包含状态字典: {path}")
    state_dict = {str(key).removeprefix("module."): value for key, value in payload.items()}
    try:
        module.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ConfigError(f"本地权重与模型不兼容 {path}: {error}") from error


def _resolve_factory(path: str) -> Callable[..., nn.Module]:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        module_name, separator, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ConfigError("factory 必须使用 package.module:callable 格式")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise ConfigError(f"无法导入编码器工厂 {path}: {error}") from error
    if not callable(factory):
        raise ConfigError(f"编码器工厂不可调用: {path}")
    return factory


def _build_visual_encoder(
    config: EncoderConfig,
    *,
    in_channels: int,
    feature_dim: int,
    force_frozen: bool = False,
    expected_spectral_adapter: bool | None = None,
) -> nn.Module:
    if config.kind == "native":
        return NativePyramidEncoder(in_channels)
    if config.kind == "online_vit":
        assert config.spectral_adapter is not None
        if (
            expected_spectral_adapter is not None
            and config.spectral_adapter != expected_spectral_adapter
        ):
            role = "HSI" if expected_spectral_adapter else "LiDAR"
            required = "true" if expected_spectral_adapter else "false"
            raise ConfigError(f"{role} 在线 ViT 必须设置 spectral_adapter={required}")
        return OnlineViTPyramidEncoder(in_channels, spectral_adapter=config.spectral_adapter)
    if config.kind == "remoteclip":
        raise ConfigError("RemoteCLIP 视觉塔必须通过共享模型构建流程创建")
    assert config.factory is not None
    factory = _resolve_factory(config.factory)
    if config.kind == "hypersigma":
        assert config.source_dir is not None
        assert config.spatial_checkpoint is not None and config.spectral_checkpoint is not None
        assert config.pretrained_in_channels is not None
        kwargs = {
            "source_dir": config.source_dir,
            "model_name": config.model_name or "base",
            "in_channels": in_channels,
            "pretrained_in_channels": config.pretrained_in_channels,
            "feature_blocks": tuple(config.feature_blocks),
        }
    else:
        assert config.checkpoint is not None
        kwargs = {
            "source_dir": config.source_dir,
            "model_name": config.model_name,
            "in_channels": 3,
        }
    try:
        backbone = factory(**kwargs)
    except TypeError as error:
        raise ConfigError(f"编码器工厂参数不兼容 {config.factory}: {error}") from error
    if not isinstance(backbone, nn.Module):
        raise ConfigError(f"编码器工厂必须返回 torch.nn.Module: {config.factory}")
    if config.kind == "hypersigma":
        if not isinstance(backbone, HyperSigmaBridge):
            raise ConfigError("HyperSIGMA 工厂必须返回 HyperSigmaBridge")
        load_hypersigma_weights(backbone, config.spatial_checkpoint, config.spectral_checkpoint)
    else:
        _load_local_weights(backbone, config.checkpoint)
        if config.kind in {"dinov3_vit", "dinov3_convnext"} and in_channels != 3:
            backbone = DinoV3InputBridge(backbone, in_channels)
    frozen = force_frozen or config.frozen
    unfreeze_blocks = 0 if force_frozen else config.unfreeze_blocks
    blocks = tuple(config.feature_blocks)
    if config.kind == "hypersigma":
        return HyperSigmaAdapter(  # type: ignore[arg-type]
            backbone,
            blocks,
            feature_dim,
            frozen=frozen,
            unfreeze_blocks=unfreeze_blocks,
        )
    if config.kind == "dinov2":
        return DinoV2Adapter(  # type: ignore[arg-type]
            backbone,
            blocks,
            feature_dim,
            frozen=frozen,
            unfreeze_blocks=unfreeze_blocks,
        )
    if config.kind == "dinov3_vit":
        return DinoV3ViTAdapter(  # type: ignore[arg-type]
            backbone,
            blocks,
            feature_dim,
            frozen=frozen,
            unfreeze_blocks=unfreeze_blocks,
        )
    if config.kind == "dinov3_convnext":
        return DinoV3ConvNeXtAdapter(  # type: ignore[arg-type]
            backbone,
            feature_stages=blocks,
            frozen=frozen,
            unfreeze_blocks=unfreeze_blocks,
        )
    raise ConfigError(f"不支持的 encoder.kind: {config.kind}")


def _load_open_clip(config: ExperimentConfig) -> tuple[nn.Module, Callable[[list[str]], Any]]:
    if config.model.clip_checkpoint is None:
        raise ConfigError("加载 OpenCLIP 时 clip_checkpoint 不得为空")
    try:
        import open_clip
    except ImportError as error:
        raise ConfigError("使用 CLIP 权重时必须安装 pretrained 可选依赖") from error
    model = open_clip.create_model(config.model.clip_model_name, pretrained=None)
    if not isinstance(model, nn.Module):
        raise ConfigError("open_clip.create_model 必须返回 torch.nn.Module")
    _load_local_weights(model, config.model.clip_checkpoint)
    tokenizer = open_clip.get_tokenizer(config.model.clip_model_name)
    if not callable(tokenizer):
        raise ConfigError("OpenCLIP tokenizer 不可调用")
    return model, tokenizer


def _encode_text_embeddings(
    config: ExperimentConfig,
    model: nn.Module,
    tokenizer: Callable[[list[str]], Any],
) -> torch.Tensor:
    encoder = ClipTextEncoder(model, tokenizer, config.model.prompt_templates)
    embeddings = encoder.encode(config.data.class_names)
    if embeddings.shape[1] != config.model.text_dim:
        raise ConfigError(
            f"CLIP 文本维度为 {embeddings.shape[1]}, 但 model.text_dim={config.model.text_dim}"
        )
    return embeddings.cpu()


def _build_text_embeddings(config: ExperimentConfig) -> torch.Tensor:
    if config.model.clip_checkpoint is None:
        LOGGER.warning(
            "clip_checkpoint=null: 使用确定性哈希文本原型; "
            "仅适合离线冒烟和消融, 不代表 CLIP 开放词汇能力"
        )
        return deterministic_text_embeddings(config.data.class_names, config.model.text_dim)
    model, tokenizer = _load_open_clip(config)
    return _encode_text_embeddings(config, model, tokenizer)


def _build_model_and_text(
    config: ExperimentConfig, hsi_bands: int
) -> tuple[nn.Module, torch.Tensor | tuple[str, ...]]:
    """Build all visual towers and text prototypes without duplicating RemoteCLIP."""

    if config.model.architecture == "clip_guided_shared_lite_vit":
        assert config.model.clip is not None
        clip_model, tokenizer = load_openai_clip(config.model.clip.checkpoint)
        return (
            CLIPGuidedSharedLiteViTSegmentor(
                SharedLiteViT(hsi_bands, 3),
                ViTMMFB(),
                ViTCMFEB(),
                TokenPyramidProjector(),
                OpenAIClipGuidance(
                    clip_model,
                    tokenizer,
                    config.model.clip.feature_blocks,
                    config.model.prompt_templates,
                    unfreeze_blocks=config.model.clip.unfreeze_blocks,
                ),
                TextCorrelationDecoder(),
            ),
            config.data.class_names,
        )
    assert config.model.hsi_encoder is not None
    assert config.model.lidar_encoder is not None
    assert config.model.structure_teacher_encoder is not None
    assert config.model.semantic_teacher_encoder is not None
    hsi_encoder = _build_visual_encoder(
        config.model.hsi_encoder,
        in_channels=hsi_bands,
        feature_dim=config.model.feature_dim,
        expected_spectral_adapter=True,
    )
    lidar_encoder = _build_visual_encoder(
        config.model.lidar_encoder,
        in_channels=3,
        feature_dim=config.model.feature_dim,
        expected_spectral_adapter=False,
    )
    structure_teacher_encoder = _build_visual_encoder(
        config.model.structure_teacher_encoder,
        in_channels=3,
        feature_dim=config.model.feature_dim,
        force_frozen=True,
    )
    semantic_config = config.model.semantic_teacher_encoder
    if semantic_config.kind == "remoteclip":
        remoteclip, tokenizer = _load_open_clip(config)
        text_embeddings = _encode_text_embeddings(config, remoteclip, tokenizer)
        visual = getattr(remoteclip, "visual", None)
        if not isinstance(visual, nn.Module):
            raise ConfigError("RemoteCLIP 模型必须公开 torch.nn.Module 类型的 visual 视觉塔")
        semantic_teacher_encoder = RemoteClipVisionAdapter(
            visual,
            tuple(semantic_config.feature_blocks),  # type: ignore[arg-type]
            config.model.feature_dim,
            frozen=True,
        )
    else:
        semantic_teacher_encoder = _build_visual_encoder(
            semantic_config,
            in_channels=3,
            feature_dim=config.model.feature_dim,
            force_frozen=True,
        )
        text_embeddings = _build_text_embeddings(config)
    model = HSILidarOVSegmentor(
        hsi_encoder=hsi_encoder,
        lidar_encoder=lidar_encoder,
        structure_teacher_encoder=structure_teacher_encoder,
        semantic_teacher_encoder=semantic_teacher_encoder,
        feature_dim=config.model.feature_dim,
        text_dim=config.model.text_dim,
        freeze_teachers=True,
    )
    return model, text_embeddings


def _device(config: ExperimentConfig) -> torch.device:
    if config.train.device == "cuda" and not torch.cuda.is_available():
        raise ConfigError("配置请求 CUDA, 但当前 PyTorch 无法使用 CUDA")
    return torch.device(config.train.device)


def _optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.AdamW:
    backbone: list[nn.Parameter] = []
    heads: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if config.model.architecture == "clip_guided_shared_lite_vit":
            (backbone if name.startswith("clip_guidance.model.") else heads).append(parameter)
        elif "encoder" in name or ".backbone." in name:
            backbone.append(parameter)
        else:
            heads.append(parameter)
    groups: list[dict[str, Any]] = []
    if heads:
        groups.append({"params": heads, "lr": config.train.learning_rate})
    if backbone:
        groups.append({"params": backbone, "lr": config.train.backbone_learning_rate})
    if not groups:
        raise ConfigError("模型没有可训练参数")
    return torch.optim.AdamW(groups, weight_decay=config.train.weight_decay)


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig | TrainConfig,
    steps_per_epoch: int,
) -> CosineAnnealingLR:
    if steps_per_epoch <= 0:
        raise ConfigError("训练 DataLoader 必须至少包含一个批次")
    train = config.train if isinstance(config, ExperimentConfig) else config
    return CosineAnnealingLR(
        optimizer,
        T_max=train.epochs * steps_per_epoch,
        eta_min=train.cosine_eta_min,
    )


def _identity(config: ExperimentConfig, hsi_bands: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        class_names=config.data.class_names,
        seen_class_ids=config.data.seen_class_ids,
        unseen_class_ids=config.data.unseen_class_ids,
        hsi_bands=hsi_bands,
        lidar_channels=3,
        feature_dim=config.model.feature_dim,
        text_dim=config.model.text_dim,
        architecture=config.model.architecture,
        clip_model_name=(
            config.model.clip.model_name
            if config.model.architecture == "clip_guided_shared_lite_vit"
            and config.model.clip is not None
            else None
        ),
    )


def _normalization_payload(stats: NormalizationStats) -> dict[str, torch.Tensor]:
    return {
        "hsi_mean": torch.from_numpy(stats.hsi.mean.copy()),
        "hsi_scale": torch.from_numpy(stats.hsi.scale.copy()),
        "lidar_mean": torch.from_numpy(stats.lidar.mean.copy()),
        "lidar_scale": torch.from_numpy(stats.lidar.scale.copy()),
    }


def _normalization_from_payload(payload: Mapping[str, torch.Tensor]) -> NormalizationStats:
    required = {"hsi_mean", "hsi_scale", "lidar_mean", "lidar_scale"}
    if set(payload) != required:
        raise CheckpointError(f"归一化统计字段不完整: {sorted(set(payload))}")
    return NormalizationStats(
        hsi=ChannelStats(payload["hsi_mean"].cpu().numpy(), payload["hsi_scale"].cpu().numpy()),
        lidar=ChannelStats(
            payload["lidar_mean"].cpu().numpy(), payload["lidar_scale"].cpu().numpy()
        ),
    )


def _primitive(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _primitive(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


def _metrics(
    config: ExperimentConfig,
    logits: torch.Tensor,
    labels: np.ndarray,
    mask: np.ndarray,
    vocabulary: ClassVocabulary | None = None,
) -> dict[str, Any]:
    predictions = (
        vocabulary.decode_logits(logits)
        if vocabulary is not None
        else logits.argmax(dim=0).to(torch.int64) + 1
    )
    ground_truth = torch.from_numpy(np.where(mask, labels, 0).astype(np.int64, copy=False))
    metrics = SegmentationMetrics(
        config.data.num_classes,
        config.data.seen_class_ids,
        config.data.unseen_class_ids,
    )
    metrics.update(ground_truth, predictions)
    return metrics.compute()


def _training_scene(scene: SceneArrays, config: ExperimentConfig) -> tuple[SceneArrays, np.ndarray]:
    training_mask, validation_mask = split_training_mask(
        scene.labels,
        scene.train_mask,
        config.data.seen_class_ids,
        config.train.validation_fraction,
        config.seed,
    )
    if not np.any(validation_mask):
        raise DataError("验证划分不包含任何已见类像素; 请检查 train_mask 或 validation_fraction")
    return (
        SceneArrays(
            hsi=scene.hsi,
            lidar=scene.lidar,
            labels=scene.labels,
            train_mask=training_mask,
            test_mask=scene.test_mask,
        ),
        validation_mask,
    )


def _validate_command(args: argparse.Namespace) -> int:
    config = load_config(args.config, check_files=not args.skip_file_checks)
    LOGGER.info("配置有效: %s (%s)", config.name, config.data.name)
    return 0


def _train_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device = _device(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    scene = load_scene(config.data, split_seed=config.seed)
    training_scene, validation_mask = _training_scene(scene, config)
    stats = fit_normalization(training_scene)
    model, text_embeddings = _build_model_and_text(config, scene.hsi.shape[-1])
    if config.model.architecture == "clip_guided_shared_lite_vit":
        train_vocabulary, test_vocabulary = _clip_vocabularies(config.data)
        training_conditioning: torch.Tensor | tuple[str, ...] = train_vocabulary.class_names
        test_conditioning: torch.Tensor | tuple[str, ...] = test_vocabulary.class_names
    else:
        train_vocabulary = test_vocabulary = None
        training_conditioning = test_conditioning = text_embeddings
    optimizer = _optimizer(model, config)
    if config.model.architecture == "clip_guided_shared_lite_vit":
        objective = (
            ClipGuidedAlignmentObjective(
                config.data.seen_class_ids, config.loss.clip_alignment_weight
            )
            if config.loss.kind == "clip_guided_alignment"
            else MaskedCrossEntropyObjective(config.data.seen_class_ids)
        )
    else:
        objective = OpenVocabularyObjective(config.loss, config.data.seen_class_ids)
    dataset = PairedTileDataset(
        training_scene,
        stats,
        pseudo_rgb_indices=config.data.pseudo_rgb_indices,
        tile_size=config.train.tile_size,
        min_seen_pixels=config.train.min_seen_pixels,
        seen_ids=config.data.seen_class_ids,
        training=True,
        seed=config.seed,
        terrain_window=config.model.terrain_window,
        class_aware_sampling=config.train.class_aware_sampling,
        class_aware_fraction=config.train.class_aware_fraction,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        generator=generator,
    )
    scheduler = _cosine_scheduler(optimizer, config, len(loader))
    trainer = Trainer(
        model,
        objective,
        optimizer,
        training_conditioning,
        device=device,
        gradient_clip=config.train.gradient_clip,
        amp=config.train.amp,
        scheduler=scheduler,
    )
    identity = _identity(config, scene.hsi.shape[-1])
    start_epoch = 0
    global_step = 0
    best_score = float("-inf")
    epochs_without_improvement = 0
    if args.resume is not None:
        restored = load_checkpoint(
            args.resume,
            trainer.model,
            trainer.optimizer,
            identity,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
        )
        stats = _normalization_from_payload(restored.normalization)
        start_epoch = restored.epoch
        global_step = restored.global_step
        if restored.selection_state is not None:
            best_score = float(restored.selection_state["best_score"])
            epochs_without_improvement = int(restored.selection_state["epochs_without_improvement"])
        LOGGER.info("从第 %d 轮、第 %d 步恢复", start_epoch, global_step)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, config.train.epochs):
        dataset.set_epoch(epoch)
        epoch_losses: list[float] = []
        for batch in loader:
            losses = trainer.train_step(batch)
            epoch_losses.append(losses["total"])
            global_step += 1
        validation_logits = sliding_window_predict(
            trainer.model,
            training_scene,
            training_conditioning,
            config.train.tile_size,
            config.train.overlap,
            device,
            pseudo_rgb_indices=config.data.pseudo_rgb_indices,
            terrain_window=config.model.terrain_window,
            stats=stats,
        )
        validation_metrics = _metrics(
            config, validation_logits, scene.labels, validation_mask, train_vocabulary
        )
        score = float(validation_metrics["seen_miou"])
        improved = score > best_score + config.train.early_stopping_min_delta
        if improved:
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        mean_loss = float(np.mean(epoch_losses))
        LOGGER.info(
            "轮次 %d/%d: loss=%.6f, validation=%s",
            epoch + 1,
            config.train.epochs,
            mean_loss,
            json.dumps(validation_metrics, ensure_ascii=False),
        )
        state = TrainingState(
            identity=identity,
            model_state=trainer.model.state_dict(),
            optimizer_state=trainer.optimizer.state_dict(),
            scheduler_state=trainer.scheduler.state_dict()
            if trainer.scheduler is not None
            else None,
            scaler_state=trainer.scaler.state_dict(),
            epoch=epoch + 1,
            global_step=global_step,
            normalization=_normalization_payload(stats),
            config=_primitive(config),
            selection_state={
                "best_score": best_score,
                "epochs_without_improvement": epochs_without_improvement,
            },
        )
        save_checkpoint(config.output_dir / "last.pt", state)
        if improved:
            save_checkpoint(config.output_dir / "best.pt", state)
        if epochs_without_improvement >= config.train.early_stopping_patience:
            LOGGER.info("验证指标连续 %d 轮未改善, 提前停止", epochs_without_improvement)
            break

    best_state = load_checkpoint(
        config.output_dir / "best.pt",
        trainer.model,
        trainer.optimizer,
        identity,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
    )
    stats = _normalization_from_payload(best_state.normalization)
    test_logits = sliding_window_predict(
        trainer.model,
        training_scene,
        test_conditioning,
        config.train.tile_size,
        config.train.overlap,
        device,
        pseudo_rgb_indices=config.data.pseudo_rgb_indices,
        terrain_window=config.model.terrain_window,
        stats=stats,
    )
    test_metrics = _metrics(config, test_logits, scene.labels, scene.test_mask, test_vocabulary)
    (config.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("最终测试指标: %s", json.dumps(test_metrics, ensure_ascii=False))
    if test_vocabulary is not None:
        predictions = test_vocabulary.decode_logits(test_logits)
        LOGGER.info("最终预测类别像素数: %s", test_vocabulary.prediction_counts(predictions))
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device = _device(config)
    scene = load_scene(config.data, split_seed=config.seed)
    training_scene, _ = _training_scene(scene, config)
    model, text_embeddings = _build_model_and_text(config, scene.hsi.shape[-1])
    if config.model.architecture == "clip_guided_shared_lite_vit":
        _, test_vocabulary = _clip_vocabularies(config.data)
        conditioning: torch.Tensor | tuple[str, ...] = test_vocabulary.class_names
    else:
        test_vocabulary = None
        conditioning = text_embeddings
    optimizer = _optimizer(model, config)
    state = load_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        _identity(config, scene.hsi.shape[-1]),
    )
    stats = _normalization_from_payload(state.normalization)
    logits = sliding_window_predict(
        model,
        training_scene,
        conditioning,
        config.train.tile_size,
        config.train.overlap,
        device,
        pseudo_rgb_indices=config.data.pseudo_rgb_indices,
        terrain_window=config.model.terrain_window,
        stats=stats,
    )
    metrics = _metrics(config, logits, scene.labels, scene.test_mask, test_vocabulary)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (
        test_vocabulary.decode_logits(logits)
        if test_vocabulary is not None
        else logits.argmax(dim=0).to(torch.int64) + 1
    )
    np.save(config.output_dir / "predictions.npy", predictions.numpy())
    (config.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("评估完成: %s", json.dumps(metrics, ensure_ascii=False))
    if test_vocabulary is not None:
        LOGGER.info("预测类别像素数: %s", test_vocabulary.prediction_counts(predictions))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HSI-LiDAR 开放词汇语义分割")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="校验实验 YAML")
    validate.add_argument("config", type=Path)
    validate.add_argument("--skip-file-checks", action="store_true", help="跳过本地文件存在性检查")
    validate.set_defaults(handler=_validate_command)

    train = subparsers.add_parser("train", help="训练一个实验")
    train.add_argument("config", type=Path)
    train.add_argument("--resume", type=Path, default=None, help="从兼容检查点恢复")
    train.set_defaults(handler=_train_command)

    evaluate = subparsers.add_parser("evaluate", help="整图滑窗评估")
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.set_defaults(handler=_evaluate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process status code."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (CheckpointError, ConfigError, DataError, ImportError, LossError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
