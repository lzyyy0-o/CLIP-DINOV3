# HSI–LiDAR Open-Vocabulary Segmentation Design

## 1. Objective

Build a standalone PyTorch project for open-vocabulary semantic segmentation of the co-registered hyperspectral and LiDAR rasters in Houston 2013, Trento, and MUUFL. The system shall preserve HSI spectral information, preserve LiDAR geometric information, align only their shared semantic representations, and classify dense pixels by similarity to frozen CLIP text prototypes.

The project must train and evaluate without depending on Detectron2. Pretrained HyperSIGMA, DINOv2, and CLIP weights are supplied as local paths; the training and test commands must never download weights implicitly.

## 2. Scope

The first release includes:

- dataset adapters for `.mat`, `.npy`, and `.npz` scene files;
- dataset specifications for Houston 2013, Trento, and MUUFL;
- leakage-free normalization, paired cropping, augmentation, and sliding-window inference;
- a visible-band pseudo-RGB semantic teacher;
- an HSI encoder interface with a HyperSIGMA adapter and a self-contained lightweight spectral-spatial encoder for offline tests and ablations;
- a LiDAR terrain adapter followed by a DINOv2 encoder interface;
- multi-level semantic alignment, gated HSI–LiDAR fusion, and a dense CLIP-space decoder;
- supervised, contrastive, and regularization losses;
- open-vocabulary seen/unseen class splits configured explicitly per experiment;
- training, evaluation, checkpointing, and metric reporting commands;
- unit and integration tests that use synthetic tensors and do not require pretrained weights or network access.

The release does not include dataset files, pretrained checkpoints, automatic model downloads, point-cloud processing, distributed training, or reproduction claims for unpublished results.

## 3. Considered Approaches

### 3.1 Standalone PyTorch project — selected

Use a small, typed package with explicit interfaces for data readers, encoders, fusion, losses, and training. This keeps the scientific components testable, avoids Detectron2 version coupling, and permits pretrained models to be replaced independently.

### 3.2 MM-OVSeg/Detectron2 fork — rejected

This would reuse the original training shell and decoder structure, but it would bring large framework dependencies and RGB-centric dataset assumptions into a project whose input contract is fundamentally different.

### 3.3 Lightweight closed-set classifier — rejected

A patch classifier would be simpler and useful as a baseline, but it would not produce full-resolution segmentation maps or exercise the CLIP text space required by the stated objective.

## 4. System Architecture

For an HSI tensor `H` with shape `[N, B, H, W]` and a co-registered LiDAR tensor `L` with shape `[N, C_l, H, W]`, the model uses three visual paths:

1. **Semantic teacher:** dataset-configured visible wavelengths are selected from `H`, percentile-stretched into pseudo-RGB, and passed through a frozen DINOv2 encoder. This path supplies a modality-stable visual semantic target.
2. **HSI student:** the complete HSI cube is passed through HyperSIGMA. A lightweight native spectral-spatial encoder implements the same feature-pyramid protocol for tests, ablations, and installations without HyperSIGMA.
3. **LiDAR student:** the LiDAR raster is converted to three terrain channels—normalized height, local relative height, and slope magnitude—and passed through DINOv2.

Each encoder returns four feature maps at strides 4, 8, 16, and 32. Adapter projections map every level to `feature_dim=256`. If a ViT exposes equal-resolution intermediate tokens, the adapter reshapes tokens to a spatial grid and constructs the required pyramid with learned resampling blocks.

The model aligns projected HSI and LiDAR features with the frozen teacher at middle and high levels. A lower-weight symmetric HSI–LiDAR alignment term encourages common semantics without forcing equality of modality-private features. Per-level gates fuse the two modalities:

`F_k = sigmoid(G_k([H_k, L_k])) * H_k + (1 - sigmoid(G_k([H_k, L_k]))) * L_k`.

An FPN-style decoder upsamples and combines fused levels into a dense embedding map. The map is L2-normalized and compared with L2-normalized CLIP text prototypes. The similarity temperature is a bounded learnable scalar. Text prototypes are averages over configured prompt templates and are cached for an evaluation run.

## 5. Stable Interfaces

The implementation exposes the following typed contracts:

```python
@dataclass(frozen=True)
class SceneArrays:
    hsi: np.ndarray          # [height, width, bands], float32
    lidar: np.ndarray        # [height, width, channels], float32
    labels: np.ndarray       # [height, width], int64; 0 is ignored
    train_mask: np.ndarray   # [height, width], bool
    test_mask: np.ndarray    # [height, width], bool

class PyramidEncoder(Protocol):
    out_channels: tuple[int, int, int, int]
    out_strides: tuple[int, int, int, int]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]: ...

@dataclass
class SegmentationOutput:
    logits: torch.Tensor
    pixel_embeddings: torch.Tensor
    alignment_features: dict[str, tuple[torch.Tensor, ...]]
    gates: tuple[torch.Tensor, ...]
```

Dataset-specific file names, array keys, band wavelengths or pseudo-RGB indices, class names, class IDs, and seen/unseen splits belong in YAML configuration. Model code must not contain dataset-specific paths or MATLAB keys.

## 6. Pretrained Backbone Integration

### 6.1 HyperSIGMA

`HyperSigmaAdapter` accepts an already constructed `torch.nn.Module` or constructs the official model through a configured Python factory path. A local checkpoint is mandatory when `encoder.kind=hypersigma`. State-dict loading reports missing and unexpected keys and rejects incompatible patch embeddings. The adapter extracts configured intermediate blocks and converts them to the common feature pyramid.

### 6.2 DINOv2

`DinoV2Adapter` builds a configured local DINOv2 implementation and loads a local checkpoint. The semantic-teacher instance is always frozen and remains in evaluation mode. The LiDAR instance starts frozen except for the terrain adapter and projection heads; staged unfreezing of its final blocks is controlled by configuration.

### 6.3 CLIP

`ClipTextEncoder` loads a local OpenCLIP-compatible checkpoint and tokenizer assets. It returns normalized text embeddings for class names and templates. Tokenization and text encoding are isolated behind an interface so tests can inject deterministic embeddings.

No pretrained adapter may fall back silently to random initialization. The self-contained native encoders are selected only by an explicit `kind=native` configuration.

## 7. Dataset and Preprocessing Contract

All three benchmarks are treated as co-registered raster scenes. Array orientation is normalized once at load time to channel-last NumPy arrays and converted to channel-first tensors only in the dataset.

- HSI non-finite values are rejected before statistics are computed.
- Per-band mean and standard deviation use training-mask pixels only and are stored with the run artifacts.
- LiDAR is normalized from training-mask pixels with robust median and interquartile scale; a small epsilon protects constant rasters.
- Local relative height uses a configurable odd window and reflection padding.
- Slope is the magnitude of centered finite differences in normalized height.
- Pseudo-RGB uses configured band indices or nearest configured wavelengths and independent 2nd–98th percentile stretching from training pixels.
- Label `0` is ignored. Positive labels must be contiguous after an explicit remapping defined by the dataset configuration.
- Spatial augmentations are sampled once and applied identically to HSI, LiDAR, labels, and masks. Spectral jitter applies only to HSI; height jitter applies only to LiDAR.

Training samples are paired tiles of 224×224 pixels. Sampling ensures a configurable minimum number of labeled seen-class pixels. Validation and test use sliding windows with 56-pixel overlap and weighted overlap averaging. Images smaller than a tile are reflection-padded; logits are cropped to the original size.

The repository provides example configuration files with class names and input conventions. File paths and array keys are deliberately explicit because public copies of these datasets use different names and MATLAB keys.

## 8. Open-Vocabulary Protocol

Every experiment config declares `seen_class_ids` and `unseen_class_ids`. Training segmentation loss is computed only for pixels belonging to seen classes. Alignment losses may use every spatial pixel because they do not consume ground-truth class IDs.

Evaluation reports:

- mean IoU and mean class accuracy over all configured classes;
- seen-class mIoU;
- unseen-class mIoU;
- harmonic mean of seen and unseen mIoU;
- overall pixel accuracy;
- per-class IoU.

Example configs use deterministic research splits and label them as project defaults, not community-standard splits. Changing a split requires only YAML edits and is recorded in the resolved run configuration.

## 9. Losses

The total loss is:

`L = L_seg + lambda_teacher * (L_hsi_teacher + L_lidar_teacher) + lambda_cross * L_hsi_lidar + lambda_gate * L_gate + lambda_private * L_private`.

- `L_seg` is masked pixelwise cross-entropy over seen-class text logits.
- Alignment losses use normalized region-pooled tokens and a symmetric InfoNCE objective. Positives share geolocation. Negatives within a configurable spatial exclusion radius are removed to reduce false negatives caused by neighboring pixels.
- `L_gate` discourages gate saturation during warm-up by penalizing deviation of the batch mean from 0.5.
- `L_private` decorrelates optional modality-private projections from the shared fused representation.

Each loss accepts a validity mask and returns a finite scalar. An empty supervised mask is a data error, not a zero loss.

## 10. Training and Checkpointing

The CLI has `train`, `evaluate`, and `validate-config` commands. Configuration is loaded from YAML into validated dataclasses; unknown keys are errors.

Training uses AdamW, separate learning-rate groups for adapters/decoder and unfrozen backbone blocks, gradient clipping, automatic mixed precision when CUDA is available, deterministic seeding, and periodic validation. Checkpoints contain model state, optimizer state, scheduler state, scaler state, epoch, global step, normalization statistics, resolved configuration, class names, and seen/unseen splits.

Resume rejects checkpoints whose class list, split, spectral band count, or model dimensions conflict with the current configuration. Best checkpoints are selected by harmonic seen/unseen mIoU when unseen classes exist, otherwise by overall mIoU.

## 11. Error Handling and Diagnostics

Configuration validation fails early for missing files, missing array keys, invalid class IDs, overlapping seen/unseen sets, absent pseudo-RGB bands, even terrain windows, invalid tile overlap, and missing required local checkpoints.

Runtime validation checks paired spatial dimensions, feature-pyramid lengths, channel counts, non-finite model outputs, and text/label class-count consistency. Error messages identify the dataset, field, expected value, and observed value.

Training logs total and component losses, learning rates, gate means, temperature, validation metrics, and checkpoint paths. The implementation uses the Python `logging` module and never relies on print statements inside library modules.

## 12. Project Structure

```text
HSI-LIDAR/
├── pyproject.toml
├── README.md
├── configs/
│   ├── base.yaml
│   ├── houston2013.yaml
│   ├── trento.yaml
│   └── muufl.yaml
├── src/hsi_lidar_ovseg/
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── io.py
│   │   ├── preprocessing.py
│   │   ├── datasets.py
│   │   └── tiling.py
│   ├── models/
│   │   ├── protocols.py
│   │   ├── hypersigma.py
│   │   ├── dinov2.py
│   │   ├── clip_text.py
│   │   ├── native.py
│   │   ├── fusion.py
│   │   ├── decoder.py
│   │   └── model.py
│   ├── losses/
│   │   ├── contrastive.py
│   │   └── objective.py
│   ├── engine/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── checkpoint.py
│   └── metrics.py
└── tests/
    ├── test_config.py
    ├── test_io.py
    ├── test_preprocessing.py
    ├── test_tiling.py
    ├── test_encoders.py
    ├── test_fusion.py
    ├── test_losses.py
    ├── test_model.py
    ├── test_metrics.py
    └── test_training_smoke.py
```

Files are divided by responsibility. External model peculiarities remain inside their adapters; data formats remain inside data modules; the core segmentation model depends only on typed protocols.

## 13. Quality and Verification

The project targets Python 3.10 or newer and PyTorch 2.1 or newer. Public functions and configuration dataclasses use type annotations. Ruff enforces formatting, imports, and common correctness rules. Pytest is the test runner.

Tests must cover:

- every supported array format and invalid shape/key errors;
- train-only normalization and finite terrain features;
- paired crop and sliding-window reconstruction;
- encoder pyramid shape contracts;
- gate bounds and fusion gradients;
- alignment masking and empty-mask errors;
- dense text-logit dimensions and normalized embeddings;
- seen/unseen and harmonic metrics;
- checkpoint incompatibility detection;
- one synthetic CPU training step and one sliding-window evaluation pass.

Completion requires all offline tests to pass, Ruff checks to pass, the package to build successfully, and CLI help plus configuration validation to execute without accessing the network.

## 14. Acceptance Criteria

The implementation is accepted when:

1. `python -m pytest` passes without data or pretrained weights.
2. `ruff check .` reports no violations.
3. `python -m build` produces source and wheel distributions.
4. `hsi-lidar-ovseg --help` lists train, evaluate, and validate-config commands.
5. A synthetic configuration completes one CPU training step and sliding-window evaluation.
6. Each real dataset configuration validates after the user supplies its paths, MATLAB keys, and local pretrained checkpoints.
7. No command downloads a model or dataset unless that behavior is added later through an explicit, separately approved feature.
