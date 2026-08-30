# Pretrained Encoder Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add offline, testable HyperSIGMA, DINOv3, and RemoteCLIP pretrained-encoder integration while preserving the native training baseline.

**Architecture:** Project-owned factories accept resolved local source directories and construct official backbones without weights or network access. Bridge modules make dataset-specific HSI and LiDAR channels compatible with pretrained RGB/fixed-band backbones, while existing four-scale adapters and decoder remain the common segmentation interface.

**Tech Stack:** Python 3.11, PyTorch, PyYAML, pytest, ruff, locally cloned official repositories, OpenCLIP.

**Spec:** `docs/superpowers/specs/2026-08-30-pretrained-encoder-integration-design.md`

**Execution status (2026-08-30):** Tasks 1–5 implemented and committed. Task 6 verification completed with `ruff check src tests`, `pytest -q` (94 passed), and `python -m build --no-isolation`.

## Global Constraints

- Never call `torch.hub`, Hugging Face, or a downloader during factory construction.
- Keep `configs/houston2013.yaml`, `configs/trento.yaml`, and `configs/muufl.yaml` on the `native` baseline.
- HyperSIGMA requires both `spatial_checkpoint` and `spectral_checkpoint`, with strict per-branch loading.
- DINOv3 uses one strict local checkpoint and retains its official three-channel first layer.
- Only project-created input adapters and decoder projections are newly initialized and trainable.
- Teachers remain frozen and are evaluated without an autograd graph.
- Keep external source trees and all checkpoints out of the project package and Git commits.

---

### Task 1: Extend the external-encoder configuration contract

**Files:**
- Modify: `src/hsi_lidar_ovseg/config.py:EncoderConfig`
- Modify: `src/hsi_lidar_ovseg/cli.py:_build_visual_encoder`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces `EncoderConfig.source_dir: Path | None`, `spatial_checkpoint: Path | None`, `spectral_checkpoint: Path | None`, and `pretrained_in_channels: int | None`.
- `EncoderConfig.external_weight_paths() -> tuple[Path, ...]` returns the strict local files needed by its kind.
- `_build_visual_encoder` passes `source_dir`, `in_channels`, and `pretrained_in_channels` to project-owned factories.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_hypersigma_requires_both_component_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="spatial_checkpoint 与 spectral_checkpoint"):
        EncoderConfig(
            kind="hypersigma",
            factory="hsi_lidar_ovseg.models.factories:create_hypersigma",
            source_dir=tmp_path,
            spatial_checkpoint=tmp_path / "spatial.pt",
            pretrained_in_channels=100,
        )


def test_dinov3_requires_source_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="source_dir"):
        EncoderConfig(
            kind="dinov3_convnext",
            checkpoint=tmp_path / "dino.pt",
            factory="hsi_lidar_ovseg.models.factories:create_dinov3",
        )
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py -q`

Expected: FAIL because the new constructor fields and validation do not exist.

- [ ] **Step 3: Add fields, path decoding, and validation**

```python
@dataclass(frozen=True)
class EncoderConfig:
    source_dir: Path | None = None
    spatial_checkpoint: Path | None = None
    spectral_checkpoint: Path | None = None
    pretrained_in_channels: int | None = None

    def external_weight_paths(self) -> tuple[Path, ...]:
        if self.kind == "hypersigma":
            assert self.spatial_checkpoint is not None and self.spectral_checkpoint is not None
            return (self.spatial_checkpoint, self.spectral_checkpoint)
        return () if self.checkpoint is None else (self.checkpoint,)
```

Decode all four fields with the existing `_path` helper. For HyperSIGMA, reject `checkpoint`, require both component weights, a source directory, and positive `pretrained_in_channels`; for DINOv3, require `checkpoint` and `source_dir`; retain the existing RemoteCLIP constraints.

- [ ] **Step 4: Run focused tests and all configuration tests**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py -q`

Expected: PASS with legacy native configurations unchanged.

- [ ] **Step 5: Commit**

```powershell
git add src/hsi_lidar_ovseg/config.py src/hsi_lidar_ovseg/cli.py tests/test_config.py
git commit -m "feat: validate external encoder configuration"
```

### Task 2: Add safe source loading and channel-adapter primitives

**Files:**
- Create: `src/hsi_lidar_ovseg/models/factories.py`
- Create: `src/hsi_lidar_ovseg/models/input_adapter.py`
- Modify: `src/hsi_lidar_ovseg/models/__init__.py`
- Test: `tests/test_external_factories.py`

**Interfaces:**
- `add_source_path(source_dir: Path, required_child: str) -> Path` validates a cloned repository layout and returns the import root.
- `ChannelAdapter(input_channels: int, output_channels: int)` exposes `forward(inputs: Tensor) -> Tensor` through a trainable `1×1` convolution.
- `create_dinov3(*, source_dir: Path, model_name: str, in_channels: int) -> nn.Module` builds with `pretrained=False` only.

- [ ] **Step 1: Write failing primitive tests**

```python
def test_channel_adapter_changes_only_channels() -> None:
    adapter = ChannelAdapter(input_channels=1, output_channels=3)
    result = adapter(torch.randn(2, 1, 32, 32))
    assert result.shape == (2, 3, 32, 32)
    assert all(parameter.requires_grad for parameter in adapter.parameters())


def test_add_source_path_requires_expected_official_layout(tmp_path: Path) -> None:
    with pytest.raises(ExternalSourceError, match="dinov3"):
        add_source_path(tmp_path, "dinov3")
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_external_factories.py -q`

Expected: FAIL because the factory and adapter modules do not exist.

- [ ] **Step 3: Implement path isolation and the adapter**

```python
class ChannelAdapter(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("通道适配器输入必须为 NCHW 张量")
        return self.projection(inputs)
```

`add_source_path` must reject nonexistent directories and expected child modules, then add only the requested repository root to `sys.path`. `create_dinov3` imports `dinov3.hub.backbones`, obtains the requested public factory, and calls it with `pretrained=False`; it must reject model names outside the DINOv3 ViT/ConvNeXt public constructors.

- [ ] **Step 4: Run focused tests**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_external_factories.py -q`

Expected: PASS without network access.

- [ ] **Step 5: Commit**

```powershell
git add src/hsi_lidar_ovseg/models/factories.py src/hsi_lidar_ovseg/models/input_adapter.py src/hsi_lidar_ovseg/models/__init__.py tests/test_external_factories.py
git commit -m "feat: add offline external encoder factories"
```

### Task 3: Implement the full HyperSIGMA bridge and dual-weight loading

**Files:**
- Create: `src/hsi_lidar_ovseg/models/hypersigma_bridge.py`
- Modify: `src/hsi_lidar_ovseg/models/hypersigma.py`
- Modify: `src/hsi_lidar_ovseg/cli.py:_load_local_weights,_build_visual_encoder`
- Modify: `src/hsi_lidar_ovseg/models/factories.py`
- Test: `tests/test_hypersigma_bridge.py`

**Interfaces:**
- `HyperSigmaBridge(spatial_encoder, spectral_encoder, input_channels, pretrained_in_channels)` exposes `forward_intermediates(inputs, indices) -> tuple[Tensor, Tensor, Tensor, Tensor]`, with NCHW outputs.
- `load_hypersigma_weights(bridge: HyperSigmaBridge, spatial_path: Path, spectral_path: Path) -> None` strictly loads each named branch.
- `create_hypersigma(*, source_dir: Path, model_name: str, in_channels: int, pretrained_in_channels: int) -> HyperSigmaBridge` builds official Base, Large, or Huge spatial/spectral encoders without weights.

- [ ] **Step 1: Write failing bridge tests with lightweight official-shaped stubs**

```python
def test_hypersigma_bridge_adapts_bands_and_modulates_four_scales() -> None:
    bridge = HyperSigmaBridge(
        spatial_encoder=FakeSpatialEncoder(),
        spectral_encoder=FakeSpectralEncoder(),
        input_channels=6,
        pretrained_in_channels=4,
    )
    outputs = bridge.forward_intermediates(torch.randn(1, 6, 32, 32), indices=(2, 5, 8, 11))
    assert [output.shape for output in outputs] == [(1, 8, 8, 8)] * 4
    assert bridge.input_adapter.projection.in_channels == 6
    assert bridge.input_adapter.projection.out_channels == 4


def test_hypersigma_loader_names_the_failing_branch(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="空间分支"):
        load_hypersigma_weights(bridge, tmp_path / "bad.pt", tmp_path / "spectral.pt")
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_hypersigma_bridge.py -q`

Expected: FAIL because no bridge or dual-weight loader exists.

- [ ] **Step 3: Implement the bridge and strict component loading**

The bridge must call the official spatial feature function exactly once, obtain its four NCHW scale features, reduce the official spectral output to one vector per sample, and apply one learned sigmoid channel gate to each scale. It must expose `patch_size`, `embed_dim`, and `blocks` from the spatial encoder so existing partial-unfreeze logic remains valid. Update `HyperSigmaAdapter` to accept NCHW intermediate features directly and project/resample them to the shared four-scale contract; retain token support for existing compatible injected backbones.

```python
def load_hypersigma_weights(bridge: HyperSigmaBridge, spatial_path: Path, spectral_path: Path) -> None:
    _load_state_dict_strict(bridge.spatial_encoder, spatial_path, "HyperSIGMA 空间分支")
    _load_state_dict_strict(bridge.spectral_encoder, spectral_path, "HyperSIGMA 光谱分支")
```

- [ ] **Step 4: Run focused bridge and adapter tests**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_hypersigma_bridge.py tests/test_adapters.py -q`

Expected: PASS, including frozen and partially unfrozen bridge paths.

- [ ] **Step 5: Commit**

```powershell
git add src/hsi_lidar_ovseg/models/hypersigma_bridge.py src/hsi_lidar_ovseg/models/hypersigma.py src/hsi_lidar_ovseg/models/factories.py src/hsi_lidar_ovseg/cli.py tests/test_hypersigma_bridge.py
git commit -m "feat: bridge dual-branch HyperSIGMA features"
```

### Task 4: Integrate DINOv3 channel adaptation and pretrained construction

**Files:**
- Create: `src/hsi_lidar_ovseg/models/dinov3_bridge.py`
- Modify: `src/hsi_lidar_ovseg/models/dinov3.py`
- Modify: `src/hsi_lidar_ovseg/cli.py:_build_visual_encoder`
- Test: `tests/test_dinov3_bridge.py`

**Interfaces:**
- `DinoV3InputBridge(backbone: nn.Module, input_channels: int)` has `input_adapter: ChannelAdapter` targeting three channels and forwards all public DINOv3 attributes needed by adapters.
- `create_dinov3` returns a raw official three-channel model; `_build_visual_encoder` loads weights before wrapping it in `DinoV3InputBridge` when data input channels differ from three.

- [ ] **Step 1: Write failing DINOv3 bridge tests**

```python
def test_dinov3_input_bridge_keeps_backbone_rgb_and_adapts_lidar() -> None:
    backbone = FakeConvNeXt()
    bridge = DinoV3InputBridge(backbone, input_channels=1)
    outputs = bridge.get_intermediate_layers(torch.randn(2, 1, 64, 64), n=(0, 1, 2, 3))
    assert bridge.input_adapter.projection.out_channels == 3
    assert len(outputs) == 4
    assert bridge.embed_dims == backbone.embed_dims
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_dinov3_bridge.py -q`

Expected: FAIL because the bridge does not exist.

- [ ] **Step 3: Implement the bridge and wire load order**

The bridge must expose `embed_dims`, `stages`, `downsample_layers`, and `get_intermediate_layers`, so `DinoV3ConvNeXtAdapter` remains unchanged. It must return the raw backbone unchanged for three-channel inputs. In `_build_visual_encoder`, strictly load the official DINOv3 state dictionary before creating a bridge; this prevents a trainable input adapter from appearing in the official checkpoint key space.

- [ ] **Step 4: Run focused DINOv3 tests and existing adapter tests**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_dinov3_bridge.py tests/test_adapters.py -q`

Expected: PASS with one-channel LiDAR and three-channel pseudo-RGB cases.

- [ ] **Step 5: Commit**

```powershell
git add src/hsi_lidar_ovseg/models/dinov3_bridge.py src/hsi_lidar_ovseg/models/dinov3.py src/hsi_lidar_ovseg/cli.py tests/test_dinov3_bridge.py
git commit -m "feat: adapt LiDAR channels for DINOv3"
```

### Task 5: Add the pretrained experiment template, dependency documentation, and end-to-end checks

**Files:**
- Create: `configs/pretrained.yaml`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- `configs/pretrained.yaml` references `hsi_lidar_ovseg.models.factories:create_hypersigma` and `create_dinov3`, local `third_party` paths, separate HyperSIGMA weights, DINOv3 weights, and RemoteCLIP weights.
- Extra dependency group `pretrained` includes the packages required by official HyperSIGMA import paths: `einops`, `mmengine`, and the existing OpenCLIP/timm/torchvision packages.

- [ ] **Step 1: Write failing template tests**

```python
def test_pretrained_template_has_only_local_external_resources() -> None:
    config = load_config(Path("configs/pretrained.yaml"), check_files=False)
    assert config.model.hsi_encoder.spatial_checkpoint is not None
    assert config.model.hsi_encoder.spectral_checkpoint is not None
    assert config.model.lidar_encoder.source_dir == Path("third_party/dinov3")
    assert config.model.semantic_teacher_encoder.kind == "remoteclip"
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py::test_pretrained_template_has_only_local_external_resources -q`

Expected: FAIL because the template is absent.

- [ ] **Step 3: Add the template and user documentation**

Create a 224-pixel, AMP, batch-size-one template with relative paths under `third_party/` and `weights/`. Update README with the exact `pip install -e ".[pretrained]"` command, source clone locations, official source revisions, expected weight filenames, GPU memory ranges from the approved design, and a validation command that performs no network access.

- [ ] **Step 4: Run configuration and CLI smoke tests**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py tests/test_cli.py -q`

Expected: PASS; the native CLI smoke test must still run without third-party dependencies or weights.

- [ ] **Step 5: Commit**

```powershell
git add configs/pretrained.yaml pyproject.toml README.md tests/test_config.py tests/test_cli.py
git commit -m "docs: add pretrained encoder experiment template"
```

### Task 6: Verify the completed integration

**Files:**
- Modify: `docs/superpowers/plans/2026-08-30-pretrained-encoder-integration.md`

**Interfaces:**
- No new runtime interface; this task verifies the integration contract created by Tasks 1–5.

- [ ] **Step 1: Format and run static analysis**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m ruff format .` then `D:\miniconda\envs\hsi-lidar\python.exe -m ruff check .`

Expected: formatting completes and ruff reports no violations.

- [ ] **Step 2: Run the full suite**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest -q`

Expected: PASS with no collection errors when weights are absent.

- [ ] **Step 3: Build the package**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m build --no-isolation`

Expected: successful sdist and wheel build; third-party source trees and weights are not packaged.

- [ ] **Step 4: Record completion and commit the plan**

```powershell
git add docs/superpowers/plans/2026-08-30-pretrained-encoder-integration.md
git commit -m "docs: add pretrained encoder integration plan"
```
