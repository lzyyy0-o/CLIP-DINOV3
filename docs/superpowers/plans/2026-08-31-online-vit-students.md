# Online ViT Students Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace target HSI and LiDAR students with trainable ViT-S/16 encoders, with a spectral adapter on HSI only.

**Architecture:** A project-owned ViT emits token features at blocks 2/5/8/11 and restores them to the existing four-scale encoder contract. Existing frozen DINOv3 and RemoteCLIP teachers, losses, fusion, and decoder remain unchanged.

**Tech Stack:** Python 3.11, PyTorch, PyYAML, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-31-online-vit-students-design.md`

## Global Constraints

- ViT-S/16 is fixed at 384 dimensions, 12 blocks, 6 heads and MLP ratio 4.
- HSI uses a trainable 1×1 spectral adapter; LiDAR does not.
- Students have no external weights or source dependencies.
- Native and pretrained experiment templates remain unchanged.
- Teachers remain frozen and use no autograd graph.

---

### Task 1: Add the online ViT four-scale encoder

**Files:**
- Create: `src/hsi_lidar_ovseg/models/online_vit.py`
- Modify: `src/hsi_lidar_ovseg/models/__init__.py`
- Test: `tests/test_online_vit.py`

**Interfaces:**
- Produces `OnlineViTPyramidEncoder(in_channels: int, spectral_adapter: bool)` with `out_channels == (384, 384, 384, 384)`.
- `forward(inputs: Tensor) -> FeaturePyramid` returns NCHW scales at strides 4/8/16/32.

- [ ] **Step 1: Write failing encoder tests**

```python
def test_hsi_online_vit_uses_spectral_adapter_and_returns_four_scales() -> None:
    encoder = OnlineViTPyramidEncoder(6, spectral_adapter=True)
    outputs = encoder(torch.randn(1, 6, 32, 32))
    assert encoder.spectral_adapter is not None
    assert [item.shape for item in outputs] == [(1, 384, 8, 8), (1, 384, 4, 4), (1, 384, 2, 2), (1, 384, 1, 1)]
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_online_vit.py -q`

Expected: import failure before implementation.

- [ ] **Step 3: Implement patch embedding, twelve Transformer blocks, token extraction, and four-scale interpolation**

```python
class OnlineViTPyramidEncoder(nn.Module):
    out_strides = (4, 8, 16, 32)
    out_channels = (384, 384, 384, 384)
    feature_blocks = (2, 5, 8, 11)
```

Use `nn.TransformerEncoderLayer(batch_first=True, norm_first=True, activation="gelu")`; HSI gets `Conv2d(in_channels, 384, 1)` before the patch projection and LiDAR uses the patch projection directly.

- [ ] **Step 4: Run encoder tests and existing model tests**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_online_vit.py tests/test_encoders.py tests/test_model.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/hsi_lidar_ovseg/models/online_vit.py src/hsi_lidar_ovseg/models/__init__.py tests/test_online_vit.py
git commit -m "feat: add online ViT student encoder"
```

### Task 2: Integrate online ViT configuration and experiment template

**Files:**
- Modify: `src/hsi_lidar_ovseg/config.py`
- Modify: `src/hsi_lidar_ovseg/cli.py`
- Create: `configs/online_vit.yaml`
- Modify: `README.md`
- Test: `tests/test_config.py`, `tests/test_cli.py`

**Interfaces:**
- `EncoderConfig.kind == "online_vit"` accepts only model name `vit_small_patch16`, feature blocks `(2,5,8,11)`, and boolean `spectral_adapter`.
- `_build_visual_encoder` constructs `OnlineViTPyramidEncoder` and enforces HSI true / LiDAR false spectral adapter settings.

- [ ] **Step 1: Write failing config and CLI construction tests**

```python
def test_online_vit_rejects_external_checkpoint() -> None:
    with pytest.raises(ConfigError, match="checkpoint"):
        EncoderConfig(kind="online_vit", checkpoint=Path("weights.pt"), spectral_adapter=True)
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py tests/test_cli.py -q`

Expected: failure because `online_vit` is unsupported.

- [ ] **Step 3: Implement config parsing, CLI dispatch, template, and documentation**

Add `spectral_adapter: bool | None` to `EncoderConfig`; reject all external fields for online ViT. Add `configs/online_vit.yaml` with online HSI/LiDAR students and unchanged local DINOv3/RemoteCLIP teachers. Document the online mode and expected 7–10 GiB peak.

- [ ] **Step 4: Run focused tests**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py tests/test_cli.py tests/test_online_vit.py -q`

Expected: PASS; native smoke remains independent of third-party sources.

- [ ] **Step 5: Commit**

```powershell
git add src/hsi_lidar_ovseg/config.py src/hsi_lidar_ovseg/cli.py configs/online_vit.yaml README.md tests/test_config.py tests/test_cli.py
git commit -m "feat: add online ViT student experiment"
```

### Task 3: Verify the online ViT student integration

**Files:**
- Modify: `docs/superpowers/plans/2026-08-31-online-vit-students.md`

- [ ] **Step 1: Run static checks without formatting third-party source**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m ruff format src tests` then `D:\miniconda\envs\hsi-lidar\python.exe -m ruff check src tests`

Expected: no violations.

- [ ] **Step 2: Run full tests and package build**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest -q` then `D:\miniconda\envs\hsi-lidar\python.exe -m build --no-isolation`

Expected: all tests pass and the package excludes `third_party` and weights.

- [ ] **Step 3: Commit verification record**

```powershell
git add docs/superpowers/plans/2026-08-31-online-vit-students.md
git commit -m "docs: record online ViT student verification"
```
