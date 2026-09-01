# CLIP 多尺度对齐与相关性残差 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结 OpenAI CLIP 的条件下，以多尺度视觉对齐和固定的原始文本相关性残差增强 HSI--LiDAR 开放词汇分割。

**Architecture:** Segmentor 返回 logits 及既有 joint/CLIP 四尺度金字塔。新目标在有效训练像素上约束 joint 特征贴近 detached CLIP 特征；解码器把四尺度原始 CLIP--文本相关图以固定系数加入学习型 FPN logits。

**Tech Stack:** Python 3.11、PyTorch、PyYAML、pytest、Ruff。

**Spec:** `docs/superpowers/specs/2026-09-01-clip-alignment-residual-design.md`

## Global Constraints

- 仅修改 `clip_guided_shared_lite_vit`，不改变 teacher-student 架构、数据切分和 seen/unseen 指标。
- HSI--LiDAR 保持六个 Lite-ViT block、四尺度、512 维输出；OpenAI CLIP ViT-B/16 保持 `unfreeze_blocks: 0`。
- 对齐中的 CLIP feature 必须 `detach()`；分割与对齐均只使用 `valid_mask` 的训练像素。
- 原始 CLIP 残差权重固定为 `0.25`，不可训练且为正；Houston 使用 `clip_guided_alignment` 与权重 `0.1`。
- 保留 `masked_cross_entropy` baseline；每项先写失败测试，结束时跑 pytest、Ruff 和 Houston 配置校验。

---

### Task 1: 公开模型的双特征金字塔

**Files:**
- Modify: `src/hsi_lidar_ovseg/models/clip_guided_model.py:12-63`
- Modify: `tests/test_clip_guided_model.py:47-64`

**Interfaces:**
- Consumes: `joint_projector` 和 `clip_guidance.visual_features()` 的四尺度 `FeaturePyramid`。
- Produces: `ClipGuidedSegmentationOutput(logits, joint_features, clip_features)`，供 Task 2 读取。

- [ ] **Step 1: 写出失败测试**

在 `test_clip_guided_model_outputs_logits_for_dynamic_class_names` 的结尾加入：

```python
expected = ((1, 512, 8, 8), (1, 512, 4, 4), (1, 512, 2, 2), (1, 512, 1, 1))
assert tuple(level.shape for level in output.joint_features) == expected
assert tuple(level.shape for level in output.clip_features) == expected
```

- [ ] **Step 2: 确认测试失败**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_clip_guided_model.py -q`

Expected: FAIL，输出没有 `joint_features`。

- [ ] **Step 3: 最小实现**

在 `clip_guided_model.py` 导入 `FeaturePyramid` 并改为：

```python
@dataclass(frozen=True)
class ClipGuidedSegmentationOutput:
    logits: Tensor
    joint_features: FeaturePyramid
    clip_features: FeaturePyramid
```

将 `forward` 的返回替换为：

```python
return ClipGuidedSegmentationOutput(
    logits=logits,
    joint_features=joint_features,
    clip_features=clip_features,
)
```

不得重复运行编码器，也不得在模型中 detach 特征。

- [ ] **Step 4: 验证并提交**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_clip_guided_model.py -q`

Expected: PASS。

```powershell
git add src/hsi_lidar_ovseg/models/clip_guided_model.py tests/test_clip_guided_model.py
git commit -m "feat: expose CLIP guided feature pyramids"
```

### Task 2: 实现带掩码的四尺度 CLIP 对齐损失

**Files:**
- Modify: `src/hsi_lidar_ovseg/losses/cross_entropy.py:1-57`
- Modify: `src/hsi_lidar_ovseg/losses/__init__.py:1-14`
- Modify: `tests/test_cross_entropy.py:1-51`

**Interfaces:**
- Consumes: 输出的 `logits`、`joint_features`、`clip_features` 与 `labels[N,H,W]`、`valid_mask[N,H,W]`。
- Produces: `ClipGuidedAlignmentObjective(seen_class_ids, clip_alignment_weight)`，返回 `total`、`segmentation`、`clip_alignment`。

- [ ] **Step 1: 写出失败测试**

在 `tests/test_cross_entropy.py` 增加 fake 输出和以下测试：

```python
def _alignment_output(joint: torch.Tensor, clip: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        logits=torch.randn(1, 2, 4, 4, requires_grad=True),
        joint_features=(joint, joint, joint, joint),
        clip_features=(clip, clip, clip, clip),
    )

def test_clip_alignment_detaches_identical_teacher_features() -> None:
    joint = torch.randn(1, 512, 4, 4, requires_grad=True)
    teacher = joint.detach().clone().requires_grad_()
    losses = ClipGuidedAlignmentObjective((1, 2), 0.1)(
        _alignment_output(joint, teacher),
        torch.tensor([[[1, 2, 1, 2]] * 4]),
        torch.ones(1, 4, 4, dtype=torch.bool),
    )
    assert losses["clip_alignment"] < 1e-5
    losses["total"].backward()
    assert joint.grad is not None
    assert teacher.grad is None
```

再新增一个测试：joint 的第 0 通道全为 1、teacher 的第 1 通道全为 1，断言 `clip_alignment > 0.9`；仅把一个不匹配像素设为 invalid 后，断言 `clip_alignment < 1e-5`。

- [ ] **Step 2: 确认测试失败**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_cross_entropy.py -q`

Expected: FAIL，无法导入 `ClipGuidedAlignmentObjective`。

- [ ] **Step 3: 最小实现**

在 `cross_entropy.py` 导入 `FeaturePyramid` 并定义：

```python
class _ClipGuidedOutput(_LogitOutput, Protocol):
    joint_features: FeaturePyramid
    clip_features: FeaturePyramid
```

实现 `ClipGuidedAlignmentObjective(MaskedCrossEntropyObjective)`。构造器拒绝非正 `clip_alignment_weight`。`forward` 先调用 `super().forward`，验证双方均为 4 层、形状相同、NCHW、512 通道；每层执行：

```python
mask = functional.interpolate(
    valid_mask[:, None].float(), size=joint.shape[-2:], mode="nearest"
).squeeze(1).bool()
if not mask.any():
    raise ValueError("对齐掩码中没有有效像素")
student = functional.normalize(joint, dim=1)
teacher = functional.normalize(clip.detach(), dim=1)
level_loss = (1 - (student * teacher).sum(dim=1))[mask].mean()
```

对四层 `level_loss` 取均值为 `alignment`，返回：

```python
return {
    "total": losses["segmentation"] + self.clip_alignment_weight * alignment,
    "segmentation": losses["segmentation"],
    "clip_alignment": alignment,
}
```

在 `losses/__init__.py` 导出该类。不得使用 batch negatives、类别标签或任意不在 mask 内的像素来计算对齐。

- [ ] **Step 4: 验证并提交**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_cross_entropy.py -q`

Expected: PASS，teacher tensor 没有对齐损失带来的梯度。

```powershell
git add src/hsi_lidar_ovseg/losses/cross_entropy.py src/hsi_lidar_ovseg/losses/__init__.py tests/test_cross_entropy.py
git commit -m "feat: align joint pyramid to frozen CLIP"
```

### Task 3: 保留固定的原始 CLIP 文本相关性残差

**Files:**
- Modify: `src/hsi_lidar_ovseg/models/correlation_decoder.py:29-105`
- Modify: `tests/test_correlation_decoder.py`

**Interfaces:**
- Consumes: 每层 CLIP feature 与 `text_features[class,prompt,512]`。
- Produces: `TextCorrelationDecoder(feature_dim=512, hidden_dim=64, clip_residual_weight=0.25)`；输出 logits 恒含原始 CLIP residual。

- [ ] **Step 1: 写出失败测试**

在 `tests/test_correlation_decoder.py` 新增：

```python
def test_decoder_keeps_fixed_raw_clip_text_correlation_residual() -> None:
    decoder = TextCorrelationDecoder(512, 8)
    for parameter in decoder.parameters():
        parameter.data.zero_()
    clip = tuple(torch.zeros(1, 512, size, size) for size in (56, 28, 14, 7))
    joint = tuple(torch.zeros_like(level) for level in clip)
    clip = tuple(level.clone() for level in clip)
    for level in clip:
        level[:, 0] = 1.0
    text = torch.zeros(2, 1, 512)
    text[0, 0, 0] = 1.0
    text[1, 0, 1] = 1.0
    logits = decoder(joint, clip, text, (56, 56))
    assert logits[:, 0].mean() > 0.20
    assert logits[:, 1].abs().max() < 1e-6
```

再断言 `TextCorrelationDecoder(512, 8, clip_residual_weight=0.0)` 抛出 `ValueError`。

- [ ] **Step 2: 确认测试失败**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_correlation_decoder.py -q`

Expected: FAIL，因为可学习参数归零后当前解码器输出全零。

- [ ] **Step 3: 最小实现**

构造器增加 `clip_residual_weight: float = 0.25`；拒绝非正值，并保存为普通 Python float，不能放入 `nn.Parameter`。循环每个尺度时将第二个相关图收集为 `raw_clip_correlations`。解码完成后：

```python
residual = torch.stack(
    [functional.interpolate(level, size=output_size, mode="bilinear", align_corners=False)
     for level in raw_clip_correlations],
    dim=0,
).mean(dim=0)
learned_logits = functional.interpolate(
    logits, size=output_size, mode="bilinear", align_corners=False
)
return learned_logits + self.clip_residual_weight * residual
```

其中每个收集的 map 必须是 `[batch, classes, height, width]`，由当前 `_correlation(clip, text_features)` 直接产生，不能取 embedding/aggregator 后的 cost volume，也不能引入类别专属可训练参数。

- [ ] **Step 4: 验证并提交**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_correlation_decoder.py -q`

Expected: PASS，包括现有动态类别、窗口聚合与梯度测试。

```powershell
git add src/hsi_lidar_ovseg/models/correlation_decoder.py tests/test_correlation_decoder.py
git commit -m "feat: preserve raw CLIP correlation residual"
```

### Task 4: 接入严格配置与训练 objective 选择

**Files:**
- Modify: `src/hsi_lidar_ovseg/config.py:417-534`
- Modify: `src/hsi_lidar_ovseg/cli.py:50,509-522`
- Modify: `configs/houston2013_shared_lite_vit_clip_rs_fusion.yaml:31`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: YAML `loss.kind` 与 `loss.clip_alignment_weight`。
- Produces: CLIP 架构可选择 `MaskedCrossEntropyObjective` baseline 或 `ClipGuidedAlignmentObjective`。

- [ ] **Step 1: 写出失败配置测试**

在 `tests/test_config.py` 使用 `_clip_guided_config_dict()` 写入：

```python
def test_clip_guided_config_accepts_alignment_loss(tmp_path: Path) -> None:
    values = _clip_guided_config_dict()
    loss = values["loss"]
    assert isinstance(loss, dict)
    loss.update(kind="clip_guided_alignment", clip_alignment_weight=0.1)
    path = tmp_path / "alignment.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    assert load_config(path, check_files=False).loss.clip_alignment_weight == 0.1

@pytest.mark.parametrize("weight", [0.0, -0.1])
def test_clip_guided_alignment_rejects_non_positive_weight(tmp_path: Path, weight: float) -> None:
    values = _clip_guided_config_dict()
    loss = values["loss"]
    assert isinstance(loss, dict)
    loss.update(kind="clip_guided_alignment", clip_alignment_weight=weight)
    path = tmp_path / "invalid-alignment.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ConfigError, match="clip_alignment_weight"):
        load_config(path, check_files=False)
```

再新增 `masked_cross_entropy` 携带 `clip_alignment_weight: 0.1` 被拒绝的测试；保留现有 baseline 成功测试。

- [ ] **Step 2: 确认测试失败**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py -q`

Expected: FAIL，新的 kind 或键尚未被支持。

- [ ] **Step 3: 实现配置与 CLI 分派**

在 `LossConfig` 增加：

```python
kind: Literal["teacher_student", "masked_cross_entropy", "clip_guided_alignment"] = "teacher_student"
clip_alignment_weight: float = 0.0
```

验证必须要求：alignment kind 的对齐权重为正且五个旧权重全零；非 alignment kind 的对齐权重为零；CLIP 架构仅容许 `masked_cross_entropy` 或 `clip_guided_alignment`；teacher-student 架构拒绝新 kind。

在 `cli.py` 导入新目标并使用：

```python
objective = (
    ClipGuidedAlignmentObjective(config.data.seen_class_ids, config.loss.clip_alignment_weight)
    if config.loss.kind == "clip_guided_alignment"
    else MaskedCrossEntropyObjective(config.data.seen_class_ids)
)
```

Houston YAML 的 `loss` 必须设为：

```yaml
loss: {kind: clip_guided_alignment, clip_alignment_weight: 0.1, structure_teacher_weight: 0.0, semantic_teacher_weight: 0.0, cross_weight: 0.0, gate_weight: 0.0, private_weight: 0.0}
```

- [ ] **Step 4: 验证并提交**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_config.py -q`

Expected: PASS。

```powershell
git add src/hsi_lidar_ovseg/config.py src/hsi_lidar_ovseg/cli.py configs/houston2013_shared_lite_vit_clip_rs_fusion.yaml tests/test_config.py
git commit -m "feat: configure CLIP pyramid alignment"
```

### Task 5: 全量回归与可运行配置验证

**Files:**
- Verify only: Task 1--4 的源文件、测试与 Houston YAML。

**Interfaces:**
- Consumes: 已完成的模型输出、对齐目标、解码器残差和 YAML。
- Produces: 服务器上可直接启动的 Houston 训练配置。

- [ ] **Step 1: 运行完整测试套件**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m pytest -q`

Expected: 全部 PASS；任何失败必须在对应 Task 修复并补充测试。

- [ ] **Step 2: 运行 Ruff**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m ruff check src tests`

Expected: `All checks passed!`。

- [ ] **Step 3: 校验 Houston 配置**

Run: `$env:PYTHONPATH='src'; D:\miniconda\envs\hsi-lidar\python.exe -m hsi_lidar_ovseg.cli validate-config configs/houston2013_shared_lite_vit_clip_rs_fusion.yaml --skip-file-checks`

Expected: exit code 0，日志确认配置有效。

- [ ] **Step 4: 仅提交验证修正并准备交付**

Run: `git status --short`

Run: `git log --oneline main..HEAD`

若验证导致代码修正，只暂存该修正涉及的已跟踪文件并提交 `test: verify CLIP alignment residual`。不得提交用户未跟踪的 `.vscode/`、`docs/figures/` 或 `third_party/`；未经明确要求不得推送远端。
