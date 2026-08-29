# DINOv3 与 RemoteCLIP 双教师 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单 DINOv2 教师网络升级为 HyperSIGMA、DINOv3 ConvNeXt、冻结 DINOv3 结构教师和同源 RemoteCLIP 语义教师/文本塔组成的双教师开放词汇分割器。

**Architecture:** 两个学生继续通过四尺度空间门控融合；冻结 DINOv3 对齐两个学生的结构特征，冻结 RemoteCLIP 对齐融合语义特征。RemoteCLIP 视觉塔和文本塔从同一份本地检查点构建，文本原型只计算一次。

**Tech Stack:** Python 3.10+、PyTorch、OpenCLIP、pytest、Ruff、YAML。

**Spec:** `docs/superpowers/specs/2026-08-28-dinov3-remoteclip-dual-teacher-design.md`

## Global Constraints

- 外部主干只允许显式本地工厂与本地检查点，禁止隐式联网。
- 四层特征的输出步长固定为 `(4, 8, 16, 32)`。
- 两个教师始终冻结并保持评估模式。
- 原生编码器和哈希文本原型仅用于离线测试与消融。
- 所有行为修改严格执行失败测试、最小实现、全量回归的 TDD 循环。

---

### Task 1: 双教师模型输出契约

**Files:**
- Modify: `tests/test_model.py`
- Modify: `src/hsi_lidar_ovseg/models/model.py`

**Interfaces:**
- Consumes: 四个实现 `out_channels` 且返回四层特征的编码器。
- Produces: `HSILidarOVSegmentor(..., structure_teacher_encoder, semantic_teacher_encoder, ...)` 和包含两种教师特征的 `SegmentationOutput`。

- [ ] **Step 1: 写失败测试**

修改模型测试，使 `alignment_features` 的期望键为：

```python
{
    "hsi",
    "lidar",
    "structure_teacher",
    "semantic_teacher",
    "fused",
}
```

并增加测试，调用 `model.train()` 后验证两个教师都处于评估模式且所有教师参数均不需要梯度。

- [ ] **Step 2: 验证测试因单教师接口而失败**

Run: `python -m pytest tests/test_model.py -q`

Expected: FAIL，原因是输出仍只有 `teacher`，且构造器没有双教师参数。

- [ ] **Step 3: 实现最小双教师前向**

将构造器改为：

```python
def __init__(
    self,
    hsi_encoder: nn.Module,
    lidar_encoder: nn.Module,
    structure_teacher_encoder: nn.Module,
    semantic_teacher_encoder: nn.Module,
    feature_dim: int,
    text_dim: int,
    *,
    freeze_teachers: bool = True,
) -> None:
```

为两种教师分别建立四个 1x1 投影，使用同一个 `pseudo_rgb` 顺序前向，并返回新的对齐键。`make_native_model` 使用两个独立的 `NativePyramidEncoder(3, ...)`。

- [ ] **Step 4: 运行模型测试并回归**

Run: `python -m pytest tests/test_model.py tests/test_training_smoke.py -q`

Expected: PASS。

### Task 2: 双教师训练目标

**Files:**
- Modify: `tests/test_losses.py`
- Modify: `src/hsi_lidar_ovseg/config.py`
- Modify: `src/hsi_lidar_ovseg/losses/objective.py`

**Interfaces:**
- Consumes: Task 1 的五组 `alignment_features`。
- Produces: `LossConfig.structure_teacher_weight`、`semantic_teacher_weight` 以及结构/语义教师损失分量。

- [ ] **Step 1: 写失败测试**

测试 fixture 提供 `structure_teacher` 与 `semantic_teacher`，期望损失键为：

```python
{
    "total",
    "segmentation",
    "hsi_structure",
    "lidar_structure",
    "fused_semantic",
    "hsi_lidar",
    "gate",
    "private",
}
```

另用手工权重验证 `total` 精确等于各分量的加权和。

- [ ] **Step 2: 验证测试因旧键和旧权重失败**

Run: `python -m pytest tests/test_losses.py -q`

Expected: FAIL，原因是 `LossConfig` 不接受新权重或目标仍要求 `teacher`。

- [ ] **Step 3: 实现最小损失变化**

定义：

```python
structure = hsi_structure + lidar_structure
total = (
    segmentation
    + config.structure_teacher_weight * structure
    + config.semantic_teacher_weight * fused_semantic
    + config.cross_weight * hsi_lidar
    + config.gate_weight * gate
    + config.private_weight * private
)
```

删除单一 `teacher_weight`，保持所有非负权重校验。

- [ ] **Step 4: 运行损失测试**

Run: `python -m pytest tests/test_losses.py -q`

Expected: PASS。

### Task 3: DINOv3 与 RemoteCLIP 视觉适配器

**Files:**
- Create: `src/hsi_lidar_ovseg/models/dinov3.py`
- Create: `src/hsi_lidar_ovseg/models/remoteclip.py`
- Modify: `src/hsi_lidar_ovseg/models/dinov2.py`
- Modify: `src/hsi_lidar_ovseg/models/__init__.py`
- Modify: `tests/test_encoders.py`

**Interfaces:**
- Produces: `DinoV3ViTAdapter`、`DinoV3ConvNeXtAdapter`、`RemoteClipVisionAdapter`。
- ConvNeXt 主干必须公开 `embed_dims` 和 `get_intermediate_layers`。
- RemoteCLIP 视觉塔必须公开 OpenCLIP 风格 `forward_intermediates`。

- [ ] **Step 1: 写 DINOv3 ConvNeXt 失败测试**

使用真实行为 fake：`get_intermediate_layers(..., reshape=True)` 返回通道 `(8, 16, 32, 64)`、步长 `(4, 8, 16, 32)` 的 NCHW 特征。断言适配器保留形状，并在 `unfreeze_blocks=1` 时只解冻最后 stage 和对应 downsample layer。

- [ ] **Step 2: 验证 DINOv3 测试因类不存在而失败**

Run: `python -m pytest tests/test_encoders.py -q`

Expected: collection FAIL，无法导入 `DinoV3ConvNeXtAdapter`。

- [ ] **Step 3: 实现 DINOv3 适配器**

`DinoV3ViTAdapter` 复用 token 金字塔逻辑。`DinoV3ConvNeXtAdapter.forward` 调用：

```python
backbone.get_intermediate_layers(
    inputs,
    n=(0, 1, 2, 3),
    reshape=True,
    return_class_token=False,
)
```

并验证四层 NCHW、通道和空间步长。

- [ ] **Step 4: 写 RemoteCLIP 失败测试**

fake 的 `forward_intermediates` 返回：

```python
{"image_intermediates": [level_1, level_2, level_3, level_4]}
```

断言输出被投影为 `feature_dim` 并恢复到四个目标尺度；错误键或层数必须抛出 `ValueError`。

- [ ] **Step 5: 实现 RemoteCLIP 适配器并运行测试**

Run: `python -m pytest tests/test_encoders.py -q`

Expected: PASS。

### Task 4: 严格配置和同源 RemoteCLIP 构建

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_cli.py`
- Modify: `src/hsi_lidar_ovseg/config.py`
- Modify: `src/hsi_lidar_ovseg/cli.py`

**Interfaces:**
- Consumes: 新编码器 kinds 与 Task 1 的模型构造器。
- Produces: 严格解析双教师配置以及 `_build_model_and_text(...) -> tuple[HSILidarOVSegmentor, Tensor]`。

- [ ] **Step 1: 写配置失败测试**

把有效配置改为 `structure_teacher_encoder` 和 `semantic_teacher_encoder`。新增测试验证：

- `dinov3_vit`、`dinov3_convnext`、`remoteclip` 是合法 kind；
- `remoteclip` 不要求 factory，但要求本地 checkpoint；
- RemoteCLIP 语义教师与 `clip_checkpoint` 不同会失败；
- RemoteCLIP 模型名与 `clip_model_name` 不同会失败。

- [ ] **Step 2: 验证配置测试失败**

Run: `python -m pytest tests/test_config.py -q`

Expected: FAIL，原因是新 kind 和新字段尚未定义。

- [ ] **Step 3: 实现严格配置**

`ModelConfig` 使用：

```python
structure_teacher_encoder: EncoderConfig
semantic_teacher_encoder: EncoderConfig
```

`EncoderConfig.kind` 接受六种 kind，并针对 `remoteclip` 放宽 factory 要求。`ExperimentConfig.validate` 执行 RemoteCLIP 同源检查。

- [ ] **Step 4: 写 CLI 配对构建失败测试**

用 monkeypatch 替换 `open_clip.create_model`、tokenizer 和本地权重加载，断言完整 RemoteCLIP 仅构建一次，文本向量来自同一实例，分割模型只持有其视觉塔。

- [ ] **Step 5: 实现配对构建并运行测试**

新增 `_build_model_and_text`。非 RemoteCLIP 配置沿用原生或外部工厂；RemoteCLIP 配置创建一次完整 OpenCLIP 模型、编码文本、抽取视觉塔并构造语义教师。

Run: `python -m pytest tests/test_config.py tests/test_cli.py -q`

Expected: PASS。

### Task 5: 示例配置与中文文档

**Files:**
- Modify: `configs/base.yaml`
- Modify: `configs/houston2013.yaml`
- Modify: `configs/trento.yaml`
- Modify: `configs/muufl.yaml`
- Modify: `README.md`

**Interfaces:**
- Produces: 可离线校验的双教师配置格式和本地权重接入说明。

- [ ] **Step 1: 更新四份配置**

将 `teacher_encoder` 替换成两个教师字段，将损失权重替换成：

```yaml
structure_teacher_weight: 1.0
semantic_teacher_weight: 1.0
```

示例数据集配置继续使用 `native` 以保持无权重时可运行；README 给出 HyperSIGMA、DINOv3、RemoteCLIP 的完整本地配置片段。

- [ ] **Step 2: 更新 README**

说明三种适配器的必需接口、RemoteCLIP 同源约束、双教师损失、冻结策略和 `224×224 / batch=2 / AMP` 的显存建议。

- [ ] **Step 3: 校验所有示例配置**

Run: `python -m hsi_lidar_ovseg.cli validate-config configs/houston2013.yaml --skip-file-checks`

对 Trento 和 MUUFL 重复执行，Expected: 全部退出码 0。

### Task 6: 全量验证

**Files:**
- Verify only.

**Interfaces:**
- Consumes: Tasks 1–5 的完整实现。
- Produces: 可交付的测试、静态检查和构建证据。

- [ ] **Step 1: 运行完整测试**

Run: `python -m pytest -q`

Expected: 所有测试通过，零失败。

- [ ] **Step 2: 运行 Ruff**

Run: `python -m ruff check .`

Expected: `All checks passed!`

- [ ] **Step 3: 构建包**

Run: `python -m build`

Expected: wheel 和 sdist 成功生成。

- [ ] **Step 4: 检查工作区范围**

Run: `git status --short` 和 `git diff --check`

Expected: 仅包含本计划涉及的代码、测试、配置、文档和先前生成的框架图，无空白错误。
