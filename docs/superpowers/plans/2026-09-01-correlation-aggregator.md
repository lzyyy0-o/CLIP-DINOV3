# 类别维文本引导相关性聚合器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持动态类别输出的前提下，使用两层类别共享的空间窗口与文本引导类别注意力增强相关性解码器。

**Architecture:** 每个尺度的联合/CLIP 文本相关图被嵌入为 `[B,C,N,H,W]` 成本体。两个共享聚合层按“非移位窗口、移位窗口、文本引导类别注意力”的顺序处理成本体，再由现有类别共享 FPN 输出 logits。

**Tech Stack:** Python 3.10+, PyTorch, pytest, Ruff。

**Spec:** `docs/superpowers/specs/2026-09-01-correlation-aggregator-design.md`

## Global Constraints

- 不改动 Shared Lite-ViT、MMFB、CMFEB、CLIP 词表切换协议、损失或数据集加载。
- 输出固定为 `[B,N,H,W]`，`N` 为运行时类别数。
- 固定 2 个聚合层，隐藏维 64，窗口 7，空间和类别注意力均 4 头。
- 每个网格高宽必须能被 7 整除，否则抛出 `ValueError`。
- 所有新参数不得与类别数绑定，必须支持从 10 个 seen 类切换至 15 个测试类。

---

### Task 1: 指定解码器的新行为

**Files:**
- Modify: `tests/test_correlation_decoder.py`

**Interfaces:**
- Consumes: `TextCorrelationDecoder(feature_dim=512, hidden_dim=64)`。
- Produces: 成本体梯度、动态类别数和窗口校验的行为测试。

- [ ] **Step 1: 写入 14×14 真实网格的失败测试**

```python
def test_correlation_decoder_aggregates_14_by_14_cost_volumes() -> None:
    decoder = TextCorrelationDecoder(feature_dim=512, hidden_dim=64)
    joint = tuple(torch.randn(1, 512, s, s, requires_grad=True) for s in (56, 28, 14, 7))
    clip = tuple(torch.randn(1, 512, s, s, requires_grad=True) for s in (56, 28, 14, 7))
    logits = decoder(joint, clip, functional.normalize(torch.randn(10, 2, 512), dim=-1), (224, 224))
    logits.mean().backward()
    assert logits.shape == (1, 10, 224, 224)
    assert joint[2].grad is not None
    assert any(parameter.grad is not None for parameter in decoder.parameters())
    assert len(decoder.aggregators) == 2
```

- [ ] **Step 2: 运行并确认红灯**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_correlation_decoder.py::test_correlation_decoder_aggregates_14_by_14_cost_volumes -v`

Expected: FAIL with `AttributeError` because `TextCorrelationDecoder.aggregators` does not yet exist.

- [ ] **Step 3: 写入动态类别和非法网格测试**

```python
def test_correlation_decoder_keeps_class_axis_dynamic_after_aggregation() -> None:
    decoder = TextCorrelationDecoder(feature_dim=512, hidden_dim=64)
    features = tuple(torch.randn(1, 512, s, s) for s in (56, 28, 14, 7))
    assert decoder(features, features, functional.normalize(torch.randn(10, 2, 512), dim=-1), (224, 224)).shape == (1, 10, 224, 224)
    assert decoder(features, features, functional.normalize(torch.randn(15, 2, 512), dim=-1), (224, 224)).shape == (1, 15, 224, 224)

def test_correlation_decoder_rejects_non_divisible_window_grid() -> None:
    decoder = TextCorrelationDecoder(feature_dim=512, hidden_dim=64)
    invalid = tuple(torch.randn(1, 512, s, s) for s in (56, 28, 14, 8))
    with pytest.raises(ValueError, match="7"):
        decoder(invalid, invalid, functional.normalize(torch.randn(10, 2, 512), dim=-1), (224, 224))
```

- [ ] **Step 4: 运行测试并提交红灯状态**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_correlation_decoder.py -v`

Expected: FAIL because the new `aggregators` attribute and the 7-window validation are absent; it must not fail due to import, spelling, or environment errors.

Commit: `git add tests/test_correlation_decoder.py; git commit -m "test: specify correlation aggregation"`

### Task 2: 创建聚合器模块

**Files:**
- Create: `src/hsi_lidar_ovseg/models/correlation_aggregator.py`
- Modify: `tests/test_correlation_decoder.py`

**Interfaces:**
- Consumes: `cost: Tensor` `[B,C,N,H,W]` 与 `text: Tensor` `[N,P,512]`。
- Produces: `CorrelationAggregatorLayer(hidden_dim=64, text_dim=512, num_heads=4, window_size=7)`，返回相同形状成本体。

- [ ] **Step 1: 写模块级失败测试**

```python
from hsi_lidar_ovseg.models.correlation_aggregator import CorrelationAggregatorLayer

def test_correlation_aggregator_preserves_dynamic_cost_volume_shape() -> None:
    layer = CorrelationAggregatorLayer(hidden_dim=64, text_dim=512, num_heads=4, window_size=7)
    cost = torch.randn(1, 64, 15, 14, 14, requires_grad=True)
    output = layer(cost, functional.normalize(torch.randn(15, 2, 512), dim=-1))
    output.square().mean().backward()
    assert output.shape == cost.shape
    assert cost.grad is not None
```

- [ ] **Step 2: 验证红灯**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_correlation_decoder.py::test_correlation_aggregator_preserves_dynamic_cost_volume_shape -v`

Expected: FAIL with `ModuleNotFoundError` for `correlation_aggregator`。

- [ ] **Step 3: 实现最小聚合器**

```python
class CorrelationAggregatorLayer(nn.Module):
    def __init__(self, hidden_dim: int, text_dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.spatial = SpatialWindowAggregator(hidden_dim, num_heads, window_size)
        self.classes = TextGuidedClassAggregator(hidden_dim, text_dim, num_heads)

    def forward(self, cost: Tensor, text: Tensor) -> Tensor:
        return self.classes(self.spatial(cost), text)
```

`SpatialWindowAggregator` 顺序运行非移位和 `window_size // 2` 移位注意力，利用 mask 阻止循环移位跨图像边界；两个块均为 pre-norm、残差、GELU MLP。`TextGuidedClassAggregator` 沿 `N` 维注意力，`text.mean(dim=1)` 的线性投影拼入 Q/K，V 仅来自成本 token。

- [ ] **Step 4: 验证绿灯并提交**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_correlation_decoder.py::test_correlation_aggregator_preserves_dynamic_cost_volume_shape -v`

Expected: PASS。

Commit: `git add src/hsi_lidar_ovseg/models/correlation_aggregator.py tests/test_correlation_decoder.py; git commit -m "feat: add text guided correlation aggregator"`

### Task 3: 接入动态类别解码器

**Files:**
- Modify: `src/hsi_lidar_ovseg/models/correlation_decoder.py:28-96`
- Modify: `tests/test_correlation_decoder.py`

**Interfaces:**
- Consumes: `CorrelationAggregatorLayer` 与现有两路相关性图。
- Produces: 原签名 `TextCorrelationDecoder.forward(joint_features, clip_features, text_features, output_size) -> Tensor`。

- [ ] **Step 1: 确认集成前解码器测试仍为红灯**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_correlation_decoder.py -v`

Expected: FAIL，因为 `TextCorrelationDecoder` 尚未调用聚合器。

- [ ] **Step 2: 接入两个共享聚合层**

```python
self.hidden_dim = hidden_dim
self.aggregators = nn.ModuleList(CorrelationAggregatorLayer(hidden_dim, 512, 4, 7) for _ in range(2))
cost = embedding(pair.reshape(batch * classes, 2, height, width))
cost = cost.reshape(batch, classes, self.hidden_dim, height, width).permute(0, 2, 1, 3, 4)
for aggregator in self.aggregators:
    cost = aggregator(cost, text_features)
levels.append(cost.permute(0, 2, 1, 3, 4).reshape(batch * classes, self.hidden_dim, height, width))
```

将 `_pyramid()` 改为 `(56,28,14,7)`，把原动态测试改为 10/15 类，并保留参数总数不随类别数增长的断言。

- [ ] **Step 3: 验证解码器、Ruff 并提交**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_correlation_decoder.py -v`

Expected: PASS。

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m ruff check src/hsi_lidar_ovseg/models/correlation_aggregator.py src/hsi_lidar_ovseg/models/correlation_decoder.py tests/test_correlation_decoder.py`

Expected: `All checks passed!`。

Commit: `git add src/hsi_lidar_ovseg/models/correlation_decoder.py tests/test_correlation_decoder.py; git commit -m "feat: aggregate text correlation cost volumes"`

### Task 4: 全模型回归

**Files:**
- Test: `tests/test_clip_guided_model.py`
- Test: `tests/test_openai_clip.py`
- Test: `tests/`

**Interfaces:**
- Consumes: 集成后的动态类别解码器。
- Produces: CLIP 引导、全仓库、Ruff 与 Houston 配置验证证据。

- [ ] **Step 1: 运行 CLIP 引导模型回归**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest tests/test_clip_guided_model.py tests/test_openai_clip.py -v`

Expected: PASS；动态类别 logits 和 CLIP 冻结行为不变。

- [ ] **Step 2: 运行全套质量检查**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m pytest -q`

Expected: PASS，测试数不少于基线。

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m ruff check src tests`

Expected: `All checks passed!`。

- [ ] **Step 3: 校验 Houston 配置并确认工作树状态**

Run: `D:\miniconda\envs\hsi-lidar\python.exe -m hsi_lidar_ovseg.cli validate-config configs/houston2013_shared_lite_vit_clip_rs_fusion.yaml --skip-file-checks`

Expected: 配置有效，仍使用 `clip_guided_shared_lite_vit`、224 tile 与冻结 CLIP。

Run: `git status --short`

Expected: 仅包含 Task 3 已提交的实现；本任务没有源代码变更时不创建空提交。
