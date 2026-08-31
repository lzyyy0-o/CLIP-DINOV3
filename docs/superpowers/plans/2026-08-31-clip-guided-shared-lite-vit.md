# CLIP 引导 Shared Lite-ViT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前教师学生 HSI-LiDAR 开放词汇分割网络扩展为无需 DINOv3/RemoteCLIP 蒸馏、由 OpenAI CLIP ViT-B/16 文本条件引导的 Shared Lite-ViT 分割架构。

**Architecture:** 新配置选择 `clip_guided_shared_lite_vit` 后，HSI 和 LiDAR 先进入共享的六层 Lite-ViT token 主干，再通过双向跨注意力 MMFB 与互补融合 CMFEB 生成四个联合阶段 token。OpenAI CLIP ViT-B/16 对伪 RGB 提取四个视觉阶段，并对动态类别文本生成 `[N,P,512]` 提示词特征；每个阶段分别构建 HSI-LiDAR/文本和 CLIP/文本相关图，由类别共享 FPN 输出 `[B,N,H,W]`。

**Tech Stack:** Python 3.10+、PyTorch 2.1+、OpenAI CLIP ViT-B/16 本地权重、PyYAML、NumPy、pytest、ruff。

**Spec:** `docs/superpowers/specs/2026-08-31-clip-guided-shared-lite-vit-design.md`

## Global Constraints

- 保留 `native.yaml`、`pretrained.yaml` 和 `online_vit.yaml` 作为既有基线；新模型只使用新增 YAML。
- 新架构禁止构建 DINOv3、RemoteCLIP、结构教师、语义教师、InfoNCE、门控正则和私有特征正则。
- OpenAI CLIP 必须从本地 ViT-B/16 checkpoint 加载，运行时不下载权重；输入 tile 固定为 `224×224`。
- HSI 仅由训练掩膜统计量归一化；HSI/LiDAR 必须严格配准；HSI 使用光谱适配器，LiDAR 不使用该适配器。
- 新模型的唯一优化目标是掩码多类别交叉熵；未见类像素不参与训练损失。
- 文本在每个训练前向重算且可反向传播；仅推理阶段按“类别列表 + 模板列表”缓存 token 与原型。
- CLIP 仅解冻视觉与文本塔的末两个 Transformer block、对应终端归一化和投影；其学习率为新建模块的十分之一。
- 类别数 `N` 必须由传入的类别名称动态决定，任何可训练卷积不得把 `N` 固定为 Houston、MUUFL 或 Trento 的类别数。
- 不触碰未跟踪的 `.vscode/`、`docs/figures/`、`third_party/`。

---

## File Structure

| 文件 | 责任 |
|---|---|
| `src/hsi_lidar_ovseg/config.py` | 增加架构判别配置、Shared Lite-ViT 与 OpenAI CLIP 配置，并保持旧 YAML 可加载。 |
| `src/hsi_lidar_ovseg/models/shared_lite_vit.py` | 6 层共享 HSI/LiDAR token 编码器与四阶段 token 输出。 |
| `src/hsi_lidar_ovseg/models/vit_fusion.py` | 四阶段 MMFB 双向跨注意力、CMFEB 互补融合和 token 金字塔投影。 |
| `src/hsi_lidar_ovseg/models/openai_clip.py` | 本地 OpenAI CLIP 加载、冻结策略、hook 中间层视觉特征、动态文本特征。 |
| `src/hsi_lidar_ovseg/models/correlation_decoder.py` | 双相关图、类别共享相关性嵌入块和 FPN 解码器。 |
| `src/hsi_lidar_ovseg/models/clip_guided_model.py` | 新架构的端到端 segmentor 与简洁输出类型。 |
| `src/hsi_lidar_ovseg/losses/cross_entropy.py` | 新架构唯一的 seen-class 掩码交叉熵。 |
| `src/hsi_lidar_ovseg/cli.py` | 新模型工厂、本地 CLIP 权重加载、参数分组、动态类别条件和配置选择。 |
| `src/hsi_lidar_ovseg/engine/trainer.py`、`engine/evaluator.py` | 支持 `Tensor | tuple[str, ...]` 类别条件，训练保持文本梯度、评估缓存文本原型。 |
| `configs/shared_lite_vit_clip.yaml` | Houston 2018 主开发配置；其它数据集配置继承其结构字段。 |
| `tests/test_*.py` | 每个新增接口的形状、梯度、冻结、动态类别数、CLI 和训练冒烟覆盖。 |
| `README.md` | 增加权重位置、训练命令、开放词汇划分和结果指标说明。 |

### Task 1: 架构判别配置与开放词汇协议

**Files:**
- Modify: `src/hsi_lidar_ovseg/config.py`
- Modify: `tests/test_config.py`
- Create: `configs/shared_lite_vit_clip.yaml`

**Interfaces:**
- Consumes: 既有 `DataConfig`、`EncoderConfig`、`ModelConfig`、`LossConfig` 与 `load_config(Path)`。
- Produces: `SharedLiteViTConfig`, `OpenAIClipConfig`，以及带 `architecture: Literal["teacher_student", "clip_guided_shared_lite_vit"]` 的 `ModelConfig`；新 YAML 可不提供教师编码器字段。

- [ ] **Step 1: 写出失败的配置测试**

```python
def test_clip_guided_config_accepts_dynamic_vocabulary_without_teachers(tmp_path: Path) -> None:
    path = tmp_path / "clip_guided.yaml"
    path.write_text(_clip_guided_yaml(), encoding="utf-8")

    config = load_config(path, check_files=False)

    assert config.model.architecture == "clip_guided_shared_lite_vit"
    assert config.model.shared_lite_vit.depths == (1, 1, 2, 2)
    assert config.model.clip.model_name == "ViT-B/16"
    assert config.loss.kind == "masked_cross_entropy"


def test_clip_guided_config_rejects_teacher_and_non_224_tile(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(_clip_guided_yaml(tile_size=192, include_teacher=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="224|教师"):
        load_config(path, check_files=False)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_config.py -q`

Expected: FAIL，因为 `architecture`、`shared_lite_vit`、`clip` 与 `loss.kind` 尚不存在。

- [ ] **Step 3: 实现严格配置解码**

在 `config.py` 中增加如下类型，并在 `_decode_model` 中仅在配置键存在时调用 `_decode_encoder`：

```python
@dataclass(frozen=True)
class SharedLiteViTConfig:
    patch_size: int = 16
    embed_dim: int = 384
    depths: tuple[int, int, int, int] = (1, 1, 2, 2)
    num_heads: int = 6
    mlp_ratio: float = 2.0


@dataclass(frozen=True)
class OpenAIClipConfig:
    checkpoint: Path
    model_name: str = "ViT-B/16"
    feature_blocks: tuple[int, int, int, int] = (2, 5, 8, 11)
    unfreeze_blocks: int = 2
```

使 `ModelConfig` 对旧架构继续保存旧编码器字段，对新架构要求 `shared_lite_vit` 和 `clip`，并拒绝新架构携带四个教师字段。令 `LossConfig.kind` 默认为 `"teacher_student"`；`"masked_cross_entropy"` 时禁止非零蒸馏权重。新 YAML 使用：

```yaml
model:
  architecture: clip_guided_shared_lite_vit
  shared_lite_vit: {patch_size: 16, embed_dim: 384, depths: [1, 1, 2, 2], num_heads: 6, mlp_ratio: 2.0}
  clip: {checkpoint: weights/openai_clip/ViT-B-16.pt, model_name: ViT-B/16, feature_blocks: [2, 5, 8, 11], unfreeze_blocks: 2}
  prompt_templates: ["aerial image of {}", "satellite image of {}", "top-down view of {}", "remote sensing image of {}"]
loss: {kind: masked_cross_entropy}
train: {tile_size: 224, overlap: 56, min_seen_pixels: 1, class_aware_sampling: true, class_aware_fraction: 0.7, validation_fraction: 0.1, early_stopping_patience: 20, early_stopping_min_delta: 0.0, cosine_eta_min: 0.000001, batch_size: 1, epochs: 100, learning_rate: 0.0001, backbone_learning_rate: 0.00001, weight_decay: 0.01, gradient_clip: 1.0, amp: true, device: cuda}
```

验证 `depths==(1,1,2,2)`、`sum(depths)==6`、`embed_dim==384`、`num_heads==6`、`patch_size==16`、`feature_blocks==(2,5,8,11)`、`unfreeze_blocks==2`，并要求新架构 `text_dim==512`、`train.tile_size==224`。

- [ ] **Step 4: 运行配置测试**

Run: `pytest tests/test_config.py -q`

Expected: PASS，且既有 YAML 均可被 `load_config(..., check_files=False)` 加载。

- [ ] **Step 5: 提交配置基础**

```bash
git add src/hsi_lidar_ovseg/config.py tests/test_config.py configs/shared_lite_vit_clip.yaml
git commit -m "feat: add CLIP guided architecture configuration"
```

### Task 2: 共享六层 Lite-ViT token 编码器

**Files:**
- Create: `src/hsi_lidar_ovseg/models/shared_lite_vit.py`
- Modify: `src/hsi_lidar_ovseg/models/__init__.py`
- Create: `tests/test_shared_lite_vit.py`

**Interfaces:**
- Consumes: `SharedLiteViTConfig` 与 NCHW HSI/LiDAR 张量。
- Produces: `SharedLiteViT.forward(hsi, lidar) -> SharedTokenOutput`，其中 `hsi_tokens`、`lidar_tokens` 为长度为 4 的 tuple，每项形状 `[B, H/16*W/16, 384]`，并公开 `grid_size: tuple[int, int]`。

- [ ] **Step 1: 写出失败的 token 形状与共享权重测试**

```python
def test_shared_lite_vit_has_six_shared_blocks_and_four_stages() -> None:
    model = SharedLiteViT(hsi_bands=6, lidar_channels=3)
    output = model(torch.randn(2, 6, 224, 224), torch.randn(2, 3, 224, 224))

    assert len(model.blocks) == 6
    assert model.spectral_adapter is not None
    assert model.lidar_patch_embed is not None
    assert output.grid_size == (14, 14)
    assert [item.shape for item in output.hsi_tokens] == [(2, 196, 384)] * 4
    assert [item.shape for item in output.lidar_tokens] == [(2, 196, 384)] * 4


def test_shared_lite_vit_rejects_unaligned_inputs() -> None:
    model = SharedLiteViT(hsi_bands=4, lidar_channels=1)
    with pytest.raises(ValueError, match="16|空间"):
        model(torch.randn(1, 4, 224, 224), torch.randn(1, 1, 208, 224))
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_shared_lite_vit.py -q`

Expected: FAIL，因为 `SharedLiteViT` 与 `SharedTokenOutput` 尚未定义。

- [ ] **Step 3: 实现最小共享主干**

实现 `SharedTokenOutput` dataclass 和 `SharedLiteViT`。HSI 先经 `Conv2d(hsi_bands,384,1)`，LiDAR 不经过光谱适配器；两者用不同 patch embedding 进入同一个 `self.blocks: ModuleList`。每个 block 依次对 HSI token、LiDAR token 调用同一个 block；在 block 索引 `0,1,3,5` 保存阶段 token：

```python
for index, block in enumerate(self.blocks):
    hsi = block(hsi)
    lidar = block(lidar)
    if index in (0, 1, 3, 5):
        hsi_stages.append(self.norm(hsi))
        lidar_stages.append(self.norm(lidar))
```

为两个模态分别保留可插值二维位置编码，要求两输入的 batch 与空间尺寸相同、且高宽均能被 16 整除。

- [ ] **Step 4: 运行单元测试与风格检查**

Run: `pytest tests/test_shared_lite_vit.py -q && ruff check src/hsi_lidar_ovseg/models/shared_lite_vit.py`

Expected: PASS。

- [ ] **Step 5: 提交共享主干**

```bash
git add src/hsi_lidar_ovseg/models/shared_lite_vit.py src/hsi_lidar_ovseg/models/__init__.py tests/test_shared_lite_vit.py
git commit -m "feat: add shared six-layer Lite-ViT encoder"
```

### Task 3: ViT-MMFB、ViT-CMFEB 与 token 金字塔

**Files:**
- Create: `src/hsi_lidar_ovseg/models/vit_fusion.py`
- Create: `tests/test_vit_fusion.py`

**Interfaces:**
- Consumes: 两个长度为 4、每项 `[B,L,384]` 的 token tuple 与 `grid_size`。
- Produces: `ViTMMFB.forward(...) -> tuple[TokenPyramid, TokenPyramid]`，`ViTCMFEB.forward(...) -> TokenPyramid`，`TokenPyramidProjector.forward(tokens, grid_size, image_size) -> FeaturePyramid`，输出通道均为 512、空间尺度为 `1/4,1/8,1/16,1/32`。

- [ ] **Step 1: 写出失败的融合测试**

```python
def test_mmfb_cmfeb_and_projector_preserve_four_stages() -> None:
    hsi = tuple(torch.randn(1, 196, 384) for _ in range(4))
    lidar = tuple(torch.randn(1, 196, 384) for _ in range(4))
    hsi_updated, lidar_updated = ViTMMFB(embed_dim=384, num_heads=6)(hsi, lidar)
    joint = ViTCMFEB(embed_dim=384, num_heads=6)(hsi_updated, lidar_updated)
    maps = TokenPyramidProjector(384, 512)(joint, grid_size=(14, 14), image_size=(224, 224))

    assert len(joint) == len(maps) == 4
    assert [item.shape for item in maps] == [(1, 512, 56, 56), (1, 512, 28, 28), (1, 512, 14, 14), (1, 512, 7, 7)]
    assert not torch.equal(hsi_updated[0], hsi[0])
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_vit_fusion.py -q`

Expected: FAIL，因为三种融合模块尚未定义。

- [ ] **Step 3: 实现阶段独立但类别无关的融合模块**

每个阶段使用独立权重。`CrossModalBlock` 必须执行双向交叉注意力、残差和 FFN：

```python
hsi_out = hsi + self.hsi_attn(self.hsi_norm(hsi), self.lidar_norm(lidar), self.lidar_norm(lidar))[0]
lidar_out = lidar + self.lidar_attn(self.lidar_norm(lidar), self.hsi_norm(hsi), self.hsi_norm(hsi))[0]
return hsi_out + self.hsi_mlp(self.hsi_ffn_norm(hsi_out)), lidar_out + self.lidar_mlp(self.lidar_ffn_norm(lidar_out))
```

`ComplementaryFusionBlock` 将 `[hsi,lidar]` 在最后维拼接，经 `Linear(768,384)`、自注意力残差与 GELU MLP 得到 joint token。`TokenPyramidProjector` 把 token 重排为 `[B,C,14,14]`，每阶段经独立 `1×1` 投影后以双线性插值变为四个目标尺寸；验证 `L == grid_h*grid_w`，不匹配时抛出 `ValueError`。

- [ ] **Step 4: 运行融合测试**

Run: `pytest tests/test_vit_fusion.py -q && ruff check src/hsi_lidar_ovseg/models/vit_fusion.py`

Expected: PASS。

- [ ] **Step 5: 提交跨模态融合**

```bash
git add src/hsi_lidar_ovseg/models/vit_fusion.py tests/test_vit_fusion.py
git commit -m "feat: add ViT multimodal fusion blocks"
```

### Task 4: 可微调的本地 OpenAI CLIP 引导器

**Files:**
- Create: `src/hsi_lidar_ovseg/models/openai_clip.py`
- Modify: `src/hsi_lidar_ovseg/models/__init__.py`
- Modify: `src/hsi_lidar_ovseg/cli.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_openai_clip.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: 本地 OpenAI `ViT-B-16.pt`、`OpenAIClipConfig`、`tuple[str, ...]` 类别名和 NCHW 伪 RGB。
- Produces: `load_openai_clip(checkpoint: Path) -> tuple[nn.Module, Tokenizer]`；`OpenAIClipGuidance.visual_features(rgb) -> FeaturePyramid`；`OpenAIClipGuidance.text_features(class_names) -> Tensor`，形状 `[N,P,512]`。

- [ ] **Step 1: 写出失败的视觉/文本梯度与冻结测试**

```python
def test_openai_clip_guidance_keeps_prompt_axis_and_unfreezes_only_last_two() -> None:
    clip = FakeOpenAIClip(width=768, text_dim=512, blocks=12)
    guidance = OpenAIClipGuidance(clip, fake_tokenize, (2, 5, 8, 11), TEMPLATES, unfreeze_blocks=2)

    text = guidance.text_features(("trees", "road", "water"))
    maps = guidance.visual_features(torch.randn(1, 3, 224, 224))
    (text.sum() + sum(item.mean() for item in maps)).backward()

    assert text.shape == (3, 4, 512)
    assert [item.shape for item in maps] == [(1, 512, 56, 56), (1, 512, 28, 28), (1, 512, 14, 14), (1, 512, 7, 7)]
    assert clip.visual.transformer.resblocks[9].weight.grad is None
    assert clip.visual.transformer.resblocks[10].weight.grad is not None
    assert clip.transformer.resblocks[10].weight.grad is not None
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_openai_clip.py tests/test_cli.py -q`

Expected: FAIL，因为 `OpenAIClipGuidance` 与 `load_openai_clip` 尚未定义。

- [ ] **Step 3: 实现本地加载、hook 与部分微调**

使用 OpenAI 官方 `clip` 包加载本地文件，禁止传入模型名触发下载：

```python
def load_openai_clip(checkpoint: Path) -> tuple[nn.Module, Tokenizer]:
    import clip
    if not checkpoint.is_file():
        raise ConfigError(f"OpenAI CLIP checkpoint 不存在: {checkpoint}")
    model, _ = clip.load(str(checkpoint), device="cpu", jit=False)
    return model.float(), clip.tokenize
```

`OpenAIClipGuidance` 只注册完整 CLIP 模型一次。向 `visual.transformer.resblocks[2,5,8,11]` 注册 forward hook，在 `encode_image` 后把 `[L,B,C]` 输出转换为 `[B,C,14,14]`，用四个独立投影映射到 512 并生成四尺度特征。`text_features` 以模板优先、类别次级的顺序 token 化，执行 `model.encode_text(tokens)`，reshape 为 `[N,P,512]` 并在最后维 L2 归一化；禁止 `torch.no_grad()`。

实现 `configure_partial_finetune(model, unfreeze_blocks=2)`：先 `requires_grad_(False)`，再解冻视觉 `resblocks[10:]`、`visual.ln_post`、`visual.proj`，文本 `transformer.resblocks[10:]`、`ln_final`、`text_projection`；任何缺失模块或 block 数不是 12 都报出明确错误。把 OpenAI CLIP 的 Git 依赖作为 `pretrained` extras 的明确直接依赖，并在 README 中给出本地安装命令。

- [ ] **Step 4: 运行 CLIP 测试**

Run: `pytest tests/test_openai_clip.py tests/test_cli.py -q && ruff check src/hsi_lidar_ovseg/models/openai_clip.py src/hsi_lidar_ovseg/cli.py`

Expected: PASS；测试不得访问网络或加载真实 checkpoint。

- [ ] **Step 5: 提交 CLIP 引导器**

```bash
git add pyproject.toml src/hsi_lidar_ovseg/models/openai_clip.py src/hsi_lidar_ovseg/models/__init__.py src/hsi_lidar_ovseg/cli.py tests/test_openai_clip.py tests/test_cli.py
git commit -m "feat: add trainable local OpenAI CLIP guidance"
```

### Task 5: 双相关图与类别共享 FPN 解码器

**Files:**
- Create: `src/hsi_lidar_ovseg/models/correlation_decoder.py`
- Create: `tests/test_correlation_decoder.py`

**Interfaces:**
- Consumes: `joint_features: FeaturePyramid`、`clip_features: FeaturePyramid`（四层 `[B,512,H,W]`）及 `text_features: Tensor`（`[N,P,512]`）。
- Produces: `TextCorrelationDecoder.forward(...) -> Tensor`，形状 `[B,N,H_out,W_out]`。

- [ ] **Step 1: 写出失败的动态类别数测试**

```python
def test_correlation_decoder_supports_runtime_class_counts() -> None:
    decoder = TextCorrelationDecoder(feature_dim=512, hidden_dim=64)
    pyramid = _pyramid(batch=1, channels=512)
    parameters_before = sum(parameter.numel() for parameter in decoder.parameters())
    two_class_logits = decoder(pyramid, pyramid, torch.nn.functional.normalize(torch.randn(2, 4, 512), dim=-1), (224, 224))
    seven_class_logits = decoder(pyramid, pyramid, torch.nn.functional.normalize(torch.randn(7, 4, 512), dim=-1), (224, 224))

    assert two_class_logits.shape == (1, 2, 224, 224)
    assert seven_class_logits.shape == (1, 7, 224, 224)
    assert sum(parameter.numel() for parameter in decoder.parameters()) == parameters_before


def test_correlation_decoder_rejects_text_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="512"):
        TextCorrelationDecoder(512, 64)(_pyramid(1, 512), _pyramid(1, 512), torch.randn(3, 4, 256), (224, 224))
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_correlation_decoder.py -q`

Expected: FAIL，因为相关性解码器尚未定义。

- [ ] **Step 3: 实现相关性与共享解码**

对每尺度规范化特征与文本进行以下计算，再沿提示词维平均：

```python
joint_corr = torch.einsum("bchw,npc->bnphw", normalize(joint, dim=1), text_features).mean(dim=2)
clip_corr = torch.einsum("bchw,npc->bnphw", normalize(clip, dim=1), text_features).mean(dim=2)
pair = torch.stack((joint_corr, clip_corr), dim=2).reshape(batch * classes, 2, height, width)
```

每个尺度将 `pair` 送入共享 `CorrelationEmbeddingBlock(2,64)`；从最粗尺度开始用相同类别共享的上采样/侧连 FPN 逐级融合，最后输出单通道，reshape 回 `[B,N,H,W]` 并插值到 `output_size`。实现中严禁创建依赖 `classes` 的 `nn.Conv2d`。检查两套 feature pyramid 的四层长度、每层空间形状一致、文本维度为 512，错误信息应指出对应条件。

- [ ] **Step 4: 运行解码器测试**

Run: `pytest tests/test_correlation_decoder.py -q && ruff check src/hsi_lidar_ovseg/models/correlation_decoder.py`

Expected: PASS。

- [ ] **Step 5: 提交相关性解码器**

```bash
git add src/hsi_lidar_ovseg/models/correlation_decoder.py tests/test_correlation_decoder.py
git commit -m "feat: add class-conditioned correlation decoder"
```

### Task 6: 端到端新 segmentor 与唯一 CE 目标

**Files:**
- Create: `src/hsi_lidar_ovseg/models/clip_guided_model.py`
- Modify: `src/hsi_lidar_ovseg/models/__init__.py`
- Create: `src/hsi_lidar_ovseg/losses/cross_entropy.py`
- Modify: `src/hsi_lidar_ovseg/losses/__init__.py`
- Create: `tests/test_clip_guided_model.py`
- Modify: `tests/test_losses.py`

**Interfaces:**
- Consumes: `SharedLiteViT`, `ViTMMFB`, `ViTCMFEB`, `TokenPyramidProjector`, `OpenAIClipGuidance`, `TextCorrelationDecoder`、标签 `[B,H,W]`、已见类别原始编号。
- Produces: `CLIPGuidedSharedLiteViTSegmentor.forward(hsi,lidar,pseudo_rgb,class_names) -> ClipGuidedSegmentationOutput(logits)` 和 `MaskedCrossEntropyObjective.forward(output,labels,valid_mask) -> {"total": Tensor, "segmentation": Tensor}`。

- [ ] **Step 1: 写出失败的端到端测试**

```python
def test_clip_guided_segmentor_backpropagates_to_text_and_shared_backbone() -> None:
    model = _model_with_fake_clip()
    output = model(torch.randn(1, 6, 224, 224), torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224), ("trees", "road", "water"))
    losses = MaskedCrossEntropyObjective((1, 2))(output, _labels(), torch.ones(1, 224, 224, dtype=torch.bool))
    losses["total"].backward()

    assert output.logits.shape == (1, 3, 224, 224)
    assert set(losses) == {"total", "segmentation"}
    assert model.shared_encoder.blocks[-1].self_attn.in_proj_weight.grad is not None
    assert model.clip_guidance.model.text_projection.grad is not None


def test_masked_cross_entropy_ignores_unseen_labels() -> None:
    output = ClipGuidedSegmentationOutput(logits=torch.randn(1, 3, 2, 2, requires_grad=True))
    loss = MaskedCrossEntropyObjective((1, 2))(output, torch.tensor([[[1, 3], [2, 3]]]), torch.ones(1, 2, 2, dtype=torch.bool))
    assert torch.isfinite(loss["total"])
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_clip_guided_model.py tests/test_losses.py -q`

Expected: FAIL，因为新 segmentor 和 CE objective 尚未定义。

- [ ] **Step 3: 连接模型并实现损失**

`CLIPGuidedSharedLiteViTSegmentor.forward` 按严格顺序执行：共享 token 编码器 → MMFB → CMFEB → joint pyramid projector；并行调用 CLIP visual features；调用 CLIP text features；最后调用相关性解码器。检查三种输入的 NCHW、相同 batch/空间、且大小为 `224×224`。输出 dataclass 只含 `logits`，不含教师特征、gate 或像素 embedding。

`MaskedCrossEntropyObjective` 必须仅对 `valid_mask & torch.isin(labels, seen_class_ids)` 计算交叉熵，并把原始标签 `1..N` 转为通道索引 `0..N-1`：

```python
pixel_logits = output.logits.permute(0, 2, 3, 1)[supervised_mask]
targets = labels[supervised_mask] - 1
segmentation = functional.cross_entropy(pixel_logits, targets)
return {"total": segmentation, "segmentation": segmentation}
```

保留旧 `OpenVocabularyObjective` 供教师学生基线使用；新 CLI 架构只能选择 `MaskedCrossEntropyObjective`。

- [ ] **Step 4: 运行模型与损失测试**

Run: `pytest tests/test_clip_guided_model.py tests/test_losses.py -q && ruff check src/hsi_lidar_ovseg/models/clip_guided_model.py src/hsi_lidar_ovseg/losses/cross_entropy.py`

Expected: PASS。

- [ ] **Step 5: 提交新模型和损失**

```bash
git add src/hsi_lidar_ovseg/models/clip_guided_model.py src/hsi_lidar_ovseg/models/__init__.py src/hsi_lidar_ovseg/losses/cross_entropy.py src/hsi_lidar_ovseg/losses/__init__.py tests/test_clip_guided_model.py tests/test_losses.py
git commit -m "feat: add CLIP guided segmentor and CE objective"
```

### Task 7: CLI、训练器、评估器与优化器集成

**Files:**
- Modify: `src/hsi_lidar_ovseg/cli.py`
- Modify: `src/hsi_lidar_ovseg/engine/trainer.py`
- Modify: `src/hsi_lidar_ovseg/engine/evaluator.py`
- Modify: `src/hsi_lidar_ovseg/engine/checkpoint.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_training_smoke.py`
- Modify: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `ExperimentConfig`、`CLIPGuidedSharedLiteViTSegmentor`、`tuple[str,...] class_names` 和旧 `Tensor text_embeddings` 基线条件。
- Produces: `_build_model_and_conditioning(config,hsi_bands) -> tuple[nn.Module, Tensor | tuple[str,...]]`；`Trainer` 和 `sliding_window_predict` 在新架构训练时不 detach 类别名称，评估时缓存文本原型。

- [ ] **Step 1: 写出失败的分组与冒烟测试**

```python
def test_clip_guided_optimizer_uses_expected_learning_rates() -> None:
    model = _model_with_fake_clip()
    optimizer = _optimizer(model, _clip_guided_config())

    assert {group["lr"] for group in optimizer.param_groups} == {1e-4, 1e-5}
    assert all(not parameter.requires_grad for parameter in model.clip_guidance.frozen_parameters())


def test_clip_guided_trainer_recomputes_text_during_training() -> None:
    model, conditioning = _model_and_class_names()
    trainer = Trainer(model, MaskedCrossEntropyObjective((1, 2)), _optimizer(model, _config()), conditioning, device=torch.device("cpu"), gradient_clip=1.0, amp=False)

    trainer.train_step(_batch_224())

    assert model.clip_guidance.text_forward_calls == 1
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_cli.py tests/test_training_smoke.py tests/test_checkpoint.py -q`

Expected: FAIL，因为 CLI 仍构建教师学生模型、Trainer 仍强制 detach 二维文本张量。

- [ ] **Step 3: 实现架构分派和条件生命周期**

在 CLI 把旧 `_build_model_and_text` 分成：

```python
def _build_model_and_conditioning(config: ExperimentConfig, hsi_bands: int) -> tuple[nn.Module, Tensor | tuple[str, ...]]:
    if config.model.architecture == "clip_guided_shared_lite_vit":
        return _build_clip_guided_model(config, hsi_bands), config.data.class_names
    return _build_teacher_student_model_and_text(config, hsi_bands)
```

`_optimizer` 按参数名/模块身份构造两组而非旧的泛化 encoder 规则：新模块为 `1e-4`，可训练 CLIP 参数为 `1e-5`，冻结参数不加入 optimizer。`Trainer.__init__` 接受 `conditioning: Tensor | tuple[str, ...]`；仅当它是 Tensor 时 `.detach().to(device)`，字符串 tuple 原样保存。`train_step` 将该条件直接传给模型，从而每个 step 调用文本塔。评估器在 `model.eval()` 前通过 `model.cache_text_features(class_names)` 生成可复用张量，滑窗循环使用缓存张量；结束后调用 `clear_text_cache()`，避免跨词表污染。

为 checkpoint identity 添加 `architecture` 与 CLIP 模型名，确保教师学生 checkpoint 不能载入新模型。现有旧基线训练、恢复与评估路径必须不改变行为。

- [ ] **Step 4: 运行集成测试**

Run: `pytest tests/test_cli.py tests/test_training_smoke.py tests/test_checkpoint.py -q`

Expected: PASS；CPU fake-CLIP 冒烟训练能完成一次反向传播，文本塔调用次数等于训练 step 数。

- [ ] **Step 5: 提交运行时集成**

```bash
git add src/hsi_lidar_ovseg/cli.py src/hsi_lidar_ovseg/engine/trainer.py src/hsi_lidar_ovseg/engine/evaluator.py src/hsi_lidar_ovseg/engine/checkpoint.py tests/test_cli.py tests/test_training_smoke.py tests/test_checkpoint.py
git commit -m "feat: train and evaluate CLIP guided architecture"
```

### Task 8: 数据集开放词汇配置、文档与全量验证

**Files:**
- Modify: `configs/houston2013.yaml`
- Modify: `configs/muufl.yaml`
- Modify: `configs/trento.yaml`
- Modify: `README.md`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: 已实现的 `DataConfig.seen_class_ids`/`unseen_class_ids` 与新模型 YAML。
- Produces: 三个可复现的 seen/unseen 协议、下载/权重路径说明和可运行命令。

- [ ] **Step 1: 写出失败的协议配置测试**

```python
@pytest.mark.parametrize("name", ["houston2013.yaml", "muufl.yaml", "trento.yaml", "shared_lite_vit_clip.yaml"])
def test_published_open_vocabulary_configs_cover_each_class_once(name: str) -> None:
    config = load_config(Path("configs") / name, check_files=False)
    assert set(config.data.seen_class_ids).isdisjoint(config.data.unseen_class_ids)
    assert set(config.data.seen_class_ids) | set(config.data.unseen_class_ids) == set(range(1, config.data.num_classes + 1))
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_config.py -q`

Expected: FAIL，直到三个公开基准的 seen/unseen 划分与新主配置均严格通过配置校验。

- [ ] **Step 3: 写入固定实验协议和使用文档**

将 Houston 2013 固定为 10 seen / 5 unseen，MUUFL 固定为 7 / 4，Trento 固定为 4 / 2；保持类名与原始标签顺序一一对应。README 添加：

```bash
pip install -e ".[pretrained,dev]"
pip install git+https://github.com/openai/CLIP.git
hsi-lidar-ovseg validate-config configs/shared_lite_vit_clip.yaml --skip-file-checks
hsi-lidar-ovseg train configs/shared_lite_vit_clip.yaml
```

说明权重路径 `weights/openai_clip/ViT-B-16.pt`、Houston 2018 下载许可要求、训练期未见标签忽略规则、以及 `mIoU_seen`、`mIoU_unseen`、`hIoU` 的报告方式。明确 Houston 2018 是主开发集，Houston 2013/MUUFL/Trento 采用独立训练和官方测试掩膜；跨数据集试验是单列的域泛化协议。

- [ ] **Step 4: 运行全量验证**

Run: `pytest -q && ruff check src tests && hsi-lidar-ovseg validate-config configs/shared_lite_vit_clip.yaml --skip-file-checks`

Expected: 全部 pytest 通过、ruff 无诊断、新 YAML 通过严格解析；不得联网、不得载入真实大权重。

- [ ] **Step 5: 提交协议、文档与验证结果**

```bash
git add configs/houston2013.yaml configs/muufl.yaml configs/trento.yaml README.md tests/test_config.py
git commit -m "docs: add open-vocabulary benchmark protocol"
```

## Plan Self-Review

- **规格覆盖：** Task 1 实现判别配置和本地 ViT-B/16 约束；Tasks 2–3 实现六层共享主干及 MMFB/CMFEB；Task 4 实现可微调 OpenAI 图文塔；Task 5 贯彻 MM-OVSeg 风格双相关性；Task 6 移除新架构全部蒸馏损失；Task 7 覆盖训练、滑窗评估、缓存与 checkpoint；Task 8 固定公开数据集开放词汇协议、文档与全量回归。
- **占位符检查：** 本计划没有未定义的后续工作项；每项实现步骤都给出类名、签名、运行命令与预期结果。
- **类型一致性：** 新模型唯一条件类型为 `tuple[str, ...]`，旧模型保留 `Tensor` 条件；相关性解码器输入固定为四层 512 通道金字塔、文本 `[N,P,512]`、输出 `[B,N,H,W]`；新 objective 的输出键固定为 `total` 和 `segmentation`。
