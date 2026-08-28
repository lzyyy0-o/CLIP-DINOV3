# HSI–LiDAR 开放词汇语义分割实施计划

> **面向智能体执行者：**必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行本计划。每一步使用复选框跟踪。

**目标：**构建一个可安装、可测试的 PyTorch 项目，对 Houston 2013、Trento 和 MUUFL 的配准 HSI–LiDAR 栅格执行开放词汇语义分割。

**架构：**数据层把不同文件格式统一为经过严格校验的场景数组，并以配对图块训练、滑窗整图推理。模型层通过可插拔的 HSI、LiDAR 和语义教师编码器产生特征金字塔，经多层对齐、门控融合和 CLIP 空间解码得到像素类别分数；训练层组合监督、对比和正则损失，并记录可复现检查点。

**技术栈：**Python 3.10+、PyTorch 2.1+、NumPy、SciPy、PyYAML、Ruff、Pytest、可选的 timm/OpenCLIP/HyperSIGMA。

**设计文档：**`docs/superpowers/specs/2026-08-28-hsi-lidar-open-vocabulary-segmentation-design.md`

## 全局约束

- Python 最低版本为 3.10，PyTorch 最低版本为 2.1。
- 采用 `src/` 包布局，公共函数和配置数据类必须具有类型标注。
- 库模块使用 `logging`，不得使用 `print`。
- 模型和数据不得隐式下载；外部权重必须通过本地路径显式提供。
- 测试必须完全离线，使用原生小模型或注入式替身，不依赖真实数据和预训练权重。
- HSI、LiDAR、标签和掩码的空间操作必须共享同一随机参数。
- 标签 `0` 始终作为忽略标签；开放词汇已见/未见划分必须来自配置。
- 新行为一律采用红—绿—重构循环，先观察目标测试因缺少行为而失败，再编写生产代码。

---

### 任务 1：项目骨架与严格配置

**文件：**
- 新建：`pyproject.toml`
- 新建：`.gitignore`
- 新建：`src/hsi_lidar_ovseg/__init__.py`
- 新建：`src/hsi_lidar_ovseg/config.py`
- 新建：`tests/test_config.py`

**接口：**
- 输入：实验 YAML 路径。
- 输出：`load_config(path: Path, *, check_files: bool = True) -> ExperimentConfig`。
- 输出数据类：`DataConfig`、`EncoderConfig`、`ModelConfig`、`LossConfig`、`TrainConfig`、`ExperimentConfig`。

- [ ] **步骤 1：写入配置失败测试**

```python
def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: demo\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown"):
        load_config(path, check_files=False)


def test_config_rejects_overlapping_seen_and_unseen() -> None:
    with pytest.raises(ConfigError, match="overlap"):
        DataConfig(
            name="demo",
            hsi_path=Path("hsi.npy"),
            lidar_path=Path("lidar.npy"),
            labels_path=Path("labels.npy"),
            train_mask_path=Path("train.npy"),
            test_mask_path=Path("test.npy"),
            hsi_key=None,
            lidar_key=None,
            labels_key=None,
            train_mask_key=None,
            test_mask_key=None,
            class_names=("a", "b"),
            seen_class_ids=(1,),
            unseen_class_ids=(1, 2),
            pseudo_rgb_indices=(0, 1, 2),
        )
```

- [ ] **步骤 2：运行测试并确认因模块缺失失败**

运行：`python -m pytest tests/test_config.py -v`

预期：收集阶段因 `hsi_lidar_ovseg.config` 不存在而失败。

- [ ] **步骤 3：实现最小严格配置加载器**

使用冻结数据类保存配置；通过 `dataclasses.fields()` 比较 YAML 键集合，发现未知键时抛出 `ConfigError`。`DataConfig.__post_init__` 校验类别编号、互斥划分、三个伪 RGB 索引和文件后缀；`ExperimentConfig.validate_files()` 只在 `check_files=True` 时检查本地文件及必要权重。

```python
class ConfigError(ValueError):
    """实验配置无效。"""


def load_config(path: Path, *, check_files: bool = True) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("配置根节点必须是映射")
    config = _decode_dataclass(ExperimentConfig, raw, context="root")
    config.validate(check_files=check_files)
    return config
```

- [ ] **步骤 4：运行配置测试并确认通过**

运行：`python -m pytest tests/test_config.py -v`

预期：全部通过。

- [ ] **步骤 5：运行 Ruff 并提交**

运行：`ruff check pyproject.toml src/hsi_lidar_ovseg/config.py tests/test_config.py`

提交：`git commit -m "feat: add strict experiment configuration"`

---

### 任务 2：场景读取与无泄漏预处理

**文件：**
- 新建：`src/hsi_lidar_ovseg/data/__init__.py`
- 新建：`src/hsi_lidar_ovseg/data/io.py`
- 新建：`src/hsi_lidar_ovseg/data/preprocessing.py`
- 新建：`tests/test_io.py`
- 新建：`tests/test_preprocessing.py`

**接口：**
- 输入：`DataConfig` 中的五组文件路径和可选数组键。
- 输出：`load_scene(config: DataConfig) -> SceneArrays`。
- 输出：`fit_normalization(scene: SceneArrays) -> NormalizationStats`。
- 输出：`normalize_scene(scene, stats) -> SceneArrays`、`terrain_channels(lidar, window_size) -> np.ndarray`、`pseudo_rgb(hsi, indices, train_mask) -> np.ndarray`。

- [ ] **步骤 1：写入数组读取失败测试**

```python
@pytest.mark.parametrize("suffix", [".npy", ".npz", ".mat"])
def test_load_array_supports_declared_formats(tmp_path: Path, suffix: str) -> None:
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = save_array(tmp_path, suffix, expected, key="cube")
    actual = load_array(path, key=None if suffix == ".npy" else "cube")
    np.testing.assert_array_equal(actual, expected)


def test_load_scene_rejects_unpaired_spatial_shapes(valid_data_config: DataConfig) -> None:
    np.save(valid_data_config.lidar_path, np.zeros((7, 8), dtype=np.float32))
    with pytest.raises(DataError, match="spatial shape"):
        load_scene(valid_data_config)
```

- [ ] **步骤 2：运行读取测试并观察失败**

运行：`python -m pytest tests/test_io.py -v`

预期：因 `SceneArrays`、`load_array` 和 `load_scene` 尚未定义而失败。

- [ ] **步骤 3：实现数组读取和场景校验**

`load_array()` 按后缀分派至 `np.load` 或 `scipy.io.loadmat`；`.npz/.mat` 缺少键时报告可用键。`load_scene()` 把二维 HSI/LiDAR 补为单通道，把可识别的通道前置数组转为通道后置，随后验证空间尺寸、有限值、掩码类型和标签范围。

- [ ] **步骤 4：运行读取测试并确认通过**

运行：`python -m pytest tests/test_io.py -v`

预期：全部通过。

- [ ] **步骤 5：写入预处理失败测试**

```python
def test_hsi_statistics_use_training_pixels_only() -> None:
    hsi = np.array([[[1.0], [3.0]], [[1000.0], [2000.0]]], dtype=np.float32)
    train_mask = np.array([[True, True], [False, False]])
    stats = fit_hsi_stats(hsi, train_mask)
    np.testing.assert_allclose(stats.mean, [2.0])
    np.testing.assert_allclose(stats.scale, [1.0])


def test_terrain_channels_are_finite_for_constant_height() -> None:
    result = terrain_channels(np.ones((9, 9, 1), dtype=np.float32), window_size=5)
    assert result.shape == (9, 9, 3)
    assert np.isfinite(result).all()
```

- [ ] **步骤 6：运行预处理测试并观察失败**

运行：`python -m pytest tests/test_preprocessing.py -v`

预期：因统计量和地形函数尚未定义而失败。

- [ ] **步骤 7：实现归一化、地形通道与伪 RGB**

使用训练掩码拟合逐波段均值/标准差；LiDAR 使用中位数和 `max(IQR, epsilon)`。局部相对高度通过 PyTorch `avg_pool2d` 或 SciPy 均值滤波计算，坡度使用中心差分；伪 RGB 对各通道执行训练像素的 2%–98% 截断和 `[0, 1]` 缩放。

- [ ] **步骤 8：运行数据测试并提交**

运行：`python -m pytest tests/test_io.py tests/test_preprocessing.py -v`

运行：`ruff check src/hsi_lidar_ovseg/data tests/test_io.py tests/test_preprocessing.py`

提交：`git commit -m "feat: add scene loading and preprocessing"`

---

### 任务 3：配对图块和滑窗重建

**文件：**
- 新建：`src/hsi_lidar_ovseg/data/tiling.py`
- 新建：`src/hsi_lidar_ovseg/data/datasets.py`
- 新建：`tests/test_tiling.py`
- 新建：`tests/test_datasets.py`

**接口：**
- 输出：`tile_origins(height, width, tile_size, overlap) -> tuple[tuple[int, int], ...]`。
- 输出：`SlidingWindowAccumulator.add(logits, top, left)` 与 `finalize() -> torch.Tensor`。
- 输出：`PairedTileDataset(scene, stats, tile_size, min_seen_pixels, seen_ids, training, seed)`。

- [ ] **步骤 1：写入滑窗覆盖与重建失败测试**

```python
def test_tile_origins_cover_bottom_and_right_edges() -> None:
    origins = tile_origins(13, 17, tile_size=8, overlap=2)
    assert (5, 9) in origins


def test_accumulator_reconstructs_constant_logits() -> None:
    acc = SlidingWindowAccumulator(num_classes=2, height=10, width=11, tile_size=8)
    for top, left in tile_origins(10, 11, 8, 2):
        acc.add(torch.ones(2, 8, 8), top, left)
    torch.testing.assert_close(acc.finalize(), torch.ones(2, 10, 11))
```

- [ ] **步骤 2：运行测试并观察失败**

运行：`python -m pytest tests/test_tiling.py -v`

预期：因平铺接口缺失而失败。

- [ ] **步骤 3：实现确定性平铺和加权累积**

`tile_origins()` 校验 `0 <= overlap < tile_size`，按 `stride=tile_size-overlap` 生成坐标并显式加入最后一行/列起点。累积器使用二维 Hann 权重并将边缘权重限制为正数，保存加权分数和权重和。

- [ ] **步骤 4：运行平铺测试并确认通过**

运行：`python -m pytest tests/test_tiling.py -v`

- [ ] **步骤 5：写入配对数据集失败测试**

```python
def test_spatial_flip_is_shared_across_modalities(scene: SceneArrays) -> None:
    dataset = PairedTileDataset(
        scene, identity_stats(scene), tile_size=4, min_seen_pixels=1,
        seen_ids=(1,), training=True, seed=7,
    )
    sample = dataset[0]
    np.testing.assert_array_equal(sample["hsi"][0].numpy(), sample["lidar"][0].numpy())
    np.testing.assert_array_equal(sample["labels"].numpy(), sample["hsi"][0].numpy())
```

- [ ] **步骤 6：实现配对图块数据集并验证**

数据集预先建立满足 `min_seen_pixels` 的候选坐标，使用按索引派生的局部随机生成器选择水平/垂直翻转和 90 度旋转，并对全部空间张量应用同一变换。输出键固定为 `hsi`、`lidar`、`pseudo_rgb`、`labels`、`valid_mask` 和 `origin`。

运行：`python -m pytest tests/test_tiling.py tests/test_datasets.py -v`

运行：`ruff check src/hsi_lidar_ovseg/data tests/test_tiling.py tests/test_datasets.py`

提交：`git commit -m "feat: add paired tiles and sliding inference"`

---

### 任务 4：编码器协议、原生编码器和外部适配器

**文件：**
- 新建：`src/hsi_lidar_ovseg/models/__init__.py`
- 新建：`src/hsi_lidar_ovseg/models/protocols.py`
- 新建：`src/hsi_lidar_ovseg/models/native.py`
- 新建：`src/hsi_lidar_ovseg/models/hypersigma.py`
- 新建：`src/hsi_lidar_ovseg/models/dinov2.py`
- 新建：`src/hsi_lidar_ovseg/models/clip_text.py`
- 新建：`tests/test_encoders.py`

**接口：**
- 输出协议：`PyramidEncoder` 和 `TextEncoder`。
- 输出：`NativePyramidEncoder(in_channels, channels) -> nn.Module`。
- 输出：`HyperSigmaAdapter(backbone, feature_blocks, feature_dim)`。
- 输出：`DinoV2Adapter(backbone, feature_blocks, feature_dim, frozen)`。
- 输出：`ClipTextEncoder(model, tokenizer, templates)`。

- [ ] **步骤 1：写入原生编码器形状失败测试**

```python
def test_native_encoder_returns_four_level_pyramid() -> None:
    encoder = NativePyramidEncoder(16, channels=(16, 24, 32, 48))
    outputs = encoder(torch.randn(2, 16, 64, 64))
    assert [tuple(x.shape) for x in outputs] == [
        (2, 16, 16, 16), (2, 24, 8, 8), (2, 32, 4, 4), (2, 48, 2, 2)
    ]
```

- [ ] **步骤 2：运行测试并观察失败**

运行：`python -m pytest tests/test_encoders.py::test_native_encoder_returns_four_level_pyramid -v`

预期：因编码器缺失而失败。

- [ ] **步骤 3：实现原生特征金字塔**

每层使用 `Conv2d -> GroupNorm -> GELU -> residual depthwise block`。首层步长为 4，后续层步长为 2；`out_channels` 与 `out_strides=(4, 8, 16, 32)` 作为只读属性。

- [ ] **步骤 4：写入外部适配器契约失败测试**

```python
def test_frozen_dino_adapter_keeps_backbone_in_eval() -> None:
    backbone = FakeTokenBackbone(embed_dim=32, patch_size=8)
    adapter = DinoV2Adapter(backbone, feature_blocks=(1, 2, 3, 4), feature_dim=16, frozen=True)
    adapter.train()
    assert not adapter.backbone.training
    assert not any(parameter.requires_grad for parameter in adapter.backbone.parameters())


def test_text_encoder_normalizes_prompt_ensemble() -> None:
    encoder = ClipTextEncoder(FakeClip(), fake_tokenizer, templates=("a {}", "satellite {}"))
    embeddings = encoder.encode(("tree", "road"))
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2))
```

- [ ] **步骤 5：实现适配器并运行编码器测试**

适配器不负责联网构建模型，只包装已注入的主干。令牌适配器读取主干返回的 `[N, T, C]` 中间特征，移除可选类别令牌，根据输入和 patch size 还原网格，再通过四个确定性重采样头得到步长 4/8/16/32 的金字塔。冻结适配器重写 `train()`，确保冻结主干始终保持评估模式。

运行：`python -m pytest tests/test_encoders.py -v`

运行：`ruff check src/hsi_lidar_ovseg/models tests/test_encoders.py`

提交：`git commit -m "feat: add native and pretrained encoder adapters"`

---

### 任务 5：门控融合、CLIP 空间解码与完整模型

**文件：**
- 新建：`src/hsi_lidar_ovseg/models/fusion.py`
- 新建：`src/hsi_lidar_ovseg/models/decoder.py`
- 新建：`src/hsi_lidar_ovseg/models/model.py`
- 新建：`tests/test_fusion.py`
- 新建：`tests/test_model.py`

**接口：**
- 输出：`GatedPyramidFusion(hsi_channels, lidar_channels, feature_dim)`。
- 输出：`DenseTextDecoder(feature_dim, text_dim)`。
- 输出：`HSILidarOVSegmentor(...).forward(hsi, lidar, pseudo_rgb, text_embeddings) -> SegmentationOutput`。

- [ ] **步骤 1：写入门控融合失败测试**

```python
def test_gated_fusion_returns_bounded_gates_and_gradients() -> None:
    hsi = tuple(torch.randn(2, 8, 8 // 2**i, 8 // 2**i, requires_grad=True) for i in range(4))
    lidar = tuple(torch.randn_like(x, requires_grad=True) for x in hsi)
    fused, gates = GatedPyramidFusion((8,) * 4, (8,) * 4, 16)(hsi, lidar)
    assert all(torch.all((gate >= 0) & (gate <= 1)) for gate in gates)
    sum(x.mean() for x in fused).backward()
    assert hsi[0].grad is not None and lidar[0].grad is not None
```

- [ ] **步骤 2：运行测试并观察失败**

运行：`python -m pytest tests/test_fusion.py -v`

- [ ] **步骤 3：实现逐层投影与门控融合**

每层分别使用 `1x1 Conv + GroupNorm` 投影 HSI 和 LiDAR；拼接后用 `3x3 Conv -> GELU -> 1x1 Conv -> sigmoid` 产生单通道空间门控，再按设计公式融合。

- [ ] **步骤 4：写入完整模型失败测试**

```python
def test_model_outputs_dense_normalized_embeddings() -> None:
    model = make_native_model(hsi_bands=12, lidar_channels=3, feature_dim=16, text_dim=24)
    text = F.normalize(torch.randn(5, 24), dim=-1)
    output = model(
        torch.randn(2, 12, 64, 64),
        torch.randn(2, 3, 64, 64),
        torch.randn(2, 3, 64, 64),
        text,
    )
    assert output.logits.shape == (2, 5, 64, 64)
    torch.testing.assert_close(
        output.pixel_embeddings.norm(dim=1), torch.ones(2, 64, 64), atol=1e-5, rtol=1e-5
    )
```

- [ ] **步骤 5：实现 FPN 解码器和完整模型**

解码器从最深层开始逐层双线性上采样并与横向 `1x1` 投影相加，最终上采样到输入分辨率并映射到 `text_dim`。模型对像素和文本嵌入归一化，使用 `exp(clamp(logit_scale, min=log(1), max=log(100)))` 缩放余弦相似度。

- [ ] **步骤 6：运行模型测试并提交**

运行：`python -m pytest tests/test_fusion.py tests/test_model.py -v`

运行：`ruff check src/hsi_lidar_ovseg/models tests/test_fusion.py tests/test_model.py`

提交：`git commit -m "feat: add gated open-vocabulary segmentor"`

---

### 任务 6：对齐损失、总目标与开放词汇指标

**文件：**
- 新建：`src/hsi_lidar_ovseg/losses/__init__.py`
- 新建：`src/hsi_lidar_ovseg/losses/contrastive.py`
- 新建：`src/hsi_lidar_ovseg/losses/objective.py`
- 新建：`src/hsi_lidar_ovseg/metrics.py`
- 新建：`tests/test_losses.py`
- 新建：`tests/test_metrics.py`

**接口：**
- 输出：`symmetric_info_nce(first, second, valid_mask=None, temperature=0.1) -> Tensor`。
- 输出：`OpenVocabularyObjective(config).forward(output, labels, valid_mask) -> dict[str, Tensor]`。
- 输出：`SegmentationMetrics(num_classes, seen_ids, unseen_ids, ignore_index=0)`。

- [ ] **步骤 1：写入对齐损失失败测试**

```python
def test_info_nce_prefers_matching_pairs() -> None:
    aligned = F.normalize(torch.eye(4), dim=-1)
    shuffled = aligned[[1, 0, 3, 2]]
    assert symmetric_info_nce(aligned, aligned) < symmetric_info_nce(aligned, shuffled)


def test_objective_rejects_empty_supervised_mask() -> None:
    with pytest.raises(LossError, match="supervised mask"):
        objective(output_fixture(), torch.zeros(1, 8, 8, dtype=torch.long), torch.ones(1, 8, 8, dtype=torch.bool))
```

- [ ] **步骤 2：实现归一化 InfoNCE 和组合目标**

InfoNCE 将两组 `[N, D]` 特征归一化，计算双向交叉熵并取平均。总目标从配置读取权重，分割项只选取已见类且有效的像素，并返回包含 `total`、`segmentation`、`hsi_teacher`、`lidar_teacher`、`hsi_lidar`、`gate` 和 `private` 的字典。

- [ ] **步骤 3：运行损失测试并确认通过**

运行：`python -m pytest tests/test_losses.py -v`

- [ ] **步骤 4：写入指标失败测试**

```python
def test_metrics_report_seen_unseen_harmonic_mean() -> None:
    metrics = SegmentationMetrics(3, seen_ids=(1, 2), unseen_ids=(3,))
    metrics.update(torch.tensor([1, 2, 3, 3]), torch.tensor([1, 2, 3, 1]))
    result = metrics.compute()
    assert result["seen_miou"] == pytest.approx(0.75)
    assert result["unseen_miou"] == pytest.approx(0.5)
    assert result["harmonic_miou"] == pytest.approx(0.6)
```

- [ ] **步骤 5：实现混淆矩阵指标并提交**

以 `int64` 混淆矩阵累计预测；从对角线、行和、列和计算每类 IoU、类别准确率、总体准确率、已见/未见均值和调和均值，类别无真实像素时从该均值中排除。

运行：`python -m pytest tests/test_losses.py tests/test_metrics.py -v`

运行：`ruff check src/hsi_lidar_ovseg/losses src/hsi_lidar_ovseg/metrics.py tests/test_losses.py tests/test_metrics.py`

提交：`git commit -m "feat: add multimodal losses and metrics"`

---

### 任务 7：检查点、训练步骤与滑窗评估

**文件：**
- 新建：`src/hsi_lidar_ovseg/engine/__init__.py`
- 新建：`src/hsi_lidar_ovseg/engine/checkpoint.py`
- 新建：`src/hsi_lidar_ovseg/engine/trainer.py`
- 新建：`src/hsi_lidar_ovseg/engine/evaluator.py`
- 新建：`tests/test_checkpoint.py`
- 新建：`tests/test_training_smoke.py`

**接口：**
- 输出：`save_checkpoint(path, state: TrainingState) -> None`。
- 输出：`load_checkpoint(path, model, optimizer, expected: CheckpointIdentity) -> TrainingState`。
- 输出：`Trainer.train_step(batch) -> dict[str, float]`。
- 输出：`sliding_window_predict(model, scene, text_embeddings, tile_size, overlap, device) -> Tensor`。

- [ ] **步骤 1：写入检查点兼容性失败测试**

```python
def test_checkpoint_rejects_class_identity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(path, training_state(class_names=("tree", "road")))
    with pytest.raises(CheckpointError, match="class_names"):
        load_checkpoint(path, model(), optimizer(), expected_identity(class_names=("road", "tree")))
```

- [ ] **步骤 2：实现原子检查点保存和身份校验**

先保存到目标目录内的临时文件，再使用 `Path.replace()` 原子替换。加载前比较类别名、已见/未见编号、HSI 波段数、LiDAR 通道数、`feature_dim` 和 `text_dim`，列出所有冲突字段后拒绝恢复。

- [ ] **步骤 3：写入单步训练失败测试**

```python
def test_cpu_training_step_updates_trainable_parameter() -> None:
    trainer, batch = make_synthetic_trainer_and_batch()
    before = trainer.model.decoder.output.weight.detach().clone()
    losses = trainer.train_step(batch)
    after = trainer.model.decoder.output.weight.detach()
    assert np.isfinite(losses["total"])
    assert not torch.equal(before, after)
```

- [ ] **步骤 4：实现训练器和滑窗评估器**

训练器负责设备搬运、`zero_grad(set_to_none=True)`、自动混合精度、反向传播、梯度裁剪、优化器/调度器更新和非有限损失检测。评估器按图块调用模型，将输出交给 `SlidingWindowAccumulator`，裁剪到原尺寸并更新指标。

- [ ] **步骤 5：运行引擎测试并提交**

运行：`python -m pytest tests/test_checkpoint.py tests/test_training_smoke.py -v`

运行：`ruff check src/hsi_lidar_ovseg/engine tests/test_checkpoint.py tests/test_training_smoke.py`

提交：`git commit -m "feat: add training evaluation and checkpoints"`

---

### 任务 8：命令行、数据集配置与使用文档

**文件：**
- 新建：`src/hsi_lidar_ovseg/cli.py`
- 新建：`configs/base.yaml`
- 新建：`configs/houston2013.yaml`
- 新建：`configs/trento.yaml`
- 新建：`configs/muufl.yaml`
- 新建：`README.md`
- 新建：`tests/test_cli.py`

**接口：**
- 输出命令：`hsi-lidar-ovseg validate-config CONFIG`。
- 输出命令：`hsi-lidar-ovseg train CONFIG`。
- 输出命令：`hsi-lidar-ovseg evaluate CONFIG CHECKPOINT`。

- [ ] **步骤 1：写入 CLI 失败测试**

```python
def test_cli_help_lists_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hsi_lidar_ovseg.cli", "--help"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert {"train", "evaluate", "validate-config"} <= set(result.stdout.split())
```

- [ ] **步骤 2：实现 argparse 命令和对象构建器**

`validate-config` 只加载并检查配置；`train` 构建数据、模型、文本原型、目标函数和训练器；`evaluate` 除此以外加载检查点并运行整图评估。入口捕获预期的配置/数据/检查点错误，记录一条错误并返回退出码 2；未预期异常保留堆栈。

- [ ] **步骤 3：编写三个数据集示例配置**

每个 YAML 显式填写类别名称、伪 RGB 波段索引、项目默认已见/未见划分和预期数组键。所有真实路径使用相对 `data/<dataset>/...`，权重路径使用相对 `weights/...`；README 明确这些是用户需要准备的本地文件，不会自动下载。

- [ ] **步骤 4：编写中文 README**

README 覆盖安装、数据布局、MATLAB 键检查方法、权重准备、配置校验、训练、恢复、评估、输出指标、开放词汇划分声明和原生编码器离线冒烟测试。

- [ ] **步骤 5：运行 CLI、测试和构建并提交**

运行：`python -m pytest tests/test_cli.py -v`

运行：`python -m hsi_lidar_ovseg.cli --help`

运行：`python -m build`

提交：`git commit -m "feat: add CLI dataset configs and documentation"`

---

### 任务 9：全量质量验证

**文件：**
- 修改：仅修改验证发现存在明确缺陷的文件；每个缺陷先在对应测试文件加入失败用例。

**接口：**
- 输入：完整仓库。
- 输出：测试、静态检查、构建和离线 CLI 验证证据。

- [ ] **步骤 1：运行全量测试**

运行：`python -m pytest -v`

预期：所有测试通过，无失败、错误或未处理警告。

- [ ] **步骤 2：运行静态检查**

运行：`ruff check .`

运行：`ruff format --check .`

预期：两条命令均以退出码 0 结束。

- [ ] **步骤 3：构建软件包**

运行：`python -m build`

预期：生成 `.tar.gz` 源码包和 `.whl` 包。

- [ ] **步骤 4：验证离线命令行**

运行：`python -m hsi_lidar_ovseg.cli --help`

运行：`python -m hsi_lidar_ovseg.cli validate-config configs/houston2013.yaml --skip-file-checks`

预期：帮助列出三个子命令，示例配置在跳过本地数据/权重存在性检查时通过结构校验；两条命令均不访问网络。

- [ ] **步骤 5：核对设计覆盖并提交修正**

逐项核对设计文档第 14 节七条验收标准。若验证产生修正，运行受影响测试及全量验证后提交：`git commit -m "test: verify complete HSI LiDAR pipeline"`。
