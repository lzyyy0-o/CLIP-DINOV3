# rs-fusion-datasets Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让工程以 `rs-fusion-datasets` 直接加载 Houston 2013、Houston 2018、Trento 和 MUUFL，同时维持现有本地文件工作流。

**Architecture:** `DataConfig.source` 在 `files` 与 `rs_fusion_datasets` 间显式分派。新的 rs-fusion 适配器把第三方返回的 CHW 数组和标签图转换成既有 `SceneArrays`；原有 `load_scene` 集中执行两种来源共享的数据校验。Trento/MUUFL 的训练掩码由确定性按类抽样生成，CLI 传入顶层实验种子。

**Tech Stack:** Python 3.10+、NumPy、PyTorch、PyYAML、rs-fusion-datasets、pytest、ruff。

**Spec:** `docs/superpowers/specs/2026-08-31-rs-fusion-datasets-adapter-design.md`

## Global Constraints

- 保留既有 `.mat/.npy/.npz` 配置和文件加载行为。
- rs-fusion-datasets 必须延迟导入；单测不得安装、下载或联网。
- 两个数据来源均须产出通道末尾、有限、严格配准的 `SceneArrays`。
- Houston 2013/2018 使用工具包提供的官方训练/测试掩码。
- Trento/MUUFL 使用 `rs_train_samples_per_class` 和顶层实验 seed 的确定性分层划分。
- 新 YAML 保持既定 10/5、14/6、4/2、7/4 seen/unseen 协议。
- 不触碰未跟踪的 `.vscode/`、`docs/figures/`、`third_party/`。

---

## File Structure

| 文件 | 责任 |
|---|---|
| `src/hsi_lidar_ovseg/config.py` | 解析并严格验证 rs-fusion 数据源字段。 |
| `src/hsi_lidar_ovseg/data/io.py` | 按来源分派并复用 `SceneArrays` 契约验证。 |
| `src/hsi_lidar_ovseg/data/rs_fusion.py` | 调用第三方 fetch 函数、转换数组和构建掩码。 |
| `src/hsi_lidar_ovseg/data/__init__.py` | 导出 rs-fusion 加载接口。 |
| `src/hsi_lidar_ovseg/cli.py` | 将实验 seed 传给 `load_scene`。 |
| `configs/*_rs_fusion.yaml` | 四数据集 rs-fusion 示例配置。 |
| `requirements.txt`、`README.md` | 声明依赖与服务器使用方式。 |
| `tests/test_config.py`、`tests/test_io.py`、`tests/test_rs_fusion.py` | 验证配置、来源分派、转换与确定性抽样。 |

### Task 1: 数据源判别配置

**Files:**
- Modify: `src/hsi_lidar_ovseg/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `DataConfig.source`, `rs_dataset`, `rs_data_home`, `rs_train_samples_per_class`。
- Consumes: 既有 YAML 与 `_decode_data`。

- [ ] **Step 1: 写失败的配置测试**

```python
def test_rs_fusion_data_config_accepts_required_fields(tmp_path: Path) -> None:
    values = _valid_config_dict()
    values["data"] = {
        "name": "Houston 2013",
        "source": "rs_fusion_datasets",
        "rs_dataset": "houston2013",
        "rs_data_home": str(tmp_path / "cache"),
        "rs_train_samples_per_class": None,
        "hsi_path": None, "lidar_path": None, "labels_path": None,
        "train_mask_path": None, "test_mask_path": None,
        "hsi_key": None, "lidar_key": None, "labels_key": None,
        "train_mask_key": None, "test_mask_key": None,
        "class_names": ["tree", "road"],
        "seen_class_ids": [1], "unseen_class_ids": [2],
        "pseudo_rgb_indices": [0, 1, 2],
    }
    path = tmp_path / "rs.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    assert load_config(path, check_files=False).data.rs_dataset == "houston2013"


@pytest.mark.parametrize(
    "data_patch",
    [
        {"rs_dataset": None},
        {"hsi_path": "hsi.mat"},
        {"rs_train_samples_per_class": 20},
    ],
)
def test_rs_fusion_config_rejects_invalid_field_combinations(
    tmp_path: Path, data_patch: dict[str, object]
) -> None:
    values = _valid_config_dict()
    data = _rs_fusion_data_dict(tmp_path, "houston2013")
    data.update(data_patch)
    values["data"] = data
    path = tmp_path / "invalid-rs.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, check_files=False)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -q`

Expected: FAIL，因为 `DataConfig` 不接受 rs-fusion 字段且路径字段不可为空。

- [ ] **Step 3: 实现严格的来源验证与解码**

将五个路径改为 `Path | None`，并在 `DataConfig.__post_init__` 做来源专属验证：

```python
if self.source == "files":
    if any(path is None for path in paths):
        raise ConfigError("source=files 时五个数组路径必须提供")
    if self.rs_dataset is not None or self.rs_data_home is not None:
        raise ConfigError("source=files 不得配置 rs_fusion 字段")
elif self.source == "rs_fusion_datasets":
    if any(path is not None for path in paths):
        raise ConfigError("rs_fusion_datasets 不得配置本地数组路径")
    if self.rs_dataset not in {"houston2013", "houston2018_ouc", "trento", "muufl"}:
        raise ConfigError("rs_dataset 不受支持")
```

在 `_decode_data` 中只对非空文件路径调用 `_path(value, context, optional=True)`；把 `rs_data_home` 译为 `Path`，并接受整数或 `0 < float < 1` 的抽样值。`validate_files` 对 files 继续检查五个文件，对 rs 模式仅要求 `rs_data_home.is_dir()`。

- [ ] **Step 4: 运行配置回归**

Run: `python -m pytest tests/test_config.py -q`

Expected: PASS，既有文件配置与新增 rs 配置测试均通过。

- [ ] **Step 5: 提交配置层**

```bash
git add src/hsi_lidar_ovseg/config.py tests/test_config.py
git commit -m "feat: add rs-fusion data configuration"
```

### Task 2: rs-fusion 适配器与共享场景验证

**Files:**
- Create: `src/hsi_lidar_ovseg/data/rs_fusion.py`
- Modify: `src/hsi_lidar_ovseg/data/io.py`
- Modify: `src/hsi_lidar_ovseg/data/__init__.py`
- Create: `tests/test_rs_fusion.py`
- Modify: `tests/test_io.py`

**Interfaces:**
- Produces: `load_rs_fusion_scene(config: DataConfig, *, split_seed: int) -> SceneArrays`。
- Changes: `load_scene(config: DataConfig, *, split_seed: int = 0) -> SceneArrays`。

- [ ] **Step 1: 写失败的适配器测试**

```python
def test_houston2013_adapter_converts_chw_and_official_masks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = SimpleNamespace(
        fetch_houston2013=lambda *, data_home: (
            np.arange(3 * 2 * 4).reshape(3, 2, 4),
            np.ones((1, 2, 4)),
            _sparse([[1, 0, 0, 0], [0, 2, 0, 0]]),
            _sparse([[0, 0, 3, 0], [0, 0, 0, 1]]),
            {},
        )
    )
    monkeypatch.setitem(sys.modules, "rs_fusion_datasets", fake)
    scene = load_scene(_rs_config("houston2013", tmp_path), split_seed=13)
    assert scene.hsi.shape == (2, 4, 3)
    assert scene.lidar.shape == (2, 4, 1)
    assert scene.labels.tolist() == [[1, 0, 3, 0], [0, 2, 0, 1]]
    assert scene.train_mask.sum() == scene.test_mask.sum() == 2


def test_trento_adapter_samples_each_class_deterministically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    labels = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int64)
    fake = SimpleNamespace(
        fetch_trento=lambda *, data_home: (
            np.ones((4, 2, 3)), np.ones((1, 2, 3)), labels, {}
        )
    )
    monkeypatch.setitem(sys.modules, "rs_fusion_datasets", fake)
    config = _rs_config("trento", tmp_path, samples_per_class=2)
    first = load_scene(config, split_seed=7)
    second = load_scene(config, split_seed=7)
    np.testing.assert_array_equal(first.train_mask, second.train_mask)
    assert first.train_mask.sum() == 4
    assert first.test_mask.sum() == 2


def test_rs_fusion_missing_dependency_has_install_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("builtins.__import__", _raise_rs_fusion_import_error)
    with pytest.raises(DataError, match="requirements.txt"):
        load_scene(_rs_config("houston2013", tmp_path), split_seed=0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_rs_fusion.py -q`

Expected: FAIL，因为 `data.rs_fusion` 和 `load_scene(config, split_seed=seed)` 尚不存在。

- [ ] **Step 3: 实现转换、掩码和确定性抽样**

在 `rs_fusion.py` 仅函数内部导入第三方包。实现转换助手：

```python
def _channel_last(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 3:
        raise DataError(f"rs-fusion {name} 必须是 CHW 三维数组")
    return np.moveaxis(values, 0, -1)

def _dense_labels(labels: object) -> np.ndarray:
    toarray = getattr(labels, "toarray", None)
    return np.asarray(toarray() if callable(toarray) else labels)
```

为 Trento/MUUFL 实现 `stratified_rs_split(labels, samples_per_class, seed)`：逐类别使用 `np.random.SeedSequence((seed, class_id))`，整数取 `min(requested, count - 1)`，比例取 `min(max(1, round(count * fraction)), count - 1)`；类别像素少于 2 时抛出 `DataError`。把未选中正标签像素标记为测试掩码。

从 `io.py` 抽出 `_validate_scene_arrays(config, hsi, lidar, labels, train_mask, test_mask)`，令文件加载和 rs 加载都调用它。`load_scene` 在 rs 模式调用适配器，避免循环导入：`rs_fusion.py` 仅从 `io` 导入 `DataError`、`SceneArrays`、`_validate_scene_arrays`。

- [ ] **Step 4: 运行数据层回归**

Run: `python -m pytest tests/test_io.py tests/test_rs_fusion.py -q && ruff check src/hsi_lidar_ovseg/data tests/test_io.py tests/test_rs_fusion.py`

Expected: PASS；测试不访问网络。

- [ ] **Step 5: 提交适配器**

```bash
git add src/hsi_lidar_ovseg/data/io.py src/hsi_lidar_ovseg/data/rs_fusion.py src/hsi_lidar_ovseg/data/__init__.py tests/test_io.py tests/test_rs_fusion.py
git commit -m "feat: load scenes through rs-fusion datasets"
```

### Task 3: CLI 传递种子和 rs-fusion 配置集

**Files:**
- Modify: `src/hsi_lidar_ovseg/cli.py`
- Create: `configs/houston2013_rs_fusion.yaml`
- Create: `configs/houston2013_shared_lite_vit_clip_rs_fusion.yaml`
- Create: `configs/houston2018_shared_lite_vit_clip_rs_fusion.yaml`
- Create: `configs/trento_rs_fusion.yaml`
- Create: `configs/muufl_rs_fusion.yaml`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `load_scene(config.data, split_seed=config.seed)` and rs-enabled `DataConfig`.
- Produces: 五份可离线校验的配置模板。

- [ ] **Step 1: 写失败的 CLI/YAML 测试**

```python
@pytest.mark.parametrize(
    "name",
    [
        "houston2013_rs_fusion.yaml",
        "houston2013_shared_lite_vit_clip_rs_fusion.yaml",
        "houston2018_shared_lite_vit_clip_rs_fusion.yaml",
        "trento_rs_fusion.yaml",
        "muufl_rs_fusion.yaml",
    ],
)
def test_cli_validates_rs_fusion_examples_without_downloading(name: str) -> None:
    result = subprocess.run(
        [
            sys.executable, "-m", "hsi_lidar_ovseg.cli", "validate-config",
            str(ROOT / "configs" / name), "--skip-file-checks",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=_environment(),
    )
    assert result.returncode == 0, result.stderr
```

在 `test_rs_fusion.py` 中补一个 CLI 调用路径测试，monkeypatch `cli.load_scene` 并断言收到 `split_seed=config.seed`。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cli.py tests/test_config.py tests/test_rs_fusion.py -q`

Expected: FAIL，因为 YAML 文件和 CLI 的 `split_seed` 调用尚未创建。

- [ ] **Step 3: 实现 CLI 分派并创建 YAML**

将训练和评估中的调用改为：

```python
scene = load_scene(config.data, split_seed=config.seed)
```

每份新 YAML 设置：

```yaml
data:
  source: rs_fusion_datasets
  rs_dataset: houston2013
  rs_data_home: data/rs_fusion_cache
  hsi_path: null
  lidar_path: null
  labels_path: null
  train_mask_path: null
  test_mask_path: null
  hsi_key: null
  lidar_key: null
  labels_key: null
  train_mask_key: null
  test_mask_key: null
```

为 Trento/MUUFL 加入 `rs_train_samples_per_class: 20`；Houston 配置中该值为 `null`。复制相应的类别名称、seen/unseen 划分、伪 RGB 波段、模型和训练参数，确保 CLIP 引导 YAML 使用 `loss.kind=masked_cross_entropy` 且五个旧损失权重全为 `0.0`。

- [ ] **Step 4: 验证配置与回归**

Run: `python -m pytest tests/test_cli.py tests/test_config.py tests/test_rs_fusion.py -q && python -m hsi_lidar_ovseg.cli validate-config configs/houston2013_shared_lite_vit_clip_rs_fusion.yaml --skip-file-checks`

Expected: PASS；无需真实数据目录或网络。

- [ ] **Step 5: 提交 CLI 与配置**

```bash
git add src/hsi_lidar_ovseg/cli.py configs tests/test_cli.py tests/test_config.py tests/test_rs_fusion.py
git commit -m "feat: add rs-fusion experiment configurations"
```

### Task 4: 服务器依赖与操作说明

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`
- Modify: `tests/test_dependencies.py`

**Interfaces:**
- Produces: `rs-fusion-datasets` 可通过 `pip install -r requirements.txt` 安装；README 说明 rs 缓存目录与训练命令。

- [ ] **Step 1: 写失败的依赖测试**

```python
def test_server_requirements_include_rs_fusion_datasets() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "rs-fusion-datasets" in requirements
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_dependencies.py -q`

Expected: FAIL，因为 requirements 尚未声明该包。

- [ ] **Step 3: 更新依赖和 README**

在 `requirements.txt` 增加 `rs-fusion-datasets`。README 增加：rs-fusion YAML 的使用方式、首次 `train` 允许第三方包下载、`data/rs_fusion_cache` 需预先创建、完整 `validate-config` 不下载数据，以及 Houston 与 Trento/MUUFL 掩码策略差异。

- [ ] **Step 4: 最终验证**

Run: `python -m pytest -q && ruff check src tests && python -m hsi_lidar_ovseg.cli validate-config configs/houston2013_shared_lite_vit_clip_rs_fusion.yaml --skip-file-checks`

Expected: 所有测试和风格检查通过，且配置解析不访问网络。

- [ ] **Step 5: 提交文档与依赖**

```bash
git add requirements.txt README.md tests/test_dependencies.py
git commit -m "docs: document rs-fusion dataset loading"
```

## Plan Self-Review

- **规格覆盖：** Task 1 实现两种来源的配置约束；Task 2 实现四数据集转换、官方掩码、确定性抽样与统一场景验证；Task 3 传递实验种子并提供五份 YAML；Task 4 声明依赖和服务器使用方法。
- **占位符检查：** 每个实现任务含具体文件、接口、失败测试、运行命令与最小实现算法；没有未定义的后续动作。
- **类型一致性：** `load_scene(config, split_seed=seed)` 与 `load_rs_fusion_scene(config, split_seed=seed)` 使用同一 `int` 种子；所有来源最终返回 `SceneArrays`；rs 配置统一以 `source="rs_fusion_datasets"` 判别。
