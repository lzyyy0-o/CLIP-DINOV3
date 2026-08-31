# rs-fusion-datasets 四数据集适配设计

## 目标

让工程可直接通过 `rs-fusion-datasets` 加载 Houston 2013、Houston 2018、Trento 与 MUUFL Gulfport，同时保持已有本地 `.mat/.npy/.npz` 加载路径不变。所有数据源最终都必须产生同一份 `SceneArrays`，从而继续使用训练区归一化、严格配准校验、类别感知密集裁剪、验证划分与整图滑窗评估。

## 配置接口

`DataConfig` 新增以下字段：

- `source: Literal["files", "rs_fusion_datasets"] = "files"`
- `rs_dataset: Literal["houston2013", "houston2018_ouc", "trento", "muufl"] | None = None`
- `rs_data_home: Path | None = None`
- `rs_train_samples_per_class: int | float | None = None`

当 `source="files"` 时，五个数组路径和键名仍为必填项，并沿用当前验证逻辑。`source="rs_fusion_datasets"` 时路径和键名必须为 `null`，`rs_dataset` 与 `rs_data_home` 必填；`rs_data_home` 是该工具包下载和缓存原始数据的目录，不承载项目检查点或输出。

`rs_train_samples_per_class` 只适用于 Trento、MUUFL：正整数表示每类训练样本数；`0 < float < 1` 表示每类比例。Houston 2013、Houston 2018 使用其提供的官方训练/测试标签，禁止配置该字段。

## 适配层

新增 `data/rs_fusion.py`，提供 `load_rs_fusion_scene(config: DataConfig, *, split_seed: int) -> SceneArrays`。`load_scene` 扩展为 `load_scene(config: DataConfig, *, split_seed: int = 0)` 并按 `source` 分派；CLI 用实验顶层 `seed` 传入 `split_seed`，本地文件模式忽略该参数。模块在函数内延迟导入 `rs_fusion_datasets`；缺包时抛出 `DataError`，说明执行 `pip install -r requirements.txt`。

适配器按 `rs_dataset` 调用：

- `fetch_houston2013(data_home=...)`：返回 HSI、DSM、官方训练标签、官方测试标签；标签取两者逐像素最大值，两个正标签图分别转为布尔掩码。
- `fetch_houston2018_ouc(data_home=...)`：返回 HSI、DSM、训练标签、测试标签、完整标签；完整标签为语义标签来源，训练/测试正标签图转为掩码。
- `fetch_trento(data_home=...)`、`fetch_muufl(data_home=...)`：返回 HSI、DSM、完整标签；由本项目的确定性分层函数按 `rs_train_samples_per_class` 和实验 `seed` 生成互斥训练/测试掩码。

工具包的 HSI 与 DSM 输入为 CHW，适配层统一转为 HWC；单通道 DSM 最终输出 `[H,W,1]`。适配后使用同一套有限值、空间配准、标签范围和伪 RGB 波段校验，禁止两条加载路径在数据契约上分叉。

## YAML 与实验协议

保留原有配置不动，新增四份配置：

- `houston2013_rs_fusion.yaml`
- `houston2013_shared_lite_vit_clip_rs_fusion.yaml`
- `houston2018_shared_lite_vit_clip_rs_fusion.yaml`
- `trento_rs_fusion.yaml`
- `muufl_rs_fusion.yaml`

Houston 2013 保持 10 seen / 5 unseen；Houston 2018 保持 14 / 6；Trento 保持 4 / 2；MUUFL 保持 7 / 4。Trento 与 MUUFL 新 YAML 固定 `rs_train_samples_per_class: 20`，种子沿用顶层 `seed`。

## 依赖与错误处理

`requirements.txt` 增加 `rs-fusion-datasets`。该包负责数据下载和缓存，工程本身不写网络下载代码。`validate-config --skip-file-checks` 只校验 rs-fusion 参数；完整 `validate-config` 不主动下载数据，而是验证 `rs_data_home` 已存在。首次真实 `train` 或 `evaluate` 可由工具包按其数据源规则完成下载。

对不支持的数据集标识、缺少缓存目录、错误的抽样参数、工具包缺失、返回数组数量/形状异常，必须提供中文的可操作错误信息。

## 测试

单测通过向 `sys.modules` 注入最小假 `rs_fusion_datasets` 模块，禁止网络访问。覆盖：

- 两套 Houston 官方掩码转换、CHW 到 HWC 转换和标签合成；
- Trento、MUUFL 的确定性分层抽样、每类数目与训练/测试互斥；
- 配置字段的文件模式兼容性、rs 模式必填项和不允许的组合；
- 所有新增 YAML 的离线解析与类别划分覆盖；
- 缺少依赖时的错误信息。

现有 `load_scene` 测试保持通过；适配器测试不得加载真实权重、真实数据或访问网络。
