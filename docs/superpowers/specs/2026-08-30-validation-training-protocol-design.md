# 训练—验证—测试协议设计

## 目标

将当前“每轮测试集选优”的训练流程改为可复现的训练、验证、测试三分离协议：训练掩码拟合与更新参数，验证掩码选择模型并早停，测试掩码仅在训练结束后报告一次最终指标。

## 范围与非目标

本次实现覆盖确定性分层验证划分、余弦学习率调度、早停、检查点恢复、最终测试产物、配置与测试。不会增加交叉验证、自动超参数搜索、多随机种子训练或改变网络结构与损失公式。

## 数据划分

新增纯 NumPy 的数据划分函数，输入 `labels`、原始 `train_mask`、`seen_class_ids`、`validation_fraction` 与 `seed`，输出不重叠的训练掩码和验证掩码。

- 只有同时满足原始 `train_mask` 与已见类别的像素可进入验证集。
- 按类别独立、确定性无放回抽样；每个类由 `(seed, class_id)` 派生随机流，因此可重放且不受类别遍历顺序影响。
- 对每个含有至少两个训练像素的类别，验证数为 `min(count - 1, max(1, round(count * validation_fraction)))`；只有一个训练像素的类别全部留在训练集。
- 原始训练掩码中不属于已见类别的像素保留在训练掩码。它们不参与监督或类别中心采样，但维持项目既有的训练区统计语义。
- 归一化、伪 RGB 统计、训练 Dataset 和类别感知中心采样使用新训练掩码；验证指标使用验证掩码。

## 配置接口

`TrainConfig` 新增并校验以下字段：

```yaml
epochs: 100
validation_fraction: 0.1
early_stopping_patience: 20
early_stopping_min_delta: 0.0
cosine_eta_min: 0.000001
```

- `validation_fraction` 必须位于 `(0, 1)`。
- `early_stopping_patience` 必须为正整数。
- `early_stopping_min_delta` 必须非负。
- `cosine_eta_min` 必须非负且小于两个初始学习率中的较小值。
- 所有发布的 Houston 2013、Trento、MUUFL、base YAML 的最大轮数改为 100，并显式写出以上字段。

## 训练编排

训练入口首先依据原始场景构造 `training_scene`，再仅对它拟合归一化统计。训练 Dataset 使用 `training_scene`；它的固定网格和类别中心候选不会读取验证或测试像素。

优化器仍使用两组学习率：新建融合/解码模块使用 `learning_rate`，可训练学生主干使用 `backbone_learning_rate`。构建 DataLoader 后创建 `CosineAnnealingLR`，令 `T_max = epochs * len(loader)`、`eta_min = cosine_eta_min`；Trainer 在每个成功优化步后推进一次调度器。

每轮完成后，整图滑窗推理一次，但仅以 `validation_mask` 计算 `seen_miou` 作为选择指标。验证掩码只从已见类训练像素中抽取，不能以必然缺少未见类的 `harmonic_miou` 选优。若新分数严格超过 `best_score + early_stopping_min_delta`，保存 `best.pt` 并将未改善计数置零；否则计数加一。连续 `early_stopping_patience` 轮未改善时停止训练。`last.pt` 每轮保存。

训练循环结束后，从 `best.pt` 恢复模型和训练状态，对原始 `test_mask` 做一次完整评估，并在输出目录写入 `test_metrics.json`。测试分数不会影响保存、早停或学习率。

## 恢复训练与检查点

检查点新增可序列化的 `selection_state`，包含 `best_score` 与 `epochs_without_improvement`，并保存余弦调度器状态。恢复训练时恢复二者，使学习率曲线和早停计数连续。

为兼容本次改动之前的检查点，缺少 `selection_state` 时以 `best_score=-inf`、`epochs_without_improvement=0` 初始化；模型、优化器、Scaler 与原有归一化统计仍按原逻辑恢复。

## 错误处理

- 验证划分后若没有任何已见类训练像素，抛出清晰的数据错误。
- 验证掩码为空时，不启动训练并提示提高类别标注数或调整划分比例。
- 余弦调度器要求训练 Dataset 至少产生一个 batch；现有 Dataset 的图块资格检查继续保证该条件。
- 恢复时若检查点包含调度器状态但当前未构建调度器，维持现有加载行为；训练命令总会构建调度器。

## 验证策略

1. 单元测试验证分层划分的可复现性、掩码不重叠、每个两像素及以上类别同时保留训练与验证像素、单像素类别不被抽入验证。
2. 单元测试验证配置边界和检查点选择状态的前后兼容与恢复。
3. CLI 离线训练烟雾测试验证 `best.pt`、`last.pt`、`test_metrics.json` 生成，调度器写入检查点，测试集评估只在训练停止后进行。
4. 运行全量 pytest、Ruff、格式检查和包构建。
