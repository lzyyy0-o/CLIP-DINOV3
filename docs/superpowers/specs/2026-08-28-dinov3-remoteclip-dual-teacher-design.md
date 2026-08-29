# DINOv3 与 RemoteCLIP 双教师设计

## 目标

把现有单教师 HSI-LiDAR 开放词汇分割器升级为以下论文目标结构：

- HSI 学生：HyperSIGMA；
- LiDAR 学生：DINOv3 ConvNeXt-Tiny；
- 结构教师：冻结 DINOv3 ViT；
- 语义教师与文本塔：冻结 RemoteCLIP 的视觉塔和文本塔；
- 融合与解码：保留四尺度门控融合、FPN 和缩放余弦分类。

原生卷积编码器继续作为离线测试和消融基线。所有外部模型必须由本地工厂和本地检查点构建，不允许隐式联网。

## 模型契约

四个视觉编码器都返回尺度为 `1/4, 1/8, 1/16, 1/32` 的四层 NCHW 特征。

- HyperSIGMA 和 DINOv3 ViT 通过中间 token 适配器恢复四层特征。
- DINOv3 ConvNeXt 直接使用四个层次化 stage 的 NCHW 特征。
- RemoteCLIP 使用 OpenCLIP `visual.forward_intermediates` 的四层 NCHW 输出，再恢复到统一尺度。
- HSI 与 LiDAR 特征经 1x1 投影进入共享 `feature_dim`，随后逐层门控融合。
- 两个冻结教师分别投影到相同的 `feature_dim`，但不进入推理主路径。

`SegmentationOutput.alignment_features` 必须包含：

```text
hsi
lidar
structure_teacher
semantic_teacher
fused
```

## 教师与文本一致性

当 `semantic_teacher_encoder.kind=remoteclip` 时：

- `semantic_teacher_encoder.checkpoint` 必须与 `clip_checkpoint` 相同；
- `semantic_teacher_encoder.model_name` 若提供，必须与 `clip_model_name` 相同；
- CLI 只加载一次完整 RemoteCLIP 模型；
- 使用其文本塔生成并归一化类别原型；
- 将同一模型的视觉塔交给语义教师适配器；
- 文本原型生成后不把文本塔保留在分割模型中。

`clip_checkpoint=null` 仍允许原生离线冒烟，但只生成确定性哈希原型，不代表真实开放词汇能力。

## 训练目标

总目标定义为：

```text
L_total = L_seg
        + lambda_structure * (InfoNCE(H, T_structure) + InfoNCE(L, T_structure))
        + lambda_semantic * InfoNCE(F, T_semantic)
        + lambda_cross * InfoNCE(H, L)
        + lambda_gate * L_gate
        + lambda_private * L_private
```

- `L_seg` 仅使用已见类像素；
- 结构教师分别约束 HSI 与 LiDAR 学生；
- 语义教师约束融合特征，避免把单一模态强制压入 RemoteCLIP 空间；
- HSI-LiDAR 对比、门控平衡和私有-共享去相关保持不变。

## 冻结与显存

- 两个教师始终 `requires_grad_(False)` 并保持 `eval()`；
- 教师前向使用 `torch.no_grad()`；
- 仅保存投影后的四层教师特征供损失使用；
- RemoteCLIP 文本原型预先计算，不在每个 batch 重算；
- DINOv3 ConvNeXt 的 `unfreeze_blocks` 表示从最后向前解冻的 stage 数量。

## 兼容性

- `make_native_model` 构造两个冻结的原生教师，保证 CPU 冒烟测试不依赖外部权重；
- 旧的 `teacher_encoder` 配置键被明确替换为 `structure_teacher_encoder` 与 `semantic_teacher_encoder`，严格配置解析不静默猜测；
- 检查点模型状态随新结构变化，旧模型检查点不承诺直接加载；
- 数据预处理、滑窗推理、指标和数据集协议不改变。
