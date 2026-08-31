# 在线 ViT 学生分支设计

## 目标

将 HSI 与 LiDAR 两条学生特征提取分支替换为从零训练、端到端更新的 ViT-S/16。冻结 DINOv3 ViT-B/16 结构教师、冻结 RemoteCLIP ViT-L/14 语义教师、文本原型、四尺度门控融合器、解码器和现有蒸馏目标保持不变。

现有 `native` 配置和 `pretrained.yaml` 保留：前者用于离线冒烟，后者用于外部预训练学生对照。新增 `online_vit.yaml` 作为论文主实验模板。

## 学生编码器

新增 `OnlineViTPyramidEncoder`，固定为 ViT-S/16：

- patch size：16；
- embedding dim：384；
- depth：12；
- attention heads：6；
- MLP ratio：4；
- 四个读取层：2、5、8、11；
- 每层 token 去除可选 CLS token 并恢复到 14×14 网格；
- 每个网格特征用独立 `1×1` 投影到融合器所需的四尺度输入通道，随后插值到步长 4、8、16、32 对应分辨率。

两条学生没有外部权重、没有冻结模块，也不依赖 `third_party`。

### HSI 学生

HSI 分支在 patch embedding 前加入显式光谱适配器：`Conv2d(B_hsi, 384, 1)`，其中 `B_hsi` 为当前数据集实际波段数。适配后的 384 通道张量再经 `Conv2d(384, 384, kernel_size=16, stride=16)` 形成 ViT token。

该适配器是学生的一部分，始终参与优化。它将不同数据集的波段维度映射到共同嵌入空间，而不会假设伪 RGB 波段或复用 RGB 权重。

### LiDAR 学生

LiDAR 分支直接采用 `Conv2d(C_lidar, 384, kernel_size=16, stride=16)` patch embedding；`C_lidar` 是数据管线产生的实际 LiDAR 通道数。它不复制单通道为 RGB，也不使用 DINOv3 权重。

## 数据流与蒸馏

```text
HSI [N,B,H,W] -> 光谱适配器 -> Online HSI ViT-S/16 -> 四尺度特征 --+
                                                                        +-> 门控融合 -> 文本空间解码器
LiDAR [N,C,H,W] -> Patch Embedding -> Online LiDAR ViT-S/16 -> 四尺度特征 -+

Pseudo RGB -> 冻结 DINOv3 ViT-B/16 -> 结构蒸馏目标
Pseudo RGB -> 冻结 RemoteCLIP ViT-L/14 -> 语义蒸馏目标 + 类别文本原型
```

训练目标不变：已见类像素交叉熵、两位学生到结构教师的多尺度对齐、融合特征到语义教师的多尺度对齐、HSI/LiDAR 跨模态对齐、门控正则和私有特征正则。只有两个在线学生、融合器、解码器和投影层保存反向图；两个教师始终 `eval()` 且运行在 `torch.no_grad()` 中。

## 配置与兼容性

新增 `encoder.kind: online_vit`，可选字段为：

- `spectral_adapter: true`：仅 HSI 学生允许；
- `model_name: vit_small_patch16`：唯一支持的在线学生结构；
- `feature_blocks: [2, 5, 8, 11]`：唯一支持的四个读取层；
- `checkpoint`、`factory`、`source_dir`、HyperSIGMA 双权重字段必须为空。

`online_vit.yaml` 使用 HSI `spectral_adapter: true` 和 LiDAR `spectral_adapter: false`，并沿用本地 DINOv3 与 RemoteCLIP 教师配置。教师相关权重与源码验证规则不改变。

`_build_visual_encoder` 根据输入通道数构造在线学生，并验证 HSI/LiDAR 分支的 `spectral_adapter` 设置；不改变 native、HyperSIGMA、DINOv3 和 RemoteCLIP 的构建路径。

## 错误处理与测试

- 在线 ViT 输入必须为 NCHW，空间尺寸必须能被 16 整除；
- `online_vit` 只能使用指定模型名与四个读取层；
- HSI 分支必须启用光谱适配器，LiDAR 分支必须关闭它；
- 单元测试覆盖 HSI/LiDAR token 与四尺度输出形状、可训练参数、输入错误、配置拒绝外部字段和 CLI 模型构造；
- 现有 94 项测试及 native CLI 冒烟流程必须继续通过。

## 资源预期

在 224×224、AMP、batch size 1、两个冻结教师下，预期训练峰值约 7–10 GiB。该范围不包含其他进程占用；最终以 CUDA 峰值探测为准。
