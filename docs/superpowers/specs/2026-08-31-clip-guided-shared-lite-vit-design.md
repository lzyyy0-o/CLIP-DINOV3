# CLIP 引导的 Shared Lite-ViT 开放词汇 HSI-LiDAR 分割设计

**日期：** 2026-08-31  
**状态：** 待用户审阅  
**范围：** 以新的 CLIP 引导架构替代当前 online ViT + DINOv3/RemoteCLIP 教师学生配置；既有配置保留为对比基线。

## 1. 目标与边界

本设计面向配准的高光谱影像（HSI）与 LiDAR 栅格（DSM 或多波段 LiDAR）地物分割。模型应：

- 使用 HSI 和 LiDAR 共同学习空间、光谱与高程互补特征；
- 对类别名称和文本提示词形成的动态类别集合输出像素级分割；
- 允许在推理时替换训练类别词表，支持 seen/unseen 开放词汇评估；
- 使用 OpenAI CLIP ViT-B/16，而不是 RemoteCLIP；
- 不保留 DINOv3、结构/语义教师、教师学生蒸馏或其相关损失。

不在本轮目标中：跨数据集端到端联合预训练、非配准 HSI-LiDAR 对齐、三维点云直接建模、全量 CLIP 微调。

## 2. 总体架构

```text
HSI ───┐
       ├→ 6 层 Shared Lite-ViT/16 → ViT-MMFB → ViT-CMFEB → F_joint^(1..4)
LiDAR ─┘                                                          │
                                                                    ├─┐
HSI 伪 RGB → 可微调 OpenAI CLIP ViT-B/16 → F_clip^(1..4)           │ │
                                                                    │ │
类别名称 + 4 个提示词 → 可微调 CLIP 文本塔 → T [N, P, 512] ────────┘ │
                                                                      ↓
             四尺度双相关图 + 类别条件共享解码器 + FPN → logits [B, N, H, W]
```

`N` 不作为任何可训练卷积的固定通道数；它由训练或推理时提供的类别名称列表动态决定。

## 3. HSI-LiDAR 在线主干

### 3.1 输入适配

- HSI 经过仅使用训练区域统计量的逐波段归一化，再经 `1×1` 光谱适配器映射为 384 维 token；
- LiDAR 保持单通道或原始 LiDAR 多通道，经独立 patch embedding 映射为 384 维 token；
- 两模态使用同一组 Transformer 权重，保证共享表征空间，但各自保留输入适配器。

### 3.2 Shared Lite-ViT/16

- patch size：16；embedding dimension：384；attention heads：6；MLP ratio：2；
- 深度配置：`[1, 1, 2, 2]`，共 6 个共享 Transformer block；
- 第 1、2、4、6 个 block 输出四个语义阶段；
- 通过独立投影和尺度转换生成解码器需要的 `1/4、1/8、1/16、1/32` 特征图。

此处的 “Lite-ViT” 是从头训练的六层共享主干，不应称作标准的预训练 ViT-S。

### 3.3 跨模态补强

- **ViT-MMFB：** 双向交叉注意力。HSI token 查询 LiDAR token，LiDAR token 查询 HSI token；每个方向均有残差与 FFN。
- **ViT-CMFEB：** 对更新后的两模态 token 进行拼接、投影、轻量自注意力与 FFN，输出互补联合 token。
- 每个阶段输出联合特征 `F_joint^(l)`，不再使用门控加权融合或 DINO 特征加和。

## 4. OpenAI CLIP ViT-B/16

### 4.1 图像分支

HSI 由 YAML 给出的 RGB 波段映射为伪 RGB，并使用 CLIP 所需的图像归一化。CLIP 图像塔从第 3、6、9、12 个 block 取得 token 特征；这些特征经投影与尺度变换得到 `F_clip^(1..4)`。CLIP 不替代 HSI-LiDAR 主干，而是提供与文本空间对齐的外观语义。

### 4.2 文本分支与文本原型

每个类别使用下列四个提示词：

```yaml
prompt_templates:
  - "aerial image of {}"
  - "satellite image of {}"
  - "top-down view of {}"
  - "remote sensing image of {}"
```

文本编码器输出 `T ∈ R^(N×P×512)`。每条提示词嵌入先 L2 归一化；用于提示词池化或文本引导时，对 `P` 个提示词取均值后再次归一化。训练期间每个前向重新计算 `T` 以保留到文本塔的梯度；评估期间按“类别列表 + 提示词模板”缓存。

类别名称在数据集 YAML 中显式给出，支持不同数据集使用准确名称（例如 `healthy grass`、`residential building`、`vineyard`）。

## 5. MM-OVSeg 式文本条件解码

### 5.1 双相关图

对每尺度 `l`，将 `F_joint^(l)` 与 `F_clip^(l)` 分别投影到 512 维，并与文本特征计算余弦相关：

```text
C_joint^(l)[b,n,p,h,w] = cosine(F_joint^(l)[b,:,h,w], T[n,p])
C_clip^(l) [b,n,p,h,w] = cosine(F_clip^(l) [b,:,h,w], T[n,p])
```

沿提示词维平均后得到每个类别的两张相关图 `C_joint^(l)`、`C_clip^(l)`，形状均为 `[B, N, H_l, W_l]`。

### 5.2 类别条件共享解码器

每类的两张相关图构成 `[B, N, 2, H_l, W_l]`。把 `B,N` 展平为 `[B×N, 2, H_l, W_l]` 后，使用类别共享的相关性嵌入块和自顶向下 FPN 进行解码，再还原为 `[B, N, H, W]`。

这个实现与 MM-OVSeg 的 “图像特征—文本特征相关性 → 文本引导解码” 一致，但没有固定类别数的卷积层、没有 DINO 分支，也不把文本限制为最后的静态分类器权重。

## 6. 训练、评估与参数策略

### 6.1 损失

```text
L_total = L_masked-CE
```

使用掩码多类别交叉熵，忽略标签为 `ignore_index` 的像素。数据集是互斥单标签地物标注，因此不复制 MM-OVSeg 的 BCE 多标签损失。移除全部蒸馏、跨模态 InfoNCE、门控和私有特征正则项。

训练时使用完整训练词表，而不只选当前 tile 出现的类别，确保所有已见类别承担负类约束。推理时使用完整评估词表。

### 6.2 可训练范围与学习率

| 模块 | 参数状态 | 学习率 |
|---|---|---:|
| 光谱适配器、Lite-ViT、MMFB、CMFEB、投影、相关性块、FPN | 全量训练 | `1e-4` |
| CLIP 图像塔第 11–12 block、`ln_post`、视觉投影 | 训练 | `1e-5` |
| CLIP 文本塔最后两个 block、`ln_final`、文本投影 | 训练 | `1e-5` |
| 其余 CLIP 参数 | 冻结 | — |

优化器使用 AdamW；训练配置维持 AMP、`batch_size=1`、梯度裁剪和余弦学习率调度。CLIP 的学习率为新增模块的十分之一，以降低小样本条件下破坏图文对齐的风险。

### 6.3 开放词汇评估

YAML 分别定义 `seen_class_names` 和 `eval_class_names`，并为每个原始标签提供文本名称映射。指标为：

```text
mIoU_seen
mIoU_unseen
hIoU = 2 × mIoU_seen × mIoU_unseen / (mIoU_seen + mIoU_unseen)
```

普通闭集分割实验仍报告 OA、AA、Kappa 与 mIoU；开放词汇主实验不应把 `mIoU_unseen` 缺失时的结果称为开放词汇性能。

## 7. 兼容性、配置与验证

- 新增独立配置 `configs/shared_lite_vit_clip.yaml`，不覆盖 `native.yaml`、`pretrained.yaml` 或 `online_vit.yaml`；
- CLIP checkpoint 使用本地 OpenAI ViT-B/16 权重，并在配置中显式声明路径、模型名和伪 RGB 波段；
- 词表/标签映射不一致、CLIP 输出维度不是 512、非 16 倍输入尺寸、HSI-LiDAR 尺寸不一致时，应在加载或前向前给出明确异常；
- 单元测试覆盖：可变类别数 `N`、相关图形状、文本分支梯度、CLIP 冻结规则、训练/评估词表切换和忽略标签 CE；
- 集成验证包括 CPU 随机张量前向/反向、真实数据单 batch 冒烟训练，以及三套基线配置回归。

## 8. 风险与消融

- 在少量标注下微调文本塔会削弱未见类别能力；主实验采用部分微调，并消融冻结文本塔与全量微调。
- 相关图随类别数线性增加；正常的 6–20 类公开基准可承受，超大词表应分块计算类别相关性。
- 主干深度消融为 Lite-4 `[1,1,1,1]`、Lite-6 `[1,1,2,2]`、Lite-8 `[2,2,2,2]`；Lite-6 是默认主模型。
- 还应比较：仅 `F_joint` 相关图、仅 `F_clip` 相关图、双相关图融合、无 MMFB/CMFEB。
