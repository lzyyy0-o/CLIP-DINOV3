# HSI-LiDAR 开放词汇语义分割

这是一个面向配准高光谱影像（HSI）与 LiDAR 栅格的独立 PyTorch 工程。它借鉴 MM-OVSeg 的核心思想：用冻结的视觉/文本语义空间约束任务模态，再通过密集像素嵌入与文本类别原型的相似度完成开放词汇分割。工程支持 Houston 2013、Trento 和 MUUFL Gulfport，并且不会自动下载数据或模型。

## 模型结构

- HSI 学生分支读取完整光谱立方体，论文目标配置使用 HyperSIGMA；项目内置卷积金字塔用于离线测试与消融。
- LiDAR 首先转换为归一化高度、局部相对高度、坡度幅值三个地形通道，论文目标配置使用 DINOv3 ConvNeXt-Tiny。
- HSI 伪 RGB 同时送入冻结 DINOv3 结构教师和冻结 RemoteCLIP 语义教师；前者约束两种学生模态，后者约束融合表示。
- RemoteCLIP 文本塔预先生成归一化类别原型，并与语义教师严格共享同一模型结构和检查点。
- 四层 HSI/LiDAR 特征经过空间门控融合，FPN 解码器把每个像素映射到文本空间。
- 训练目标包含已见类分割、HSI/LiDAR 到结构教师的对齐、融合特征到语义教师的对齐、跨模态对齐、门控与私有特征正则项。

四个视觉编码器统一返回步长为 `4/8/16/32` 的 NCHW 特征金字塔。DINOv3 ConvNeXt 直接保留原生层次；HyperSIGMA、DINOv3 ViT 和 RemoteCLIP ViT 的中间特征由适配器恢复成四尺度表示。两个教师始终冻结并处于 `eval()`，教师前向不记录梯度。

所有外部主干均采用“注入本地模型 + 显式本地检查点”的方式，不调用在线模型仓库。

## 环境与安装

已验证环境为 Python 3.11、PyTorch 2.7.1、CUDA 12.8。项目最低声明为 Python 3.10 和 PyTorch 2.1；建议直接使用已验证组合：

```powershell
conda create -n hsi-lidar python=3.11 -y
conda activate hsi-lidar
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,pretrained]"
```

只运行原生编码器时，可以省略 `pretrained`：

```powershell
pip install -e ".[dev]"
```

验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 数据准备

建议布局如下；文件名可以改变，但必须同步修改 YAML：

```text
data/
├── houston2013/
│   ├── hsi.mat
│   ├── lidar.mat
│   ├── labels.mat
│   ├── train_mask.mat
│   └── test_mask.mat
├── trento/
│   └── ...
└── muufl/
    └── ...
weights/
├── hypersigma.pt
├── dinov2.pt
├── dinov3-convnext-tiny.pt
├── dinov3-vit.pt
└── remoteclip.pt
```

输入支持 `.mat`、`.npy` 和 `.npz`。统一数组契约为：

- HSI：`[H, W, B]`，有限浮点数；
- LiDAR：`[H, W]` 或 `[H, W, C]`；
- 标签：`[H, W]`，`0` 为忽略标签，正类别必须从 `1` 连续编号；
- 训练/测试掩码：`[H, W]` 布尔或 0/1 数组；
- 五个数组必须严格空间配准。

不同公开副本的 MATLAB 键通常不一致。先检查文件，再填写 `hsi_key` 等字段：

```powershell
python -c "from scipy.io import loadmat; print([k for k in loadmat('data/houston2013/hsi.mat') if not k.startswith('__')])"
```

MATLAB v7.3 文件可用：

```powershell
python -c "import h5py; f=h5py.File('data/houston2013/hsi.mat'); print(list(f.keys()))"
```

Houston 2013 的官方竞赛页面给出了 15 类的编号顺序；MUUFL 的数据和场景标签来自官方发布仓库。示例配置保留这些类别顺序，但文件路径、数组键和伪 RGB 波段索引仍需按你的数据副本核对：

- [Houston 2013 官方数据说明](https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/)
- [MUUFL Gulfport 官方数据发布](https://zenodo.org/records/1186326)

## 配置

`configs/houston2013.yaml`、`configs/trento.yaml` 和 `configs/muufl.yaml` 都是完整、可独立修改的示例。先执行结构校验：

```powershell
hsi-lidar-ovseg validate-config configs/houston2013.yaml --skip-file-checks
```

数据和权重准备完成后执行完整校验：

```powershell
hsi-lidar-ovseg validate-config configs/houston2013.yaml
```

三套配置中的已见/未见类划分是项目默认的确定性研究划分，不是社区标准划分。论文实验必须明确报告所用划分；改变划分只需同时修改 `seen_class_ids` 与 `unseen_class_ids`，二者必须互斥并覆盖所有类别。

### 原生离线基线

示例默认使用 `kind: native`，便于在没有外部代码和权重时验证完整流程。`clip_checkpoint: null` 会显式启用确定性的哈希文本原型，只适用于工程冒烟和消融，不具备 CLIP 的语言语义，不应作为开放词汇最终实验结果。

### HyperSIGMA、DINOv3 与 RemoteCLIP 目标配置

外部视觉编码器需提供一个不访问网络的 Python 工厂和本地检查点，例如：

```yaml
model:
  hsi_encoder:
    kind: hypersigma
    factory: third_party.hypersigma:create_model
    model_name: base
    checkpoint: weights/hypersigma.pt
    feature_blocks: [2, 5, 8, 11]
    frozen: false
    unfreeze_blocks: 2
  lidar_encoder:
    kind: dinov3_convnext
    factory: third_party.dinov3:create_model
    model_name: dinov3_convnext_tiny
    checkpoint: weights/dinov3-convnext-tiny.pt
    feature_blocks: [0, 1, 2, 3]
    frozen: false
    unfreeze_blocks: 1
  structure_teacher_encoder:
    kind: dinov3_vit
    factory: third_party.dinov3:create_model
    model_name: dinov3_vitb16
    checkpoint: weights/dinov3-vit.pt
    feature_blocks: [2, 5, 8, 11]
    frozen: true
  semantic_teacher_encoder:
    kind: remoteclip
    model_name: ViT-L-14
    checkpoint: weights/remoteclip.pt
    feature_blocks: [5, 11, 17, 23]
    frozen: true
  clip_checkpoint: weights/remoteclip.pt
  clip_model_name: ViT-L-14
  prompt_templates:
    - "a remote sensing image of {}"
    - "an aerial view of {}"
  feature_dim: 256
  text_dim: 768

loss:
  structure_teacher_weight: 1.0
  semantic_teacher_weight: 1.0
  cross_weight: 0.5
  gate_weight: 0.01
  private_weight: 0.01
  temperature: 0.1
```

工厂格式是 `package.module:callable`。工厂应返回已构造但未加载权重的 `torch.nn.Module`；本项目随后严格加载本地状态字典。HyperSIGMA 和 DINOv3 ViT 主干需公开 `patch_size`、`embed_dim`/`num_features` 和中间层接口；DINOv3 ConvNeXt 需公开 `embed_dims`、四个 `stages`、四个 `downsample_layers` 以及 `get_intermediate_layers`。

`remoteclip` 不配置 `factory`，而是由本地 OpenCLIP 注册结构创建。其 `checkpoint` 必须与 `clip_checkpoint` 相同，`model_name` 若提供则必须等于 `clip_model_name`。CLI 只加载一次完整 RemoteCLIP：先生成文本原型，再把同一实例的视觉塔装入语义教师，避免在显存中保留两份权重。

`clip_model_name` 必须是 OpenCLIP 本地注册的模型结构；为防止隐式联网，配置会拒绝 `hf-hub:` 模型名。自定义视觉工厂同样不得在构造过程中下载权重或配置。

该完整组合明显重于原生配置。以 `224×224`、AMP、batch size 1 为起点较稳妥；显存峰值取决于 HyperSIGMA/DINOv3/RemoteCLIP 的具体规模和解冻层数。若显存不足，依次减小 batch size、把 LiDAR 学生的 `unfreeze_blocks` 设为 `0`、启用梯度累积，再考虑缩小主干。教师虽不保存反向图，仍需驻留权重并保存四层投影特征。

## 训练、恢复与评估

开始训练：

```powershell
hsi-lidar-ovseg train configs/houston2013.yaml
```

从兼容检查点恢复：

```powershell
hsi-lidar-ovseg train configs/houston2013.yaml --resume outputs/houston2013/last.pt
```

整图滑窗评估：

```powershell
hsi-lidar-ovseg evaluate configs/houston2013.yaml outputs/houston2013/best.pt
```

每轮训练都在测试掩码上执行滑窗评估。输出目录包含：

- `last.pt`：最近一轮的完整训练状态；
- `best.pt`：按已见/未见调和 mIoU 选择的最佳状态；
- `predictions.npy`：评估命令生成的一基类别预测图；
- `metrics.json`：mIoU、类别准确率、总体准确率、已见/未见 mIoU、调和 mIoU 和逐类指标。

检查点恢复会校验类别名称及顺序、已见/未见编号、HSI 波段数、LiDAR 模型输入通道数、融合维度和文本维度。任一字段不一致都会拒绝加载。

## 测试与代码规范

测试完全离线，不需要真实数据或预训练权重：

```powershell
python -m pytest -v
ruff check .
ruff format --check .
python -m build
```

项目使用 `src/` 包布局、类型标注、严格 YAML 键校验、训练区统计归一化、同步空间增强、原子检查点和库级 `logging`。标签 `0` 始终被忽略，训练分割损失不会使用未见类标签。

## 研究边界

本仓库不包含或下载 Houston 2013、Trento、MUUFL、HyperSIGMA、DINOv2、CLIP 的数据和权重，也不声明已经复现任何论文数值。真实结果取决于数据版本、预处理、权重、类别划分、提示模板与训练超参数；发表结果时应完整记录这些信息。
