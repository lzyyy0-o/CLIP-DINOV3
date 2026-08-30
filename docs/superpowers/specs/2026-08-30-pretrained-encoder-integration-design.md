# 预训练编码器离线接入设计

## 目标与范围

本设计将本地保存的 HyperSIGMA、DINOv3 和 RemoteCLIP 源码接入 HSI-LiDAR OVSeg，形成一份可运行的预训练实验模板。现有 Houston 2013、Trento 和 MUUFL 的 `native` 配置及离线冒烟路径保持不变。

本次只接入本地源码和本地权重，不允许构造过程下载模型、配置或权重。数据集和大体积权重不纳入 Git。

## 外部源码边界

源码目录固定为：

- `third_party/hypersigma`：官方 HyperSIGMA，当前修订 `07e9ea24e3072fcb5c3a92a2bcb8185e43b295b9`；
- `third_party/dinov3`：官方 DINOv3，当前修订 `6876159a11b4df116f30f667f8c9888617df0751`；
- `third_party/remoteclip`：官方 RemoteCLIP，当前修订 `a6a4787507e441f444c20404c90dd18520a8960d`。

项目新增受控工厂模块。它接收 `source_dir`、`model_name` 和输入通道参数，临时定位对应官方源码后构造未加载权重的主干；它不会编辑或复制第三方源码。配置中的 `source_dir` 必须存在，且工厂导入失败时应显示缺少的目录或 Python 依赖。

## 配置契约

`EncoderConfig` 扩展以下字段：

- `source_dir`：HyperSIGMA 与 DINOv3 的官方源码目录；
- `spatial_checkpoint`、`spectral_checkpoint`：仅 HyperSIGMA 使用，必须同时存在；
- `pretrained_in_channels`：仅 HyperSIGMA 使用，等于官方权重的输入波段数；
- `input_adapter`：由工厂隐式建立的可训练通道映射，不作为 YAML 可选项。

HyperSIGMA 不再使用通用的 `checkpoint` 字段；DINOv3 仍使用 `checkpoint`；RemoteCLIP 的视觉塔与文本塔继续共用 `clip_checkpoint`。验证逻辑拒绝互斥字段、缺失双权重、无效路径和非正预训练输入波段数。

新增 `configs/pretrained.yaml` 作为真实预训练组合模板：

- HSI 学生：HyperSIGMA Base，空间与光谱权重分开声明；
- LiDAR 学生：DINOv3 ConvNeXt-Tiny；
- 结构教师：冻结 DINOv3 ViT-B/16；
- 语义教师和文本塔：冻结 RemoteCLIP ViT-L/14；
- 默认 tile 为 224、batch size 为 1，所有权重和源码路径均是相对项目根目录的本地路径。

## 模型数据流

### HSI 学生

各数据集 HSI 的实际波段数不同，而 HyperSIGMA 权重的 patch embedding 具有固定输入通道。因此工厂构造时保留官方主干的 `pretrained_in_channels`，并在其前加入可训练的 `1×1` 光谱适配器：`B_dataset -> B_pretrained`。空间分支和光谱分支接收同一适配后的张量。

桥接模块从官方空间分支提取四个中间 NCHW 特征；利用官方光谱分支产生的全局光谱表示生成四个通道门控，逐尺度调制空间特征。然后现有 `HyperSigmaAdapter` 将四层特征投影到解码器的 `feature_dim`，对齐四尺度融合接口。双权重分别严格加载到对应分支；新建适配器和投影层不从官方权重加载。

### LiDAR 学生与结构教师

LiDAR 的实际通道数为 1 或少量派生通道，但 DINOv3 官方视觉权重要求三通道输入。DINOv3 ConvNeXt-Tiny 保留三通道官方输入卷积，其前加入可训练 `1×1` 通道适配器：`C_lidar -> 3`。适配器不改变 DINOv3 主干的四阶段接口，因此现有 `DinoV3ConvNeXtAdapter` 继续获取四层原生特征。

结构教师输入由数据管线生成的伪 RGB，天然为三通道；DINOv3 ViT-B/16 无需输入适配器，并保持冻结。

### 语义教师与文本原型

RemoteCLIP 仍通过 OpenCLIP 以本地权重构造一次。文本塔生成类别原型，同一实例的视觉塔作为冻结语义教师；不从 `third_party/remoteclip` 直接导入模型实现。保留源码目录的目的是记录官方实现、许可证和原始权重说明。

## 冻结与优化

输入通道适配器始终属于学生，参与反向传播和优化。HSI 与 LiDAR 主干仍按现有 `frozen` / `unfreeze_blocks` 策略冻结或解冻尾部块。结构教师和语义教师强制冻结且保持 eval 模式。教师输出只用于蒸馏损失，不生成反向图。

## 错误处理与兼容性

- 工厂不得调用 `torch.hub`、Hugging Face 或任何下载函数；
- 权重加载继续严格匹配官方分支状态字典，错误信息必须指出空间分支、光谱分支或 DINOv3 主干；
- 只有输入适配器和项目投影层允许缺失预训练权重；
- 如果 tile 不满足 DINOv3 patch / ConvNeXt 步幅要求，沿用已有适配器的形状检查；
- `native` 配置不需要第三方源码、额外依赖或预训练权重。

## 测试与验收

新增或更新测试覆盖：

1. 预训练配置的字段、路径和互斥关系校验；
2. 源码目录缺失、工厂不可导入与禁止在线加载的错误路径；
3. 用轻量模拟官方主干验证 HyperSIGMA 双分支加载、HSI 光谱适配器、四尺度输出和冻结策略；
4. 用轻量模拟 ConvNeXt 验证 LiDAR 通道适配器、四尺度输出和部分解冻；
5. 原有 native CLI 冒烟测试及完整测试套件持续通过。

真实预训练运行前，用户需自行将权重放入 `weights/`，安装可选依赖，并用 `validate-config configs/pretrained.yaml` 进行本地文件检查。
