# FFN、VGGT 与 3D/4D Gaussian Splatting 在重建任务中的关系

生成日期：2026-05-06  
用途：Notion 导入版 Markdown

## 结论摘要

FFN（Feed-Forward Network，前馈网络）本质上是“输入经过一组确定的层直接得到输出”的神经网络，没有循环状态，也不需要在推理时反复优化同一个样本。它在 3D/4D 重建里有两种常见含义：

1. 狭义 FFN：Transformer block 里的 position-wise MLP 子层，通常是线性层、非线性激活、线性层，用来对每个 token 做通道维度变换。
2. 广义 feed-forward reconstruction：像 VGGT 这样，把多张图像一次送入预训练模型，直接预测相机、深度、点图和轨迹；测试时基本不依赖 COLMAP/SfM/BA/每场景训练。

VGGT 之所以在“重建耗时”上比很多 3DGS/4DGS 快，不是因为它的渲染器比 GS 快，而是因为它把大量几何优化成本前置到了离线训练阶段。部署时，它做的是一次神经网络前向推理；而大多数 3D/4D Gaussian Splatting 流程需要对每个新场景做 COLMAP/SfM 初始化、数千到数万步梯度优化、密度控制、动态场拟合或 deformation 学习。换句话说，VGGT 快在“建模/初始化/几何估计”，GS 强在“训练完成后的高质量实时渲染”。

## 1. FFN 是什么

### 1.1 基本定义

FFN 是一种没有反馈环、没有显式递归状态的网络：输入 `x` 经过若干层映射得到输出 `y`，计算图按层单向流动。典型形式是：

```text
y = W2 * sigma(W1 * x + b1) + b2
```

其中 `sigma` 是 ReLU/GELU/SwiGLU 等非线性激活。MLP、多层感知机、Transformer 中的 MLP block 都可以看成 FFN 的具体形式。

### 1.2 Transformer 里的 FFN

在 Transformer 中，FFN 通常是 attention 子层之后的逐 token 全连接网络。Attention 负责跨 token 聚合信息，FFN 负责在每个 token 内部做通道维度的非线性变换。原始 Transformer 论文把它描述为“position-wise feed-forward network”：同一层内对每个位置使用相同参数，但不同层有不同参数。

这点很关键：Transformer FFN 不负责“看见多视角关系”，多视角关系主要由 self-attention/cross-attention 建立；FFN 更像是每个 token 的局部非线性处理器。

## 2. FFN 在 3D/4D 重建任务里的典型用法

### 2.1 直接几何回归

最直接的用法是把图像、图像 token、体素特征或点云特征输入网络，输出几何量：

| 输入 | FFN/MLP 输出 | 用途 |
|---|---|---|
| 单张图像 | 深度、法线、点图、相机内参先验 | 单目深度、单图 3D 估计 |
| 多视角图像 token | 相机外参/内参、dense depth、point map | SfM/MVS 的学习式替代或初始化 |
| 3D 坐标 + 条件特征 | occupancy、SDF、UDF | 隐式表面重建、mesh extraction |
| 3D 坐标 + view direction | density、color | NeRF/神经辐射场 |
| Gaussian 属性特征 | 位置、旋转、尺度、透明度、SH/color | feed-forward GS 初始化或动态变形 |
| 3D 点 + 时间 | deformation、velocity、visibility | 4D 动态重建、动态 GS |

这种用法的优点是推理快、可批处理、可学习数据先验。缺点是泛化边界由训练数据决定，极端相机、强反光、透明物体、大形变、稀疏纹理区域常常需要额外优化或人工约束。

### 2.2 隐式场解码器

NeRF 是最典型的例子：用一个全连接网络表示连续 5D 辐射场，输入空间位置和视角方向，输出体密度和视角相关颜色。Occupancy Networks 则把 3D 表面表示成神经分类器的连续决策边界。这里的 MLP/FFN 不一定让整个系统“秒级”，因为训练仍可能是每场景优化；但它提供了连续、紧凑、可微的几何/外观表示。

### 2.3 特征提取与几何匹配

在学习式 SfM/MVS 中，FFN 常作为 CNN/ViT backbone 或 Transformer block 的一部分，负责特征投影、token mixing 后的非线性变换、匹配置信度回归、深度概率体解码等。它本身不替代几何约束，但可以提高匹配鲁棒性，减少手工特征和启发式过滤。

### 2.4 优化初始化器

实际工程里，FFN 最常见的价值不是完全取代优化，而是给优化一个更好的起点：

- 预测相机 pose，减少 SfM 搜索和 BA 迭代。
- 预测深度/点图，减少三角化失败和尺度漂移。
- 预测 confidence mask，让后端忽略天空、水面、镜面、动态目标。
- 预测初始 3D Gaussians，缩短 GS warm-up。
- 预测动态 deformation 初值，让 4DGS 更快收敛。

这类 hybrid pipeline 往往比纯 feed-forward 更稳，也比纯几何/纯 GS 训练更快。

## 3. VGGT 是什么

VGGT（Visual Geometry Grounded Transformer）是 CVPR 2025 的视觉几何基础模型。它是一个大型 feed-forward Transformer：输入一张、少量或上百张同一场景图像，直接输出关键 3D 属性，包括：

- 相机内参、外参；
- depth maps；
- point maps；
- 3D point tracks；
- 可选地导出 COLMAP 格式，再接 BA 或 Gaussian Splatting。

VGGT 的核心设计不是引入复杂的 3D 专用模块，而是使用大规模 Transformer backbone，加上 frame-wise/global attention 交替建模多视角关系；然后用 camera head、DPT dense heads、track head 等分支输出不同几何量。

官方 GitHub README 给出的 H100 上 aggregator runtime 是：

| 输入帧数 | 1 | 2 | 4 | 8 | 10 | 20 | 50 | 100 | 200 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 时间/秒 | 0.04 | 0.05 | 0.07 | 0.11 | 0.14 | 0.31 | 1.04 | 3.12 | 8.75 |
| 显存/GB | 1.88 | 2.07 | 2.45 | 3.23 | 3.63 | 5.58 | 11.41 | 21.15 | 40.63 |

注意：这是 backbone/aggregator 的 benchmark；实际总耗时还取决于是否跑 depth/point/track head、是否导出 COLMAP、是否做 BA、是否进行可视化。官方也特别指出，3D 点云可视化可能花几十秒，这不是 VGGT 模型本身的推理耗时。

## 4. 为什么 VGGT 比很多 3D/4D GS 重建流程快

### 4.1 比较对象不同：重建时间 vs 渲染时间

3DGS 和 4DGS 的强项是训练完成后的实时渲染。原始 3DGS 论文目标是高质量、实时 novel-view synthesis，官方实现默认训练迭代数为 30,000，并依赖 SfM/COLMAP 输入。4DGS 官方项目页给出的定位是：在一张 RTX 3090 上，真实动态高分辨率场景约 30 分钟内学习完成，并可 30+ FPS 或更高帧率渲染。

VGGT 的强项是：从图像集合直接得到几何结果。它不需要为每个场景从零优化一个 radiance field 或 Gaussian 场。因此如果比较“我给你 N 张图，多久得到相机、深度、点云/点图”，VGGT 往往是秒级；如果比较“训练完成后每秒渲染多少张高质量新视角图”，GS 往往更强。

### 4.2 VGGT 把优化摊销到了预训练

传统 SfM/MVS/GS 流程在每个新场景上都要解决优化问题：

1. 特征提取与匹配；
2. 相机位姿估计；
3. 三角化/稀疏点云；
4. bundle adjustment；
5. dense reconstruction 或 Gaussian 初始化；
6. 上万步 photometric optimization；
7. densification/pruning/opacity reset；
8. 动态场景还要估计 time-dependent deformation。

VGGT 的策略是用大规模 3D 标注数据离线训练，把这些几何先验吸收到模型参数里。测试时不再为每个场景求解完整优化，而是一次前向推理。这个思路类似“用模型参数记住通用几何求解器”，所以速度优势来自 amortized inference。

### 4.3 一次处理多视角，而不是 pairwise 再全局拼接

DUSt3R/MASt3R 一类方法已经非常接近 feed-forward geometry，但很多版本一次主要处理图像对，后续需要全局对齐或优化来把 pairwise reconstruction 拼成多视角一致结果。VGGT 的 backbone 通过全局 attention 直接在多帧 token 间传播信息，减少 pairwise graph 构建、全局 alignment、反复融合带来的开销。

论文附录 IMC camera pose estimation 表中，VGGT feed-forward 版本运行约 0.2s/scene，VGGT + BA 约 1.8s/scene；同表里的 COLMAP、PixSfM、DUSt3R、MASt3R、VGGSfM/VGGSfMv2 多在 6-20s 范围。这个比较说明：即使加上 BA，VGGT 仍可比很多优化式几何流程快一个量级。

### 4.4 它输出的是几何，不是完整可渲染资产

速度优势也来自任务范围更窄。VGGT 输出相机、深度、点图、tracks，这些是几何中间结果；它不直接给出一个经过 photometric optimization 的高保真 radiance field。GS 则要拟合颜色、透明度、尺度、旋转、球谐系数，并通过渲染损失持续优化，让 novel view 画面尽可能逼真。

所以 VGGT 快的代价是：

- 几何可能不如特定场景深度优化后稳定；
- 复杂材质、透明/反光、极端鱼眼/全景、大旋转、大非刚体运动可能退化；
- 若目标是最终高保真新视角渲染，仍可能需要 GS/NeRF/BA 后处理；
- 输出点云/深度用于测量或建图时，需要额外做尺度、置信度、滤波和坐标系处理。

### 4.5 GS 的每场景优化成本不可忽略

原始 3DGS 官方实现明确包含 PyTorch 优化器、实时 viewer、COLMAP/NeRF Synthetic 数据转换脚本；默认训练迭代为 30,000，并建议 24GB VRAM 以达到论文评测质量。4DGS 还要引入时间维度，通常包含 3D Gaussian + 4D neural voxel/deformation decoder。即使它们渲染快，训练阶段仍是“对每个场景拟合一个表示”。

## 5. 工程选型建议

### 5.1 什么时候用 VGGT

适合：

- 快速从视频帧/照片集得到相机、深度、点云初始化；
- 给 COLMAP/BA/GS 提供初始化；
- 机器人/AR/SLAM 原型中做秒级几何感知；
- 数据清洗：快速判断多视角覆盖、相机是否合理、重建是否可用；
- 大批量场景预处理，追求吞吐而不是每个场景极致画质。

不适合单独承担：

- 高精度测绘级重建；
- 高保真可渲染资产交付；
- 强动态、大形变、反光透明材质主导的场景；
- 严格鱼眼/全景相机，除非专门 fine-tune 或做相机模型适配。

### 5.2 什么时候用 3DGS/4DGS

适合：

- 最终目标是 novel view synthesis 或实时渲染；
- 已有较好相机 pose 和足够覆盖；
- 可接受每个场景数分钟到数十分钟训练；
- 对纹理、视角相关外观、渲染质量要求高。

不适合：

- 只需要快速相机/深度/点云；
- 现场设备算力不足，无法每个场景优化；
- 输入照片很少且 pose 不稳定；
- 动态物体多但没有足够时间拟合 4D 表示。

### 5.3 推荐 hybrid pipeline

如果目标是“既快又要能接后端渲染/建图”，更合理的是：

1. 用 VGGT 做快速相机、深度、点图和 tracks。
2. 导出 COLMAP 格式。
3. 对关键场景或高价值片段做轻量 BA。
4. 用 VGGT 的几何结果初始化 3DGS/4DGS。
5. 对最终需要展示/交付的场景再做 GS photometric optimization。

这样可以把 VGGT 的吞吐优势和 GS 的渲染优势结合起来。

## 6. 一句话对比

| 方法 | 本质 | 快在哪里 | 慢在哪里 | 最终输出 |
|---|---|---|---|---|
| FFN/MLP 层 | 局部非线性映射 | GPU 矩阵乘高效并行 | 单独不建模全局几何 | 特征/隐式值/属性 |
| VGGT | 预训练 feed-forward 几何模型 | 测试时一次前向、多视角直接输出 | 大模型显存高，极端场景泛化有限 | 相机、深度、点图、tracks |
| 3DGS | 每场景优化的显式 Gaussian 表示 | 训练后实时渲染强 | 训练和初始化耗时 | 可实时渲染的静态场景 |
| 4DGS | 带时间/deformation 的动态 Gaussian 表示 | 训练后动态渲染强 | 动态场训练更复杂 | 可实时渲染的动态场景 |
| VGGT + GS | 快速几何初始化 + 渲染优化 | 减少前端几何成本 | 仍需 GS 后端训练 | 几何较稳、渲染较好 |

## 7. 风险与注意事项

- 不要把“feed-forward”理解为“永远比优化准”。它快是因为学习了先验，但失败模式也更依赖训练分布。
- 不要把“VGGT 秒级重建”理解为“秒级得到最终可交付数字孪生”。导出、滤波、BA、mesh/GS 训练、可视化都可能增加耗时。
- 不要用训练完成后的 GS FPS 去反驳 VGGT 的重建速度。两者评估的是不同阶段。
- 对机器人/自动驾驶/测绘等高风险场景，VGGT 输出需要置信度过滤、尺度校准、多传感器校验和后端优化。
- 对 4D 场景，VGGT 论文也承认大幅非刚体变形不是其当前强项；4DGS/动态 NeRF 仍有价值。

## 参考资料

- Transformer FFN：Attention Is All You Need, section 3.3 Position-wise Feed-Forward Networks. https://ar5iv.labs.arxiv.org/html/1706.03762
- VGGT paper：VGGT: Visual Geometry Grounded Transformer. https://ar5iv.labs.arxiv.org/html/2503.11651
- VGGT official GitHub：runtime、COLMAP export、GS integration、license updates. https://github.com/facebookresearch/vggt
- 3D Gaussian Splatting official project page. https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 3DGS official implementation：optimizer、hardware、default iterations. https://github.com/graphdeco-inria/gaussian-splatting
- 3DGS arXiv abstract. https://arxiv.org/abs/2308.04079
- 4D Gaussian Splatting official project page. https://guanjunwu.github.io/4dgs/
- NeRF arXiv：MLP-based neural radiance field. https://arxiv.org/abs/2003.08934
- Occupancy Networks project page：implicit surface as neural classifier decision boundary. https://is.mpg.de/avg/publications/occupancy-networks
