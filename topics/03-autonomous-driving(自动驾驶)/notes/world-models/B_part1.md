# B类（3D占用·神经场景型）论文分析报告 Part 1

---

## OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving

**机构/会议**：清华大学自动化系 / ECCV 2024

**核心贡献**：
1. 首次在3D语义占用空间构建世界模型，同时预测周围场景演化与自车运动轨迹；
2. 提出基于VQVAE的3D占用场景分词器，将高维占用体素压缩为紧凑离散token；
3. 设计空间-时间因果自注意力生成Transformer（GPT架构），实现自回归多帧预测。

**方法要点**：两阶段训练——先训练占用场景分词器（VQVAE编码+BEV下采样+向量量化），再训练空间-时间生成Transformer，利用空间聚合与多尺度时序因果注意力预测下一帧场景token及ego位移token；支持LiDAR或纯视觉输入。

**主要结果**：在nuScenes上，3s预测平均IoU 26.63、mIoU 17.13；轨迹规划L2误差1.16m，无需实例或地图标注即达到竞争性规划性能。

**优缺点**：
- 优点：表达细粒度3D结构，统一场景预测与规划，自监督可扩展；
- 缺点：对新驶入视野的车辆预测能力弱，仅依赖占用历史帧存在信息瓶颈。

---

## Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion

**机构/会议**：Waabi / 多伦多大学 / ICLR 2024

**核心贡献**：
1. 提出首个基于离散扩散的无监督点云世界模型，打通"tokenization + 离散扩散生成"范式；
2. 将MaskGIT重新阐释为吸收均匀噪声的离散扩散模型，并以少量改动显著提升其性能；
3. 在BEV空间设计时空Transformer，支持并行token解码，解决自动驾驶大token数的效率瓶颈。

**方法要点**：用VQVAE将点云tokenize为BEV离散latent，解码器采用NeRF式可微深度渲染重建点云；世界模型以时空Transformer在离散token上运行离散扩散（改进MaskGIT训练+采样流程），并支持classifier-free guidance条件生成。

**主要结果**：在NuScenes、KITTI Odometry、Argoverse2上，1s预测Chamfer距离较SOTA降低65–75%，3s预测降低50%以上。

**优缺点**：
- 优点：完全无监督，点云级预测精度大幅领先，支持多模态未来预测；
- 缺点：两阶段（tokenizer + world model）训练较复杂，推理时扩散步骤带来额外计算开销。

---

## SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction

**机构/会议**：清华大学自动化系 / CVPR 2024

**核心贡献**：
1. 首个仅用视频序列（无3D标注）实现环视相机合理3D占用预测的自监督方法；
2. 将3D表示转化为SDF场，通过多视图体积渲染提供自监督信号；
3. 提出MVS-embedded深度学习策略，沿极线扩大深度优化感受野，稳定NeRF式深度收敛。

**方法要点**：图像Backbone提取特征，3D编码器（BEVFormer或TPVFormer）将2D特征提升至BEV/TPV 3D表示；MLP将3D表示解码为SDF场；利用可微体积渲染合成相邻帧颜色/深度作为自监督；辅以Hessian正则、Eikonal约束、时序帧采样策略保证SDF质量；可选接2D分割器实现语义占用。

**主要结果**：SemanticKITTI上以IoU 21.97超越SceneRF 58.7%；Occ3D-nuScenes上mIoU 9.30为首个自监督合理结果；深度估计与新视角合成指标均达SOTA自监督水平。

**优缺点**：
- 优点：无需任何3D标注，可扩展至海量驾驶视频；SDF表示语义/几何双通道；
- 缺点：语义性能与有监督方法差距仍大（mIoU 9.30 vs 28+），语义依赖外部2D分割器。
