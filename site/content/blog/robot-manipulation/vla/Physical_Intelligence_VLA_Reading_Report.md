---
title: "Physical_Intelligence_VLA_Reading_Report"
date: 2026-05-11
tags: ["robot-manipulation", "VLA 与基础模型"]
summary: "来自 机器人操控 研究笔记"
draft: false
---

# Physical Intelligence VLA 模型族综合阅读报告

## 概述

本报告涵盖 Physical Intelligence 文件夹中的 **9 篇论文**，围绕 **Vision-Language-Action (VLA)** 模型展开。VLA 模型是将视觉观测、语言指令端到端映射为机器人动作的基础模型，代表了机器人操作学习的前沿方向。

论文分为两组：
- **核心论文（4篇）**：π₀、π₀.5、FAST、π₀-FAST —— Physical Intelligence 公司的模型演进线
- **相关改进工作（5篇）**：CogACT、DexGraspVLA、HybridVLA、OpenVLA-OFT、SpatialVLA —— 来自学术界的互补改进

---

## 第一部分：核心论文

### 1. π₀：A Vision-Language-Action Flow Model for General Robot Control

**作者**：Physical Intelligence (Kevin Black, Noah Brown, Danny Driess 等)

**核心贡献**：提出首个基于 **Flow Matching** 的大规模 VLA 基础模型。

**架构设计**：
- **VLM 骨干**：基于 PaLI-Gemma（3B 参数），SigLIP 400M 视觉编码器 + Gemma 2.6B 语言模型
- **动作专家（Action Expert）**：额外的 300M 参数 MLP 层，专门处理机器人状态和动作 token
- **Mixture of Experts 设计**：VLM token 和 action token 使用不同的权重集（two mixture elements），类似 Transfusion 架构
- **总参数量**：3.3B

**Flow Matching 机制**：
- 使用条件 Flow Matching 对动作的连续分布建模
- 训练时：从噪声 $A_t^\tau = \tau A_t + (1-\tau)\epsilon$ 出发，训练网络预测去噪向量场 $v_\theta(A_t^\tau, o_t)$
- 推理时：从随机噪声 $A_t^0 \sim \mathcal{N}(0,I)$ 出发，通过 10 步 Euler 积分生成动作
- Action Chunk：每次预测 H=50 个未来时间步的动作

**训练策略**：
- **预训练数据**：PI 自有灵巧操作数据（903M 步，来自 22 台机器人、7 种机器人配置、68 个任务族）+ OXE 开源数据集（9.1%）
- **后训练**：使用高质量任务特定数据微调
- 预训练/后训练分离策略类比 LLM 的训练范式：预训练提供广泛能力和恢复能力，后训练提供精确流畅执行

**支持的机器人平台**：UR5e、双臂 UR5e、Franka、双臂 Trossen、双臂 ARX/AgileX、Mobile Trossen/ARX、Mobile Fibocom（涵盖 7-18 DoF）

**实验结果**：
- 在 5 个开箱即用任务（衬衫折叠、餐桌清理、杂货打包、烤面包、餐桌设置）上全面超越 OpenVLA 和 Octo
- 即使仅训练 160k 步的"平价版"π₀ 也超越所有基线
- 验证了 VLM 预训练对语言指令跟随的重要性
- 成功完成折叠衣物（5-20分钟多阶段任务）等高难度灵巧操作

**关键洞察**：
- Flow Matching 相比自回归离散化能更好地处理高频（50Hz）灵巧操作
- VLM 初始化对语言跟随和泛化能力至关重要
- 预训练+后训练的分离策略是有效的——多样化低质量数据提供恢复能力，高质量数据提供流畅执行

---

### 2. π₀.5：A Vision-Language-Action Model with Open-World Generalization

**作者**：Physical Intelligence

**核心贡献**：在 π₀ 基础上引入 **高层 VLM 推理** 实现开放世界泛化。

**架构创新——分层策略**：
- **高层 VLM**：接收总体任务指令（如"清理桌子"），生成子指令序列（如"拿起杯子" → "放入垃圾桶"）
- **低层 π₀**：接收子指令，执行具体操作
- 高层 VLM 在每个子任务完成后根据当前观测重新规划

**训练规模**：
- **920K 个 episode**，横跨 **65 个任务族**
- 多机器人形态训练
- 这是迄今最大规模的机器人学习实验之一

**核心能力**：
- **开放世界零样本泛化**：无需微调即可处理训练中未见过的场景、物体和环境
- 高层语义推理 + 低层精确控制的解耦使系统具备组合泛化能力
- VLM 的常识知识使其能理解新物体和新指令

**关键洞察**：
- 单纯扩大低层策略规模不足以实现开放世界泛化，需要高层语义推理
- 分层架构使系统更具可解释性和可调试性

---

### 3. FAST：Efficient Action Tokenization for Vision-Language-Action Models

**作者**：Karl Pertsch, Kyle Stachowicz 等（Physical Intelligence + UC Berkeley + Stanford）

**核心贡献**：提出基于 **离散余弦变换（DCT）** 的动作序列压缩分词方案，解决自回归 VLA 在高频数据上的训练困难。

**问题诊断**：
- 现有 VLA（如 OpenVLA）使用朴素分桶（per-dimension, per-timestep binning）将连续动作离散为 256 个 bin
- 高频动作数据中，相邻 token 高度相关，导致每个 token 的信息增量趋近于零
- 自回归模型在这种情况下退化为简单复制上一个 token，无法学习有效策略

**FAST 分词流程**（5步）：
1. **归一化**：将每个动作维度映射到 [-1, 1]（使用 1st/99th 百分位数）
2. **DCT 变换**：对每个动作维度独立进行离散余弦变换，将时域信号转换为频域表示
3. **量化**：通过缩放（scale γ=10）和四舍五入将 DCT 系数量化为整数，稀疏矩阵
4. **展平**：按列优先（低频优先）将 |A|×H 矩阵展平为 1D 序列
5. **BPE 压缩**：使用字节对编码（词汇表大小 1024）将整数序列压缩为紧凑的动作 token

**压缩效果**：
| 数据集 | 动作维度 | 控制频率 | 朴素分词 token 数 | FAST token 数 | 压缩比 |
|--------|----------|---------|-----------------|-------------|--------|
| BridgeV2 | 7 | 5 Hz | 35 | 20 | 1.75x |
| DROID | 7 | 15 Hz | 105 | 29 | 3.6x |
| Bussing | 7 | 20 Hz | 140 | 28 | 5.0x |
| Shirt Fold | 14 | 50 Hz | 700 | 53 | 13.2x |

**FAST+ 通用分词器**：
- 在约 100 万条真实机器人动作轨迹上训练的通用 BPE 词汇表
- 可作为黑盒分词器应用于任意机器人设置
- 已通过 HuggingFace AutoProcessor 开源发布

**实验结果**：
- 在高频任务（Table Bussing 20Hz, T-Shirt Folding 50Hz）上，朴素分词完全失效，FAST 可正常训练
- 在 DROID 数据集上实现首个成功的零样本泛化策略
- FAST 相比 FSQ（有限标量量化）更简单且效果更好，无需额外神经网络训练

**关键洞察**：
- 动作分词质量是自回归 VLA 性能的关键瓶颈
- 压缩而非简单离散化是正确方向——DCT 基于分析方法，简单、快速、少超参数
- 低频成分优先的展平策略对自回归训练稳定性至关重要

---

### 4. π₀-FAST：Combining π₀ with FAST Action Tokenization

**作者**：Physical Intelligence

**核心贡献**：将 π₀ 架构与 FAST 离散分词结合，用统一的自回归生成替代 Flow Matching 头。

**架构变化**：
- 移除 π₀ 的 Flow Matching action expert
- 使用 FAST 将动作 chunk 编码为离散 token 序列
- 所有 token（图像、语言、动作）通过统一的自回归 next-token prediction 训练
- 架构更简单——无需 Mixture of Experts，无需迭代去噪推理

**核心发现**：
- **离散分词 ≈ 连续 Flow Matching**：在大多数任务上 π₀-FAST 与 π₀ 性能相当甚至更优
- **训练速度提升约 5 倍**：自回归训练效率显著高于 Flow Matching
- 推理更简单：无需多步去噪，单次前向传播即可生成动作 token
- 在 10k 小时机器人数据上验证了大规模可扩展性

**与 π₀ 的对比**：
| 维度 | π₀（Flow Matching） | π₀-FAST（自回归） |
|------|---------------------|-------------------|
| 动作表示 | 连续向量场 | 离散 FAST token |
| 推理 | 10步迭代去噪 | 单次自回归生成 |
| 架构 | MoE（VLM + Action Expert） | 统一 Transformer |
| 训练效率 | 基线 | ~5x 更快 |
| 性能 | 基线 | 相当或更优 |

**关键洞察**：
- 打破了"连续动作需要连续生成方法"的假设
- FAST 分词的质量足以弥合离散/连续的精度差距
- 统一自回归架构简化了系统设计，且便于利用 LLM 的语言理解能力

---

## 第二部分：相关改进工作

### 5. CogACT：A Foundational VLA Model for Synergizing Cognition and Action

**作者**：Tsinghua University + Microsoft Research Asia + USTC

**核心思想**：将 VLM 的"认知"与动作"执行"解耦为两个模块。

**架构设计**：
- **认知模块**：~7B VLM（基于 OpenVLA 的 Prismatic VLM），处理视觉+语言输入，输出"认知 token"（cognition feature）
- **动作模块**：基于 **扩散 Transformer（DiT）** 的动作解码器，以认知 token 为条件生成动作 chunk
- 关键创新：动作模块不直接看原始图像，而是通过认知 token 获取语义信息

**动作模块设计细节**：
- 使用 DiT-Base 作为默认动作模型
- 输入：认知特征 + 带噪声的动作序列 + 去噪步骤信息
- 预测当前 + 未来 N=15 个时间步的动作
- 8 步扩散去噪

**自适应动作集成（Adaptive Action Ensemble, AAE）**：
- 根据历史预测与当前预测的余弦相似度自适应加权
- 避免不同模式动作的不合理聚合

**实验结果**：
- 在 Google Robot（SIMPLER）上平均成功率 74.8%（Visual Matching）和 61.3%（Variant Aggregation）
- 超越 RT-1、RT-1-X、RT-2-X（55B）、OpenVLA
- 真实机器人上超越 OpenVLA 达 59.1%
- **有利的缩放行为**：增大动作模块参数（几百M）带来显著提升，而增大 7B VLM 代价更高

**关键洞察**：
- 认知/动作解耦是高效的 VLA 缩放路线——扩展专用的动作模块比扩展通用 VLM 更划算
- DiT 在动作序列建模上表现出色，且扩展性好
- 顺序建模（autoregressive action generation via DiT denoising steps）优于单步预测

---

### 6. DexGraspVLA：A Vision-Language-Action Framework Towards General Dexterous Grasping

**作者**：Peking University + PKU-PsiBot Joint Lab + HKUST + UPenn

**核心贡献**：首个面向 **灵巧手抓取** 的分层 VLA 框架，实现在 1200+ 未见场景中 90%+ 抓取成功率。

**分层架构**：
- **规划器（Planner）**：基于 Qwen-VL 预训练 VLM
  - 输入自由文本指令（如"清理桌面"）
  - 分解为抓取指令序列（如"抓取蓝色酸奶"）
  - 为每个指令生成目标物体 bounding box 作为 **领域不变的任务表征**
  - 监控执行进度，失败后重新规划
- **控制器（Controller）**：基于 DiT 的扩散策略
  - 输入：bounding box → SAM 分割 → Cutie 追踪 → 掩码
  - 视觉特征：DINOv2（头部相机 + 腕部相机）→ 领域不变表征
  - 输出：13 维动作 chunk（7 DoF 手臂 + 6 DoF 灵巧手）

**领域不变性（Domain Invariance）设计**：
- Bounding box 统一了不同语言和视觉输入的任务表征
- DINOv2 特征在不同环境下保持一致性（对光照、背景、桌面纹理鲁棒）
- 物体掩码追踪提供精确的视觉注意力引导

**数据收集**：2,094 个成功演示，36 种家居物品，杂乱场景，人类演示速度约 3.5 秒/次

**大规模泛化评估**：
- 360 个未见物品、6 个未见背景、3 个未见光照条件
- **Ours@1**：91.1% 未见物品 / 90.5% 未见背景 / 90.9% 未见光照 → 90.8% 总体
- **Ours@3**（允许 3 次尝试）：96.7% / 96.7% / 97.4% → 96.9%
- 长程任务（"清理桌面"多步抓取）成功率 89.6%
- 扩展到非抓握操作（推、拨等）成功率 84.7%

**关键洞察**：
- 在领域不变表征上做模仿学习是实现灵巧手泛化的有效途径
- 基础模型提供的视觉/语言不变性是关键——DINOv2 特征比原始像素鲁棒得多
- 分层设计使规划和执行独立优化，各用最合适的基础模型

---

### 7. HybridVLA：Collaborative Diffusion and Autoregression in a Unified VLA Model

**作者**：Peking University + BAAI + CUHK

**核心贡献**：首个在 **单一 LLM 内** 同时集成扩散和自回归两种动作生成范式的统一框架。

**动机**：
- 自回归 VLA：利用 VLM 的推理能力，但离散化损失精度
- 扩散 VLA：连续动作精确，但未充分利用 VLM 的语义推理
- HybridVLA：两者互补——扩散擅长精细操作，自回归擅长语义理解任务

**架构设计**：
- **基座**：7B LLAMA-2（初始化自 Prismatic VLM）+ DINOv2 + SigLIP 视觉编码器
- **Token 序列设计**：`[视觉 tokens] [语言 tokens] [机器人状态] [<BOD> 扩散 tokens <EOD>] [自回归 tokens]`
- 扩散部分：扩散噪声+去噪步骤信息投影到 LLM 嵌入空间，通过 DDIM 4 步采样
- 自回归部分：在扩散 token 之后生成，条件化于扩散的连续动作表征
- 关键：扩散 token 在前，自回归 token 在后，避免信息泄漏

**协作训练策略（Collaborative Training Recipe）**：
- 混合损失：$L_{hybrid} = L_{dif} + L_{ce}$（扩散 MSE + 自回归交叉熵）
- 两种损失共享 LLM 骨干，联合反向传播
- 扩散为自回归提供连续动作表征条件
- 自回归为扩散提供语义推理辅助

**协作动作集成（Collaborative Action Ensemble）**：
- 推理时同时产生扩散动作 $a^d_{t+1}$ 和自回归动作 $a^{ar}_{t+1}$
- 使用自回归 token 的平均置信度 $c^{ar}_{t+1}$ 判断可靠性
- 若 $c^{ar}_{t+1} > \theta$（阈值 0.96）：取两者均值
- 否则：仅使用扩散动作
- 发现：扩散在精细操作（如"手机放底座"）表现更好，自回归在语义理解任务（如"浇花"）更好

**实验结果**：
- 在 RLBench 模拟中超越现有 SOTA VLA 14%
- 真实世界单臂/双臂任务超越 SOTA 19%
- 对未见物体、背景、空间位置、光照条件展现强泛化能力
- HybridVLA-dif（7B）变体仅使用扩散推理，达到 9.4 Hz

**关键洞察**：
- 离散与连续不必二选一——两者在同一模型中互补增强
- 将扩散去噪嵌入到 next-token prediction 过程中（而非附加独立头）是充分利用 VLM 推理能力的关键
- 自回归 token 的置信度是有效的自适应切换指标

---

### 8. OpenVLA-OFT：Optimized Fine-Tuning for OpenVLA

**作者**：Karl Pertsch 等（Physical Intelligence + Stanford + UC Berkeley）

**核心贡献**：针对 OpenVLA 的三项优化微调改进，使开源 VLA 接近闭源性能。

**OpenVLA 的三个问题**：
1. **顺序解码慢**：7 DoF 动作需要 7 次自回归推理步
2. **离散化精度低**：256 bin 朴素分桶损失精度
3. **微调效率差**：全参数微调或 LoRA 微调效果不理想

**三项改进**：

**(1) 并行解码**：
- 将 7 个动作维度的 token 同时并行预测（而非顺序自回归）
- 各维度仅条件化于视觉+语言 token，不互相条件化
- 推理速度直接提升约 7 倍

**(2) 连续动作损失**：
- 用连续 L1 回归损失替代离散交叉熵损失
- 通过 MLP 头将 LLM 隐状态映射为连续动作值
- 消除了 256 bin 离散化带来的精度上限

**(3) 高效微调**：
- 使用 FiLM（Feature-wise Linear Modulation）条件化注入任务信息
- 比标准 LoRA 更高效，且保留预训练知识

**实验结果**：
- 在 SimplerEnv 模拟中显著超越原始 OpenVLA
- 真实机器人上大幅提升成功率
- 微调数据需求大幅减少
- 推理延迟显著降低

**关键洞察**：
- 并行解码证明动作维度间的条件依赖性并不关键
- 连续损失比离散化损失更适合动作预测
- 这些改进可推广到其他自回归 VLA

---

### 9. SpatialVLA：Exploring Spatial Representations for Visual-Language-Action Model

**作者**：Shanghai AI Lab + Fudan + SJTU + 浙大 + 上海科技大学 + 西工大

**核心贡献**：引入 **3D 空间理解** 到 VLA，通过 Ego3D Position Encoding 和 Adaptive Action Grids 实现空间感知的通用操作策略。

**两大创新**：

**(1) Ego3D Position Encoding（自中心 3D 位置编码）**：
- 使用 ZoeDepth 估计深度图 D
- 结合相机内参将像素反投影为自中心 3D 坐标 $\mathbf{p} = \{x, y, z\}$
- 通过 MLP 将 3D 坐标编码为位置嵌入 $P'$
- 最终：$O_{3d} = X + MLP(\gamma(P))$，将 2D 语义特征与 3D 空间信息融合
- 关键优势：无需外参标定，对不同机器人/相机设置通用

**(2) Adaptive Action Grids（自适应动作网格）**：
- 将连续 7D 动作分为：平移 $\Delta T$（极坐标 $\phi, \theta, r$）+ 旋转 $\Delta R$（roll, pitch, yaw）+ 夹爪
- 对每个动作维度根据训练数据的高斯分布自适应划分网格
- 方向 $(\phi, \theta)$ 用更多 bin，距离 $r$ 用较少 bin
- **仅需 3 个空间动作 token**（平移+旋转+夹爪）代替传统的 7 个 token → 推理速度大幅提升

**Spatial Embedding Adaption（后训练适应）**：
- 为新机器人拟合新的高斯分布 $\mathcal{N}(\mu_{new}, \Sigma_{new})$
- 通过三线性插值从预训练 action token 嵌入初始化新嵌入
- 有效迁移预训练的空间动作知识

**模型规模与训练**：
- 基于 PaLI-Gemma 2 + SigLIP 视觉编码器
- 在 **1.1M 真实机器人 episode** 上预训练（OXE + RH20T 数据集）
- 64 x A100 GPU 训练 10 天

**实验结果**：
- **SimplerEnv Google Robot**：Visual Matching 71.9%、Variant Aggregation 68.8%（超越所有对比方法）
- **SimplerEnv WidowX**：总体 42.7%（超越 RoboVLM 34.4%、OpenVLA 1.0%）
- 真实 Franka 多任务：强零样本表现
- **推理速度 ~20 Hz**（单张 RTX 4090），显著快于 OpenVLA（6.5 Hz）和 TraceVLA（5.2 Hz）
- 仅需 8.5 GB GPU 显存

**关键洞察**：
- 3D 空间理解对机器人操作至关重要——现有 VLA 缺乏深度/空间感知
- 自中心坐标系消除了外参标定需求，提升跨机器人迁移能力
- 自适应动作网格的极坐标分解是优雅的设计——方向精度比距离精度更重要
- 更少的 token = 更快的推理 + 更好的泛化

---

## 第三部分：跨论文分析

### 3.1 核心设计维度：离散 vs. 连续动作表示

这是贯穿所有论文的核心技术争论：

| 方法 | 动作表示 | 代表论文 | 优势 | 劣势 |
|------|---------|---------|------|------|
| 朴素分桶 | 离散 (256 bins) | OpenVLA, RT-2 | 简单，利用 LLM 架构 | 精度低，高频失效 |
| FAST 分词 | 离散 (DCT+BPE) | FAST, π₀-FAST | 高效，高精度压缩 | 需要 BPE 训练 |
| Flow Matching | 连续 | π₀ | 高精度，多模态 | 需迭代去噪，额外参数 |
| 扩散 (DiT) | 连续 | CogACT, DexGraspVLA | 高精度，处理多模态动作 | 计算开销 |
| 混合 | 离散+连续 | HybridVLA | 互补优势 | 架构复杂 |
| 自适应网格 | 离散 (空间感知) | SpatialVLA | 3D 对齐，token 少 | 需要深度估计 |
| 连续回归 | 连续 (L1 loss) | OpenVLA-OFT | 简单，并行解码 | 可能损失多模态性 |

**关键结论**：π₀-FAST 证明了高质量离散分词可匹配连续方法，HybridVLA 证明了两者可互补。最终趋势是：**表示质量比表示类型更重要**。

### 3.2 VLM 骨干选择

| 论文 | VLM 骨干 | 参数量 | 视觉编码器 |
|-----|---------|--------|-----------|
| π₀ / π₀-FAST | PaLI-Gemma | 3B | SigLIP 400M |
| π₀.5 | PaLI-Gemma（扩展） | >3B | SigLIP |
| CogACT | OpenVLA/Prismatic | ~7B | DINOv2 + SigLIP |
| HybridVLA | Prismatic (LLAMA-2) | 7B | DINOv2 + SigLIP |
| DexGraspVLA | Qwen-VL (规划器) | ~7B | DINOv2 (控制器) |
| OpenVLA-OFT | Prismatic | 7B | DINOv2 + SigLIP |
| SpatialVLA | PaLI-Gemma 2 | 3.5B | SigLIP |

**趋势**：DINOv2 + SigLIP 双编码器组合是主流选择，前者提供空间几何特征，后者提供语义对齐特征。

### 3.3 架构范式演进

```
RT-2 (朴素离散化 VLA)
  │
  ├── OpenVLA (开源复现, 7B)
  │     ├── OpenVLA-OFT (并行解码 + 连续损失)
  │     └── CogACT (认知/动作解耦 + DiT)
  │
  ├── π₀ (Flow Matching VLA, 3.3B)
  │     ├── π₀.5 (分层推理)
  │     └── π₀-FAST (FAST 离散分词替代 Flow Matching)
  │           └── FAST (通用动作分词器)
  │
  ├── HybridVLA (扩散+自回归统一)
  ├── DexGraspVLA (灵巧手 + 领域不变性)
  └── SpatialVLA (3D 空间感知 + 自适应网格)
```

### 3.4 关键趋势总结

1. **从单一范式到混合范式**：早期 VLA 选择离散或连续一种方式；HybridVLA 证明两者可在单一模型中协同

2. **从端到端到模块化**：CogACT、DexGraspVLA 都采用认知/动作解耦设计，允许各模块独立优化和缩放

3. **从 2D 到 3D 理解**：SpatialVLA 的 Ego3D 位置编码开辟了空间感知 VLA 的新方向

4. **分词质量是关键**：FAST 证明了好的分词方案可以让简单的自回归架构匹配复杂的扩散方法

5. **预训练规模持续扩大**：从 OXE 的几十万条轨迹到 π₀ 的 903M 步和 SpatialVLA 的 1.1M episode

6. **推理效率受关注**：SpatialVLA（3 token/20Hz）、OpenVLA-OFT（并行解码）、π₀-FAST（无迭代去噪）都在优化推理速度

7. **泛化是核心目标**：π₀.5 的开放世界泛化、DexGraspVLA 的零样本抓取、SpatialVLA 的跨机器人迁移

### 3.5 与 UMI 项目的潜在联系

这些 VLA 模型与 UMI 项目存在互补关系：
- **UMI 提供数据收集管道**，VLA 模型提供策略学习框架
- π₀-FAST / SpatialVLA 等可直接使用 UMI 收集的数据进行训练
- FAST+ 通用分词器已包含 UMI 格式的数据支持（Fig. 8 显示压缩比）
- DexGraspVLA 的灵巧手工作与 DexUMI 互补——前者提供策略框架，后者提供数据收集接口
- SpatialVLA 的 Ego3D 位置编码与 UMI 的相机设计有潜在集成可能

---

## 附录：论文速查表

| # | 论文 | 机构 | 核心关键词 | 动作表示 |
|---|------|------|---------|---------|
| 1 | π₀ | Physical Intelligence | Flow Matching VLA, Action Expert | 连续 (Flow Matching) |
| 2 | π₀.5 | Physical Intelligence | 分层推理, 开放世界泛化 | 连续 (Flow Matching) |
| 3 | FAST | PI + Berkeley + Stanford | DCT 压缩, BPE 分词, 通用分词器 | 离散 (DCT+BPE) |
| 4 | π₀-FAST | Physical Intelligence | 统一自回归, FAST 分词替代 FM | 离散 (FAST) |
| 5 | CogACT | Tsinghua + MSRA | 认知/动作解耦, DiT 动作模块 | 连续 (DiT 扩散) |
| 6 | DexGraspVLA | PKU + HKUST | 灵巧手抓取, 领域不变性 | 连续 (DiT 扩散) |
| 7 | HybridVLA | PKU + BAAI + CUHK | 扩散+自回归统一, 协作集成 | 混合 |
| 8 | OpenVLA-OFT | PI + Stanford + Berkeley | 并行解码, 连续损失, 高效微调 | 连续 (L1 回归) |
| 9 | SpatialVLA | Shanghai AI Lab + Fudan | Ego3D 位置编码, 自适应动作网格 | 离散 (空间 token) |
