---
title: "H_part3"
date: 2026-05-11
tags: ["autonomous-driving", "世界模型"]
summary: "来自 自动驾驶 研究笔记"
draft: false
---

# H 经典基础论文分析报告（Part 3）

---

## [7] DIAMOND: Diffusion for World Modeling: Visual Details Matter in Atari

**机构/会议**：日内瓦大学、爱丁堡大学、Microsoft Research / NeurIPS 2024

**核心贡献**（2-3点）：
- 首次将扩散模型（Diffusion Model）用作世界模型的主干，在图像空间直接建模环境动态，规避了离散潜变量压缩导致的视觉信息损失问题
- 提出 DIAMOND（DIffusion As a Model Of eNvironment Dreams），在 Atari 100k 基准上完全在世界模型内训练智能体，取得平均 HNS 1.46 的新 SOTA
- 将扩散世界模型应用于 CS:GO 静态游戏视频，生成可交互神经网络游戏引擎

**方法要点**：
采用 EDM 扩散范式，以 U-Net 2D 为向量场网络 $F_\theta$，将历史观测与动作通过通道拼接和自适应归一化层作为条件输入，迭代去噪预测下一帧观测；奖励与终止信号由独立的 CNN-LSTM 模型预测；智能体在世界模型"想象"中以 REINFORCE + 值网络训练。

**主要结果**：
Atari 100k 基准 26 款游戏，均值 HNS 1.46，IQM 0.64，超越 DreamerV3、IRIS、STORM 等方法，是目前完全在世界模型内训练的智能体中最优结果。

**优缺点**：
- 优点：保留视觉细节，适合视觉敏感任务；扩散模型天然支持多模态分布，不易模式崩塌；世界模型可作为环境替代品直接可视化
- 缺点：扩散推理需多步 NFE，计算成本显著高于单步预测方法；长时序滚动仍存在累积误差

---

## [8] Genie: Generative Interactive Environments

**机构/会议**：Google DeepMind / arXiv 2024-02

**核心贡献**（2-3点）：
- 提出首个"生成式交互环境"范式：仅用无标注互联网视频（无动作标签）训练，即可生成可玩、可控的虚拟世界
- 设计无监督潜在动作模型（LAM），从视频帧对中自动发现并量化离散潜在动作空间（|A|=8），实现帧级别的动作可控生成
- 11B 参数的基础世界模型（Foundation World Model），支持文本/图片/手绘草图等多模态提示，具备良好的扩展性

**方法要点**：
三模块架构：(1) ST-ViViT 视频 tokenizer（VQ-VAE + 时空 Transformer）将视频压缩为离散 token；(2) LAM 编解码器以 VQ 目标从帧差中无监督学习潜在动作；(3) MaskGIT 解码器动态模型根据历史 token 与潜在动作自回归预测下一帧 token，再解码回图像。训练数据为 30k 小时 2D 平台游戏视频（55M clips）。

**主要结果**：
在平台游戏数据集上训练的 11B 模型可通过任意图像提示生成一致的可交互场景；潜在动作空间在不同视频输入中保持语义一致性；模型规模从 40M 到 2.7B 参数均表现出良好的扩展规律。

**优缺点**：
- 优点：无需动作标注，可利用海量互联网视频；通用提示接口灵活；奠定了"基础世界模型"的概念框架
- 缺点：当前仅验证于 2D 平台游戏；生成质量与真实游戏引擎仍有差距；潜在动作含义不可解释

---

## [9] V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video

**机构/会议**：Meta FAIR（INRIA、NYU 等合作）/ arXiv 2024-02

**核心贡献**（2-3点）：
- 将特征预测（Feature Prediction）确立为视频自监督学习的独立有效目标，无需像素重建、文本监督或对比负样本
- 提出 V-JEPA：视频联合嵌入预测架构，仅用特征预测目标在 200 万视频上预训练，冻结主干即可在多种下游任务上达到 SOTA
- 系统性消融验证：特征空间预测优于像素空间预测，且训练效率更高（更短 schedule 下可比甚至超越像素方法）

**方法要点**：
x-encoder 处理未被遮盖的视频块，predictor 根据 x-encoder 输出及目标块的时空位置嵌入，预测被遮盖区域的表示；y-encoder（EMA 更新）生成预测目标；使用 $\ell_1$ 回归 + stop-gradient 防止表示坍塌。遮罩策略采用多块掩码（短程覆盖 15%，长程覆盖 70%），平均遮盖率约 90%。骨干网络为 ViT-L/16、ViT-H/16、ViT-H/16$_{384}$，在 VideoMix2M 数据集上预训练。

**主要结果**：
ViT-H/16 冻结评估：Kinetics-400 81.9%，Something-Something-v2 72.2%，ImageNet-1K 77.9%；在运动理解任务（SSv2）上超越所有同类方法约 +6%；特征预测在标签效率（低标注量）场景下优势更为显著。

**优缺点**：
- 优点：纯视频自监督，无需任何额外标注；冻结骨干即可迁移，实用性强；对细粒度时序理解（SSv2）显著优于像素重建方法
- 缺点：预训练计算量仍较大（ViT-H 级别）；对外观为主的任务（Kinetics）与基于图像的 DINOv2 相比优势有限；尚未扩展到生成/预测下游应用
