---
title: "World_Model_Survey_Autonomous_Driving"
date: 2026-05-11
tags: ["autonomous-driving", "世界模型"]
summary: "来自 自动驾驶 研究笔记"
draft: false
---

# 自动驾驶领域 World Model 深度综述报告

> **撰写时间**: 2026年3月
> **覆盖范围**: 2018—2026年学术界与工业界主要工作
> **关键词**: World Model, 自动驾驶, 端到端自动驾驶, 仿真, 视频生成, 3D占用预测, 扩散模型

---

## 目录

1. [引言与背景](#1-引言与背景)
2. [经典 World Model 基础](#2-经典-world-model-基础)
3. [技术路线分类](#3-技术路线分类)
4. [第一类：视频生成型 World Model](#4-第一类视频生成型-world-model)
5. [第二类：3D 占用预测型 World Model](#5-第二类3d-占用预测型-world-model)
6. [第三类：端到端自动驾驶中的 World Model](#6-第三类端到端自动驾驶中的-world-model)
7. [第四类：仿真导向型 World Model](#7-第四类仿真导向型-world-model)
8. [第五类：基础模型 / 大规模工业 World Model](#8-第五类基础模型--大规模工业-world-model)
9. [第六类：NeRF / 3DGS 神经场景重建型 World Model](#9-第六类nerf--3dgs-神经场景重建型-world-model)
10. [第七类：World Model + LLM/VLM 融合](#10-第七类world-model--llmvlm-融合)
11. [工业界布局全景](#11-工业界布局全景)
12. [中国研究力量](#12-中国研究力量)
13. [2025—2026 前沿趋势](#13-20252026-前沿趋势)
14. [核心学术争论：视频生成 ≠ 世界理解？](#14-核心学术争论视频生成--世界理解)
15. [定量基准对比](#15-定量基准对比)
16. [自回归 vs 扩散：两大范式深度对比](#16-自回归-vs-扩散两大范式深度对比)
17. [World Model 作为数据引擎](#17-world-model-作为数据引擎)
18. [关键挑战与开放问题](#18-关键挑战与开放问题)
19. [评测基准与数据集](#19-评测基准与数据集)
20. [总结与展望](#20-总结与展望)
21. [附录：全量论文索引表](#21-附录全量论文索引表)

---

## 1. 引言与背景

### 1.1 什么是 World Model

World Model（世界模型）是一种能够**学习环境动力学的内部表示**，使智能体能够在其"想象"中预测未来状态、评估行动后果并据此做出决策的模型。其核心思想源自认知科学——人类大脑持续构建和更新对外部世界的内部模型，从而实现高效的感知、预测和规划。

在自动驾驶语境下，World Model 指的是一类能够：
- **预测未来驾驶场景**（视觉、3D结构、交通参与者行为）
- **模拟不同驾驶动作的后果**（action-conditioned prediction）
- **生成逼真的驾驶场景**用于训练和测试
- **在潜在空间进行规划**（latent imagination for planning）

的深度学习模型。

### 1.2 为什么自动驾驶需要 World Model

| 需求 | World Model 的价值 |
|------|-------------------|
| **安全验证** | 生成海量长尾场景用于仿真测试，替代昂贵的实车路测 |
| **数据增强** | 生成多样化训练数据，解决稀有场景数据不足问题 |
| **端到端规划** | 在想象中"rollout"不同决策路径，选择最优轨迹 |
| **可解释性** | 提供对"自动驾驶系统认为世界会如何变化"的可视化洞察 |
| **闭环仿真** | 替代传统基于规则的仿真器，实现更真实的交互式仿真 |

### 1.3 研究热度

自2023年 Wayve 发布 GAIA-1 以来，自动驾驶 World Model 研究经历了爆发式增长。仅 2024 年一年，相关论文数量超过 **60 篇**，覆盖 CVPR、ICLR、NeurIPS、ECCV、AAAI 等顶级会议。NVIDIA 于2025年1月推出开源 Cosmos 世界基础模型，标志着这一方向从学术研究正式进入工业落地阶段。

---

## 2. 经典 World Model 基础

在深入自动驾驶具体工作之前，有必要回顾对该领域产生深远影响的经典 World Model 工作。

### 2.1 World Models (Ha & Schmidhuber, 2018)

**论文**: "World Models" (NeurIPS 2018, arXiv:1803.10122)

开创性工作，提出了三组件架构：
- **V (Vision Model)**: VAE 将观测压缩为紧凑的潜在表示
- **M (Memory Model)**: MDN-RNN 在潜在空间预测未来状态
- **C (Controller)**: 简单线性控制器将潜在状态映射为动作

**核心贡献**: 首次证明智能体可以**完全在学到的世界模型中训练**（"做梦"），然后将策略迁移到真实环境。在 VizDoom 和 CarRacing 上验证。

### 2.2 PlaNet (Hafner et al., 2019)

**论文**: "Learning Latent Dynamics for Planning from Pixels" (ICML 2019)

引入了对后续工作影响深远的 **RSSM (Recurrent State-Space Model)**：
- 结合确定性和随机性状态分量建模潜在动力学
- 使用 CEM (Cross-Entropy Method) 在潜在空间直接规划
- 相比无模型方法减少 **50倍** 环境交互

### 2.3 Dreamer 系列 (Hafner et al., 2020—2023)

| 模型 | 年份/会议 | 核心创新 |
|------|----------|---------|
| **DreamerV1** | ICLR 2020 | 潜在空间 actor-critic + RSSM，连续高斯潜在状态 |
| **DreamerV2** | ICLR 2021 | 离散分类潜在表示替代连续表示，首个达到人类水平的 model-based RL (Atari) |
| **DreamerV3** | ICLR 2023 | **固定超参数**在 150+ 任务上达到人类水平，引入 symlog 预测。首次在 Minecraft 中从零开始收集钻石 |

### 2.4 IRIS (Micheli, Alonso, Fleuret, 2023)

**论文**: "Transformers are Sample-Efficient World Models" (ICML 2023)

- 使用 VQ-VAE 将游戏帧 tokenize 为离散 token
- **Transformer** 自回归预测未来 token
- 仅 100K 环境交互即在 10/26 个 Atari 游戏达到人类水平

**重要意义**: 桥接了 RL world model 和大语言模型范式——**离散 tokenization + 自回归 Transformer** 成为后续 AD world model 的重要技术路线。

### 2.5 DIAMOND (2024)

使用**扩散模型**作为 world model 进行 RL 训练。代表了自回归之外的另一条技术路线。

### 2.6 Genie / Genie 2 (Google DeepMind, 2024)

- **Genie**: 11B 参数基础 world model，在 200K+ 小时游戏视频上无监督训练，可从单张图片生成可交互环境
- **Genie 2** (2024.12): 扩展到 3D 环境

### 2.7 视频生成基础技术

自动驾驶 World Model 大量借鉴通用视频生成技术：

| 技术路线 | 代表模型 | 方法 |
|---------|---------|------|
| 自回归 token 预测 | VideoGPT, VideoPoet | VQ-VAE/VQGAN tokenize + GPT 式预测 |
| 扩散模型 | Sora (OpenAI), SVD (Stability AI), Runway Gen-3 | 潜在扩散 + 时间建模 |
| 混合方法 | 各种变体 | 结合以上两种范式 |

**OpenAI Sora** 尤其值得关注——OpenAI 官方将其定位为 **"世界模拟器"(world simulator)**，认为视频生成模型可以学习物理世界规律。这一观点极大推动了 AD World Model 研究。

---

## 3. 技术路线分类

自动驾驶 World Model 可按**技术路线**和**应用目标**两个维度进行分类：

### 3.1 按技术路线

```
World Model 技术路线
├── 自回归 (Autoregressive)
│   ├── Token 预测 (GAIA-1, OccWorld, SMART)
│   └── 序列建模 (RNN/RSSM/Transformer)
├── 扩散模型 (Diffusion-based)
│   ├── 连续扩散 (GenAD, Panacea, MagicDrive, Drive-WM)
│   └── 离散扩散 (Copilot4D)
├── GAN (DriveGAN, SurfelGAN) ← 逐渐被取代
├── NeRF / 3D 高斯溅射 (3DGS)
│   ├── NeRF (UniSim, EmerNeRF, MARS)
│   └── 3DGS (MagicDrive3D, RenderWorld, GaussianWorld, Street Gaussians)
├── 占用预测 (Occupancy Prediction)
│   ├── 有监督 (OccWorld, Tesla Occupancy Networks)
│   └── 自监督 (SelfOcc, UnO)
└── 点云/LiDAR 生成 (Copilot4D, LidarDM, LiDARGen)
```

### 3.2 按应用目标

```
World Model 应用目标
├── 视频生成 / 场景生成 (数据增强、仿真)
├── 3D 占用预测 (感知、预测)
├── 端到端规划 (在想象中规划)
├── 闭环仿真 (替代传统仿真器)
└── 基础模型 / 通用平台 (可微调用于多种任务)
```

---

## 4. 第一类：视频生成型 World Model

这是当前最活跃的方向，目标是生成逼真的多视角驾驶视频，用于数据增强和仿真。

### 4.1 核心工作总览

| 论文 | 机构 | 会议/年份 | 技术方法 | 核心贡献 |
|------|------|----------|---------|---------|
| **GAIA-1** | Wayve | arXiv 2023 | 自回归 Transformer + VQ-VAE 视频 tokenizer | 首个大规模驾驶生成式 world model；9B 参数；多模态输入（视频+文本+动作） |
| **DriveDreamer** | GigaAI, CASIA | ECCV 2024 | 视频扩散模型 + 结构化交通条件引导 | 首个完全基于真实世界数据的驾驶 world model |
| **DriveDreamer-2** | GigaAI, CASIA | AAAI 2025 | LLM 生成 HDMap → 统一多视角扩散模型 | LLM 增强的驾驶视频生成；用户可定义天气、道路条件 |
| **DriveDreamer4D** | GigaAI, Tsinghua | CVPR 2025 | World model 视频 + 4D 高斯溅射 | 桥接视频生成与 4D 场景重建 |
| **Drive-WM** | Tsinghua | CVPR 2024 | 扩散模型多视角一致性视频生成 | 首个生成多相机同步一致驾驶视频的 world model |
| **GenAD** | NJU, Shanghai AI Lab | CVPR 2024 | 扩散模型 + 结构先验建模 | 统一预测与规划的生成式框架 |
| **MagicDrive** | CUHK, Huawei | ICLR 2024 | 潜在扩散 + 多条件编码 (3D框, BEV图, 相机位姿) | 精确 3D 几何控制的街景生成 |
| **MagicDrive3D** | CUHK, Huawei | arXiv 2024 | 可变形高斯溅射 + MagicDrive | 扩展到任意视角可渲染的 3D 场景 |
| **Panacea** | Li Auto, USTC, MSRA | CVPR 2024 | 多视角全景视频扩散生成 + BEV/3Dbox/文本条件 | 可控的全景多相机驾驶视频生成 |
| **Vista** | MIT, Waymo Research | CVPR 2025 | 高保真通用驾驶 world model | 动作/语言/场景条件化；强泛化能力；长时间 rollout |
| **WoVoGen** | — | arXiv 2024 | World Volume 表示 + 扩散 | 3D 体素感知的多相机视频生成 |
| **DriveScape** | — | arXiv 2024 | 扩散模型高分辨率多视角生成 | 高分辨率可控多视角驾驶场景 |
| **WorldDreamer** | Tsinghua | arXiv 2024 | 掩码 token 预测 | 区别于扩散和自回归的新范式 |
| **SubjectDrive** | — | CVPR 2025 | 主体一致性驾驶视频生成 | 保持生成视频中物体身份一致性 |
| **InfinityDrive** | — | arXiv 2025 | 无限长度驾驶视频生成 | 解决长时序视频时间一致性 |

### 4.2 关键技术细节

#### GAIA-1 (Wayve, 2023) — 里程碑式工作

GAIA-1 是自动驾驶 World Model 领域的**开山之作**之一：

- **架构**: 视频 tokenizer (VQ-VAE) 将驾驶视频编码为离散 token，文本通过预训练语言模型编码，动作（速度、转向）直接编码。所有模态的 token 拼接为序列，由大规模自回归 Transformer 进行 next-token 预测。
- **规模**: 9B 参数，在 Wayve 伦敦车队数据上训练
- **涌现能力**: 模型自发学习了 3D 几何理解、场景深度感知、交通规则遵守、物体持久性等能力
- **意义**: 证明了 scaling + 自回归生成在驾驶场景的可行性，启发了后续大量工作

#### DriveDreamer 系列 (GigaAI/Shanghai AI Lab) — 中国最具影响力的系列工作

- **DriveDreamer (ECCV 2024)**: 首个完全在真实世界数据上训练的视频扩散 world model。引入结构化交通条件作为生成引导。
- **DriveDreamer-2 (AAAI 2025)**: 创新地使用 LLM 理解和生成交通场景布局 → 扩散模型生成多视角视频。用户可通过自然语言定义场景。
- **DriveDreamer4D (CVPR 2025)**: 将 world model 生成的新视角视频作为监督信号，优化 4D 高斯溅射场景表示。

#### MagicDrive (CUHK + Huawei, ICLR 2024) — 精细控制的典范

- 通过 3D 包围框、BEV 地图、道路拓扑、文本描述等多种条件精确控制生成内容
- 保证多视角几何一致性
- MagicDrive3D 进一步扩展到完整 3D 场景生成

### 4.3 多视角一致性：核心难题

多相机系统是自动驾驶的标配（通常 6 个环视相机），但独立生成每个视角的视频会导致**几何不一致**。以下工作专门解决这一问题：

| 方法 | 策略 |
|------|------|
| Drive-WM | 显式建模多视角时间一致性约束 |
| Panacea | 全景 BEV 条件化确保空间一致 |
| WoVoGen | 引入 3D World Volume 表示 |
| DriveScape | 多相机联合生成 + 交叉注意力 |

---

## 5. 第二类：3D 占用预测型 World Model

与视频生成型关注 2D 图像不同，这一方向在**3D 体素空间**中建模世界的演化。

### 5.1 核心工作总览

| 论文 | 机构 | 会议/年份 | 技术方法 | 核心贡献 |
|------|------|----------|---------|---------|
| **OccWorld** | Tsinghua | ECCV 2024 | GPT 式 Transformer 预测未来 3D 占用 | 从物体级到密集体素级的 world model |
| **Copilot4D** | Waabi, UToronto | ICLR 2024 | VQ-VAE tokenize LiDAR + 离散扩散 | 首个无监督 LiDAR world model |
| **SelfOcc** | Tsinghua | CVPR 2024 | 自监督 3D 占用 (NeRF渲染监督 + SDF) | 无需昂贵 3D 标注 |
| **UnO** | Waabi, UToronto | CVPR 2024 | 神经占用场 (raw LiDAR + ego-motion) | 无监督感知+预测统一框架 |
| **OccSora** | Tsinghua | arXiv 2024 | 扩散 Transformer (DiT) 生成 4D 占用序列 | Sora 范式应用于 3D 占用世界模拟 |
| **RenderWorld** | — | arXiv 2024 | 3DGS 自监督 3D 标签 + world model | 高斯溅射提供几何约束 |
| **GaussianWorld** | Tsinghua | arXiv 2024 | 3DGS 作为流式占用预测的场景表示 | 高效流式 3D 占用预测 |

### 5.2 关键技术分析

#### OccWorld (清华大学, ECCV 2024)

OccWorld 是 3D 占用 world model 的代表作：

- **表示方式**: 将 3D 占用网格 tokenize 为离散 token
- **生成模型**: GPT 式 Transformer 自回归预测未来占用状态和场景流
- **规划集成**: 联合预测 4D 占用和自车运动，直接用于规划
- **优势**: 相比物体级预测，密集体素表示能捕获非结构化物体（路面、植被等）

#### Copilot4D (Waabi, ICLR 2024) — 离散扩散的创新应用

- 使用 VQ-VAE 将 LiDAR 点云编码为离散 token
- 应用**离散扩散模型**（而非连续扩散）预测未来 LiDAR 场景
- 完全无监督——不需要任何人工标注
- 与 Waabi 的仿真平台深度集成

#### OccSora (清华大学, 2024) — Sora 范式在 3D 的应用

- 受 OpenAI Sora 启发，将扩散 Transformer (DiT) 应用于 4D 占用生成
- 生成时空一致的长时序 3D 占用序列
- 作为自动驾驶世界模拟器使用

### 5.3 Tesla 占用网络

Tesla 在 AI Day 2022 展示的占用网络是工业界最早的 3D 占用表示：
- 纯视觉输入 → 体素化 3D 占用网格
- 替代传统逐相机 2D 检测管线
- 虽然 Tesla 未公开论文，但其技术路线极大推动了学术界的占用预测研究

---

## 6. 第三类：端到端自动驾驶中的 World Model

这一方向将 world model 直接集成到端到端自动驾驶系统中，用于规划和决策。

### 6.1 核心工作总览

| 论文 | 机构 | 会议/年份 | 技术方法 | 核心贡献 |
|------|------|----------|---------|---------|
| **MILE** | Wayve (Oxford) | NeurIPS 2022 | World model + 模仿学习；潜在空间推理 | 先驱工作：world model 用于城市驾驶 |
| **UniWorld** | NUDT | arXiv 2023 | 统一预训练框架（world model 学习 3D 表示） | World model 作为预训练策略 |
| **TrafficBots** | ETH Zurich | ICRA 2023 | 多智能体策略学习；个性化条件 (目的地, 驾驶风格) | World model 方法做多智能体交通仿真 |
| **GameFormer** | NTU | NeurIPS 2023 | Level-k 博弈论 + Transformer | 博弈论联合预测与规划 |
| **GUMP** | Waymo | 2024 | 生成式统一运动规划 | 将运动规划视为生成任务 |
| **Think2Drive** | — | ECCV 2024 / AAAI 2025 | 潜在 world model + RL | 在潜在空间"思考"做 RL，CARLA 上有竞争力 |
| **DriveWorld** | Beihang | CVPR 2024 | Memory State Space Model + 4D 预训练 | 4D 预训练提升下游检测、分割、规划 |
| **GenAD** | NJU, Shanghai AI Lab | CVPR 2024 | 扩散式生成框架 | 桥接 world model 生成与端到端规划 |
| **LAW** | Shanghai AI Lab | arXiv 2024 | 可迁移 world model (感知+预测+规划统一) | 学习可迁移的自动驾驶 world model |
| **Doe-1** | — | AAAI 2025 | 大型 world model 统一感知/预测/规划 | 闭环 world model 驾驶 |
| **EMMA** | Waymo | arXiv 2024.11 | 基于 Gemini 的多模态 LLM | 证明 foundation model 可统一驾驶任务 |

### 6.2 关键技术分析

#### MILE (Wayve, NeurIPS 2022) — 端到端 World Model 先驱

- 联合学习**潜在动力学模型**（world model）和**驾驶策略**
- 通过模仿学习训练，无需环境交互
- 在潜在空间中"想象"未来，评估不同轨迹
- 在 CARLA 城市驾驶基准上表现优异
- Wayve 端到端自动驾驶战略的学术基础

#### Think2Drive (ECCV 2024) — RL + World Model

- 在学到的潜在 world model 中进行 RL 训练（"在想象中学习"）
- 大幅降低 RL 所需的环境交互量
- 在 CARLA Leaderboard 上取得有竞争力的结果
- 挑战了模仿学习在 AD 中的主导地位

#### GUMP (Waymo, 2024) — 生成式运动规划

- 将运动规划重新定义为**生成任务**
- World model 预测整个场景的演化，同时作为预测器和规划器
- 代表了 Waymo 在 world model 方向的研究投入

#### EMMA (Waymo, 2024) — 多模态 LLM 作为 World Model

- 基于 Google Gemini 构建
- 将驾驶视为**视觉-语言问题**
- 统一处理规划、检测、路网估计等任务
- 证明了多模态大模型可以作为统一的驾驶 world model

### 6.3 World Model 在端到端 AD 中的三种集成模式

```
模式1：潜在空间规划
  感知 → 编码器 → 潜在 World Model → 潜在空间规划 → 轨迹
  代表：MILE, Think2Drive

模式2：生成式预测+规划
  感知 → World Model 生成多种未来 → 评估/选择最优规划
  代表：GenAD, GUMP, Drive-WM

模式3：统一基础模型
  多模态输入 → 大模型统一处理感知/预测/规划
  代表：EMMA, Doe-1
```

---

## 7. 第四类：仿真导向型 World Model

这一方向旨在替代或增强传统的自动驾驶仿真器（CARLA、SUMO等），使用学习型 world model 生成更真实的仿真环境。

### 7.1 核心工作总览

| 论文 | 机构 | 会议/年份 | 技术方法 | 核心贡献 |
|------|------|----------|---------|---------|
| **UniSim** | Waabi, UToronto | CVPR 2023 | 神经渲染闭环多传感器仿真 | 开创性神经闭环仿真器 |
| **VISTA** | MIT CSAIL | 2024 | 数据驱动神经渲染 + 学习动力学 | 数据驱动仿真引擎 |
| **DriveArena** | ECNU, Shanghai AI Lab | arXiv 2024 | Traffic Manager + World Dreamer 模块化 | 无限驾驶场景的闭环生成式仿真 |
| **LidarDM** | UIUC, MIT | CVPR 2024 | 扩散模型先生成 3D 布局再生成 LiDAR | 生成式 LiDAR 仿真 |
| **SMART** | — | 2024 | Next-token 预测的多智能体轨迹生成 | Waymo WOSAC 挑战赛冠军 |
| **CTG++** | Columbia, NVIDIA, Stanford | 2024 | 场景级条件扩散 + 引导式生成 | 可控交通场景生成（含安全关键场景） |

### 7.2 开环 vs 闭环仿真

| 特性 | 开环 (Open-loop) | 闭环 (Closed-loop) |
|------|-----------------|-------------------|
| **交互性** | 回放式，不响应 agent 动作 | agent 动作影响后续观测 |
| **真实性** | 低（无因果关系） | 高（模拟真实交互） |
| **代表方法** | DriveDreamer, Panacea | UniSim, DriveArena |
| **评测价值** | 视觉质量评估 | 策略安全性评估 |

**关键趋势**: 从开环向闭环演进是当前最重要的方向之一。

### 7.3 UniSim (Waabi) — 闭环仿真的标杆

- 使用神经渲染生成逼真的**多传感器**（相机 + LiDAR）仿真数据
- 自车动作实时影响未来观测——真正的闭环
- 基于真实驾驶日志初始化场景
- 支持 AD 系统的端到端测试

### 7.4 DriveArena — 无限场景生成

- **模块化设计**: Traffic Manager（交通流生成）+ World Dreamer（场景渲染）
- 不依赖预录驾驶日志——可生成**无限**新场景
- 支持闭环交互式驾驶

---

## 8. 第五类：基础模型 / 大规模工业 World Model

### 8.1 核心工作

| 模型 | 组织 | 年份 | 规模与方法 | 核心贡献 |
|------|------|------|----------|---------|
| **GAIA-1** | Wayve | 2023 | 9B 参数；自回归 Transformer | 首个大规模工业驾驶 world model |
| **GAIA-2** | Wayve | 2025 | GAIA-1 升级版；更高质量、更长生成 | 精细可控性（天气、交通密度等） |
| **Cosmos** | NVIDIA | 2025.01 (CES) | 世界基础模型家族；扩散+自回归 | 开源；20M+ 小时视频训练；物理感知 |
| **EMMA** | Waymo | 2024.11 | 基于 Gemini；多模态 LLM | foundation model 统一驾驶任务 |
| **PRISM-1** | Waabi | 2024 | 生成式传感器仿真 (LiDAR + camera) | 仿真优先的自动驾驶（卡车） |
| **Tesla FSD** | Tesla | 2024—2025 | 端到端神经网络 + 隐式 world model | 最大规模真实部署 |

### 8.2 NVIDIA Cosmos (2025) — 里程碑事件

NVIDIA 于 2025年1月 CES 发布的 Cosmos 具有标志性意义：

- **定位**: 世界基础模型 (World Foundation Model, WFM) 平台
- **两类模型**: Cosmos 扩散模型 + Cosmos 自回归模型
- **Cosmos Tokenizer**: 视觉 tokenizer，将图像/视频转换为 token
- **训练数据**: **2000万+小时**视频数据
- **物理感知**: 理解重力、摩擦、惯性等物理规律
- **开源**: GitHub (NVIDIA/Cosmos) 和 HuggingFace 开放权重
- **早期用户**: Toyota, Wayve, Agility Robotics
- **意义**: 标志着 World Model 从学术研究进入**通用基础设施**阶段

---

## 9. 第六类：NeRF / 3DGS 神经场景重建型 World Model

这一方向使用神经辐射场 (NeRF) 或 3D 高斯溅射 (3DGS) 重建驾驶场景的 3D 表示，支持新视角合成和闭环仿真。

### 9.1 核心工作总览

| 论文 | 机构 | 会议/年份 | 技术方法 | 核心贡献 |
|------|------|----------|---------|---------|
| **EmerNeRF** | NVIDIA | ICLR 2024 | 自监督时空场景分解 (静态+动态) | 无需显式监督即可分离动静态场景 |
| **NeuRAD** | Zenseact, Chalmers | CVPR 2024 | 多传感器统一神经渲染 (camera+LiDAR) | 统一框架处理动态场景的闭环仿真 |
| **Street Gaussians** | — | ECCV 2024 | 3DGS + 动态前景物体跟踪分离 | 实时渲染动态城市场景 |
| **UrbanGIRAFFE** | Zhejiang Univ. | ICCV 2023 | 组合式 3D-aware 生成模型 | 可控城市场景生成（相机位姿、物体布局、风格） |
| **S-NeRF** | Fudan Univ. | ICLR 2023 | 街景专用 NeRF | 处理无界场景、动态光照、动态物体 |
| **MARS** | — | 2024 | 实例感知的模块化驾驶场景 NeRF | 模块化场景重建与新视角合成 |
| **READ** | — | AAAI 2024 | 大规模户外场景神经渲染 | 处理大尺度无界驾驶环境 |
| **NeuralAD** | — | CVPR 2024 | 闭环神经渲染 | 自车偏离原始轨迹时仍可重渲染 |
| **LiDAR4D** | — | CVPR 2024 | 4D 动态神经辐射场 (LiDAR) | 时空新视角 LiDAR 合成 |

### 9.2 NeRF → 3DGS 的范式转移

| 维度 | NeRF | 3DGS |
|------|------|------|
| 渲染速度 | 慢（体积渲染，秒级/帧） | **快（光栅化，实时 30+ FPS）** |
| 训练速度 | 慢（数小时） | 快（分钟级） |
| 编辑性 | 困难 | **容易（显式点云表示）** |
| 质量 | 高 | 高（略有不同特征） |
| 动态场景 | 需要额外设计 | 需要额外设计 |

**关键趋势**: 2024年以来，3DGS 正在快速取代 NeRF 成为驾驶场景 world model 中的 3D 表示标准：
- MagicDrive3D、RenderWorld、GaussianWorld、DriveDreamer4D、Street Gaussians 均采用 3DGS
- 实时渲染能力使其更适合交互式闭环仿真

---

## 10. 第七类：World Model + LLM/VLM 融合

大语言模型 (LLM) 和视觉语言模型 (VLM) 与 World Model 的结合是 2024-2025 年的重要新兴方向。

### 10.1 核心工作总览

| 论文 | 机构 | 会议/年份 | 技术方法 | 核心贡献 |
|------|------|----------|---------|---------|
| **ADriver-I** | MEGVII (Face++) | arXiv 2024 | 多模态 LLM 交替预测动作和未来视频 | 同时作为决策者和世界模拟器 |
| **DriveVLM** | Tsinghua | 2024 | VLM + 链式思维推理 → 层次化规划 | 场景描述 → 场景分析 → 规划 |
| **LMDrive** | CUHK, UToronto | CVPR 2024 | LLM + camera/LiDAR → 直接输出控制 | 语言引导的闭环端到端驾驶 |
| **DriveGPT4** | HKU, Huawei | arXiv 2023 | 多帧视频 + 文本查询 → 动作+语言解释 | 可解释的端到端驾驶 |
| **LanguageMPC** | PKU, UC Berkeley | arXiv 2023 | LLM 高层决策 → MPC 参数化执行 | 桥接 LLM 常识推理与 MPC 数学严谨性 |
| **Lingo-1 / Lingo-2** | Wayve | 2023/2024 | 视觉-语言-动作模型 (VLAM) | 语言既是输出（解释）又是输入（指令） |
| **DriveDreamer-2** | GigaAI | AAAI 2025 | LLM 生成场景布局 → 扩散生成视频 | LLM 增强的 world model |
| **EMMA** | Waymo | arXiv 2024 | Gemini 多模态 LLM 统一驾驶任务 | foundation model 即 world model |

### 10.2 三种融合模式

```
模式A：LLM 作为场景生成器
  LLM 理解/生成场景描述 → World Model 生成对应视觉内容
  代表：DriveDreamer-2

模式B：VLM 作为统一驾驶大脑
  视觉+语言输入 → VLM 感知/理解/规划 → 控制输出
  代表：EMMA, DriveVLM, DriveGPT4, LMDrive

模式C：LLM 作为高层决策者 + 传统规划器执行
  LLM 推理场景语义 → 参数化传统规划器 (MPC/PID)
  代表：LanguageMPC
```

### 10.3 Wayve Lingo 系列 — 视觉-语言-动作模型

- **Lingo-1**: 驾驶场景理解 → 生成自然语言评论（为什么减速？为什么变道？）
- **Lingo-2 (2024)**: **闭环** — 语言指令可以调节驾驶行为（"在下个路口右转"），实现真正的双向交互

---

## 11. 工业界布局全景

### 11.1 全球布局一览

| 公司 | 地区 | 核心 World Model 工作 | 战略定位 |
|------|------|---------------------|---------|
| **Wayve** | 英国 | GAIA-1/2, MILE | 端到端 AI 驾驶，world model 为核心 |
| **NVIDIA** | 美国 | Cosmos, DRIVE Sim, DriveGAN | 开源 WFM 平台 + 仿真基础设施 |
| **Waymo** | 美国 | GUMP, EMMA, SMART, WOSAC | 研究驱动，foundation model 路线 |
| **Tesla** | 美国 | 占用网络, 端到端 FSD | 最大规模实车部署 |
| **Waabi** | 加拿大 | Copilot4D, UnO, UniSim, PRISM-1 | 仿真优先的自动驾驶卡车 |
| **Aurora** | 美国 | 大规模仿真管线 | 卡车 + 出行 |
| **Zoox (Amazon)** | 美国 | 仿真驱动测试验证 | 专用 Robotaxi |
| **comma.ai** | 美国 | 探索 world model + 端到端 | 开源民主化自动驾驶 |

### 11.2 Wayve — World Model 战略最坚定的玩家

- 2023年发布 GAIA-1，2025年升级 GAIA-2
- 学术基础：MILE (NeurIPS 2022)
- 获得 SoftBank 10亿美元+投资（2024年）
- 与 Microsoft、Uber 合作
- **核心理念**: World Model 是实现通用自动驾驶的关键路径

### 11.3 Tesla — 最大规模实战部署

- 占用网络：纯视觉 3D 占用表示
- FSD v12/v13：转向端到端神经网络架构，隐式包含 world model
- Dojo 超算支撑大规模训练
- 数十亿英里驾驶数据
- FSD v13 报告干预间隔提升 5-10 倍

---

## 12. 中国研究力量

### 12.1 主要机构

| 机构 | 代表工作 | 核心方向 |
|------|---------|---------|
| **清华大学** (Jiwen Lu 组) | OccWorld, SelfOcc, OccSora, GaussianWorld, WorldDreamer, Drive-WM | 3D 占用 world model 集群 |
| **上海人工智能实验室** | GenAD, LAW, DriveArena | 端到端 AD + 仿真 |
| **GigaAI / CASIA** | DriveDreamer 系列 (1/2/4D) | 最高产 world model 系列 |
| **香港中文大学 + 华为** | MagicDrive, MagicDrive3D | 精细控制场景生成 |
| **南京大学** | GenAD | 生成式端到端 AD |
| **北航** | DriveWorld | 4D 预训练 |
| **国防科技大学** | UniWorld | World model 预训练 |
| **理想汽车** | Panacea, DriveVLM | 全景生成 + VLM 驾驶 |
| **华为** | MagicDrive, ADS 系统 | 场景生成 + 量产 AD |
| **商汤科技** | SenseAuto 仿真 | 仿真平台 |
| **蔚来 / 小鹏** | 内部 world model 探索 | 量产端到端 AD |

### 12.2 中国团队在综述方面的贡献

- Yanchen Guan et al. — "World Models for Autonomous Driving: An Initial Survey" (IEEE T-IV 2025)
- Teng Zhong et al. — "A Survey on World Models for Autonomous Driving"
- Shu Yang et al. — "Driving into the Future: A Comprehensive Survey on World Models in Autonomous Driving"

### 12.3 清华大学占用 World Model 集群

清华大学（尤其是鲁继文组）在**3D 占用 world model** 方向形成了密集的研究集群：

```
清华 Occupancy World Model 系列
├── OccWorld (ECCV 2024) — GPT 式 3D 占用预测
├── SelfOcc (CVPR 2024) — 自监督 3D 占用
├── OccSora (arXiv 2024) — DiT 4D 占用生成
├── GaussianWorld (arXiv 2024) — 3DGS 流式占用
├── WorldDreamer (arXiv 2024) — 掩码 token 预测
└── RenderWorld (arXiv 2024) — 3DGS 自监督 world model
```

### 12.4 GigaAI DriveDreamer 系列

GigaAI（与中科院自动化所合作）打造了最完整的**视频生成 world model 系列**：

```
DriveDreamer 系列演进
DriveDreamer (ECCV 2024)
  └→ 真实世界数据 + 结构化条件引导
DriveDreamer-2 (AAAI 2025)
  └→ + LLM 理解场景 + 多视角统一生成
DriveDreamer4D (CVPR 2025)
  └→ + 4D 高斯溅射 + 新视角生成
```

---

## 13. 2025—2026 前沿趋势

### 趋势 1：4D World Model（空间 + 时间）

从 2D 视频生成向显式 4D（3D 空间 + 时间）表示演进：

- **OccWorld / OccSora**: 体素化 3D 占用随时间演化
- **DriveDreamer4D**: 视频生成 + 4D 高斯溅射
- **Copilot4D**: 4D LiDAR 点云预测
- **DriveWorld**: Memory SSM 4D 预训练

**核心动因**: 2D 视频虽然视觉逼真，但缺乏显式 3D 结构，难以直接用于规划和控制。

### 趋势 2：Tokenized World Model

离散 tokenization 成为主流范式之一：

- **Copilot4D**: VQ-VAE tokenize LiDAR → 离散扩散
- **NVIDIA Cosmos Tokenizer**: 通用视觉 tokenizer
- **GAIA-1**: 视频 token + next-token 预测
- **SMART**: 轨迹 token + 自回归生成
- **OccWorld**: 占用 token + GPT 式预测

**核心洞察**: 将连续的物理世界离散化为 token，使得 LLM 范式（next-token prediction）直接适用于 world model。

### 趋势 3：Action-Conditioned World Model

从被动预测到主动规划：

- **Vista**: 支持转向/速度的动作条件化生成
- **Doe-1**: 闭环动作条件化 world model
- **Think2Drive**: 潜在 world model 中的 RL
- **ACT-Bench**: 专门评估 world model 是否正确响应动作输入的基准

**关键问题**: "向左转"的指令是否真的能让 world model 生成向左转的场景？当前很多模型对动作输入的响应是**微弱或不可靠**的。

### 趋势 4：多模态 World Model（相机 + LiDAR + 更多）

打破单一传感器的限制：

- **UniSim**: 相机 + LiDAR 联合仿真
- **Copilot4D**: LiDAR 原生 world model
- **BEVWorld**: 统一 BEV 潜在空间中的多模态生成

### 趋势 5：长时域生成

- **Vista**: 扩展 rollout 能力
- **InfinityDrive**: 无限长度驾驶视频
- **挑战**: 长序列中误差累积导致时空一致性退化

### 趋势 6：可控生成用于仿真

- **DriveDreamer-2**: LLM 引导的可控场景
- **MagicDrive**: 多条件精细控制
- **CTG++**: 安全关键场景的引导式生成
- **控制维度**: 文本、动作、布局、包围框、HDMap、天气参数

### 趋势 7：World Model 用于闭环评测

- **UniSim → DriveArena → Doe-1**: 从开环到闭环的演进
- **WorldSimBench**: 评估 world model 作为闭环模拟器的能力
- **核心转变**: model 输出反馈为输入（closed-loop），而非一次性预测（open-loop）

### 趋势 8：3D 高斯溅射 (3DGS) 快速渗透

- **MagicDrive3D, RenderWorld, GaussianWorld, Street Gaussians, DriveDreamer4D**
- 3DGS 相比 NeRF 渲染速度提升 10-100 倍
- 正在取代 NeRF 成为 world model 中的默认 3D 表示

---

## 14. 核心学术争论：视频生成 ≠ 世界理解？

这是 World Model 领域最根本的哲学分歧，直接影响技术路线选择。

### 20.1 Yann LeCun 的 JEPA 立场

LeCun 在 2022 年发布的 "A Path Towards Autonomous Machine Intelligence" 中提出了一个完整的认知架构，包含六个模块：世界模型模块、感知模块、代价模块、行动者模块、配置器和短期记忆。其核心论点是：

**"在像素空间预测是根本错误的"**：

1. **容量浪费**: 预测每个像素迫使模型将大量容量用于低级细节（纹理、光照、精确位置），这些对理解和推理毫无价值
2. **模糊-或-幻觉困境**: 面对不确定性，生成模型要么平均所有可能性（产生模糊输出），要么幻觉出特定细节。LeCun 认为这是**本质限制，而非规模问题**
3. **组合爆炸**: 像素级别的未来可能性数量呈天文数字增长
4. **人类认知类比**: 人类思考"球会落在那边"，而不是在脑中渲染每一帧的每个像素

**JEPA (Joint Embedding Predictive Architecture)** 的核心思想：
- 在**学到的抽象表示空间**中预测，而非像素/观测空间
- 仅预测在适当抽象层次上可预测的内容，自然丢弃不相关细节
- 使用**能量函数**框架而非概率/生成框架

**V-JEPA (Meta, 2024.2)**: 自监督视频模型，预测被遮挡视频区域的表示——从不生成像素。在 Kinetics-400、Something-Something-v2 上取得强性能。

### 20.2 "视频生成模型真的是 World Model 吗？"

这一争论在 OpenAI 宣称 Sora 是"世界模拟器"后达到高潮：

**反对方论点**:
- 视频生成模型经常违反基本物理定律（物体凭空出现/消失、重力不一致、碰撞动力学错误）
- 生成视觉逼真的视频 ≠ 学习了底层因果结构
- 这些模型在**像素空间**操作，而非构建抽象的世界因果表示
- **PhysBench** 等物理评估基准发现视频生成模型的物理理解**时有时无**

**支持方论点**:
- 要生成物理合理的视频，模型必须学习某种隐式的世界动力学表示
- 模型确实展示了涌现的 3D 一致性、物理和物体交互能力
- 随着规模增长，物理理解能力持续提升

**中间立场（多数研究者）**:
- 这些模型学习了**近似/部分**世界模型，但不足以称为真正的世界建模
- 真正的世界模型应支持**规划、推理和反事实思考**——而非仅仅产生看起来合理的输出
- **关键测试**: 一个 world model 应该能回答 "如果我执行动作 A 而非 B，世界会如何不同？"

### 20.3 对自动驾驶的启示

| 路线 | 代表 | 哲学立场 |
|------|------|---------|
| **像素空间预测** | GAIA-1, DriveDreamer, Vista | 隐式学习物理（生成派） |
| **潜在空间预测** | MILE, Think2Drive, Dreamer 系列 | 抽象表示中推理（JEPA 派） |
| **结构化 3D 预测** | OccWorld, Copilot4D | 显式 3D 结构理解（结构派） |

**务实结论**: 当前自动驾驶 world model 实际上同时需要**视觉保真度**（用于仿真和数据增强）和**结构化理解**（用于规划）。纯像素预测和纯潜在空间预测各有优劣，最终可能收敛于**分层架构**——高层在抽象空间推理，低层生成具体的视觉/传感器输出。

---

## 15. 定量基准对比

### 15.1 nuScenes 视频生成质量 (FVD / FID)

| 方法 | FID ↓ | FVD ↓ | 多视角 | 视频 | 分辨率 |
|------|-------|-------|--------|------|--------|
| DriveGAN | 73.4 | 502.3 | No | Yes | — |
| DriveDreamer | 52.6 | 452.0 | No | Yes | — |
| WoVoGen | 27.6 | 417.7 | Yes | Yes | — |
| MagicDrive | 16.20 | — | Yes | No | — |
| GenAD | 15.4 | 184.0 | Yes | Yes | — |
| Drive-WM | 15.8 | 122.7 | Yes | Yes | 192×384 |
| **Vista** | **6.9** | **89.4** | **Yes** | **Yes** | **576×1024** |

Vista 在 FID 上超越次优方法 55%（6.9 vs 15.4），FVD 上超越 27%（89.4 vs 122.7）。

### 15.2 CARLA 驾驶评分

| 方法 | 基准 | Driving Score | Route Complete % | 类型 |
|------|------|--------------|-----------------|------|
| Roach (Expert) | CARLA v1 | 84.0 | 95.0 | 模仿学习 |
| **Think2Drive** | **CARLA v1** | **90.2** | **99.7** | **World Model + RL** |
| **Think2Drive** | **CARLA v2** | **56.8** | **98.6** | **World Model + RL** |

Think2Drive 在 CARLA v1 上取得 90.2 驾驶分，在单张 A6000 上仅需 3 天训练。

### 15.3 Waymo WOSAC 仿真真实度

| 方法 | Realism Meta ↑ | Kinematic | Interactive | Map-based |
|------|---------------|-----------|-------------|-----------|
| SMART 101M | **0.7614** | 0.4786 | 0.8066 | 0.8648 |
| SMART 7M | 0.7591 | 0.4759 | 0.8039 | 0.8632 |
| BehaviorGPT | 0.7473 | 0.4333 | 0.7997 | 0.8593 |
| GUMP | 0.7431 | 0.4780 | 0.7887 | 0.8359 |
| TrafficBOTv1.5 | 0.6988 | 0.4304 | 0.7114 | 0.8360 |

SMART (next-token 预测) 在 WOSAC 2024 排名第一，展示了自回归范式在交通仿真中的优势。

### 15.4 OccWorld 4D 占用预测 (nuScenes)

| 方法 | 输入 | mIoU 1s | mIoU 2s | mIoU 3s | mIoU Avg |
|------|------|---------|---------|---------|----------|
| Copy&Paste | 3D-Occ | 14.91 | 10.54 | 8.52 | 11.33 |
| **OccWorld-O** | **3D-Occ** | **25.78** | **15.14** | **10.51** | **17.14** |
| OccWorld-D | Camera | 11.55 | 8.10 | 6.22 | 8.62 |

### 15.5 规划性能对比 (nuScenes, L2 误差 / 碰撞率)

| 方法 | L2 Avg (m) ↓ | Col% Avg ↓ | FPS |
|------|-------------|-----------|-----|
| UniAD | 1.03 | 0.31 | 1.8 |
| VAD-Base | 0.72 | 0.22 | 4.5 |
| **OccWorld-O** | **0.64** | 0.24 | **18.0** |
| Drive-WM | 0.80 | 0.26 | — |

### 15.6 推理速度 / 计算开销

| 模型 | 速度 | GPU | 备注 |
|------|------|-----|------|
| OccWorld-O | **18.0 FPS** | — | 占用输入，最快的 world model 之一 |
| SMART 7M | **17.2 ms/step** | V100 | 最佳性价比 |
| SMART 101M | 46.6 ms/step | V100 | 最高质量 |
| VAD-Tiny | 16.8 FPS | — | 规划基线 |
| UniAD | 1.8 FPS | — | 规划基线 |
| Drive-WM | ~1-5 FPS (估) | A40 | 50 DDIM 步 |
| Vista | — | — | 576×1024@10Hz 输出 |

**关键观察**: 扩散模型 world model 通常 1-5 FPS（因多步去噪），远低于实时要求（20-30 FPS）。自回归/token 模型如 SMART 单步延迟 <50ms，更适合实时部署。

---

## 16. 自回归 vs 扩散：两大范式深度对比

### 16.1 技术特征对比

| 维度 | 自回归 (Autoregressive) | 扩散 (Diffusion) |
|------|----------------------|-----------------|
| **代表工作** | GAIA-1, OccWorld, SMART, Copilot4D | Drive-WM, Vista, DriveDreamer, GenAD, MagicDrive |
| **生成方式** | 逐 token 顺序生成 | 从噪声迭代去噪 |
| **时间一致性** | ✅ 天然优势（顺序依赖） | ❌ 需要额外设计 |
| **单帧质量** | ❌ 较低 | ✅ 更高（Vista FID=6.9） |
| **推理速度** | ✅ 快（单步前向） | ❌ 慢（50+ 去噪步） |
| **动作条件化** | ✅ 自然嵌入序列 | ❌ 较难精确控制 |
| **多样性** | ❌ 有限 | ✅ 固有随机性 |
| **误差累积** | ❌ 长时域退化 | ✅ 全局一致性 |
| **Scaling Law** | ✅ SMART 验证了幂律缩放 | ❓ 尚未系统验证 |

### 16.2 Scaling Law — SMART 的关键发现

SMART (NeurIPS 2024) 首次在自动驾驶 world model 中实证了**类 LLM 缩放定律**：

- 预测精度与模型参数/训练数据/计算量呈**幂律关系**（指数 β = -0.157）
- 从 1M → 7M → 26M → 101M 参数，Realism Meta 持续提升
- 训练从 8 小时（1M）到 1 周（101M），在 32× V100 上

**深远意义**: 如果 LLM 式缩放定律在 world model 中普遍成立，那么"simply scale up"（增加数据和参数）就可以系统性地提升 world model 质量——这为工业界大规模投入提供了理论基础。

### 16.3 OccWorld 消融实验的启示

OccWorld 消融实验 (Table 4) 显示：
- 移除时间注意力 → 预测 mIoU 从 17.14 → 8.98
- 规划碰撞率从 0.60% → 2.56%

这证实了**自回归时间建模**对占用预测和规划的关键作用。

### 16.4 趋势判断

当前看来，两种范式将**长期共存并逐步融合**：
- **扩散模型**主导高保真视觉生成场景（数据增强、仿真渲染）
- **自回归模型**主导需要实时推理和强时间一致性的场景（在线规划、交通仿真）
- **混合架构**（如 Cosmos 同时提供扩散和自回归模型）代表未来方向

---

## 17. World Model 作为数据引擎

### 17.1 数据飞轮范式

World Model 正在成为自动驾驶**数据飞轮 (Data Flywheel)** 的核心组件：

```
┌─────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ 真实驾驶数据  │─────→│  训练 World Model  │─────→│ 生成海量合成数据   │
│ (车队采集)    │      │                  │      │ (含稀有场景)       │
└─────────────┘      └──────────────────┘      └────────┬─────────┘
       ↑                                                │
       │           ┌──────────────────┐                 │
       └───────────│ 部署改进后的模型    │←────────────────┘
                   │ (感知+规划)        │   训练/微调
                   └──────────────────┘
```

### 17.2 工业实践

**Tesla 数据飞轮**:
- 车队持续收集真实驾驶数据，识别问题区域
- World model / 仿真引擎生成数百万合成场景（包含稀有边缘案例）
- 在真实+合成数据上重训练，部署回车队
- Dojo 超算 + 大规模 GPU 集群支撑仿真规模
- 这一循环是 FSD v12+ 开发的核心

**Wayve GAIA 模型**:
- GAIA-1/2 可通过文本条件生成多样场景（"伦敦雨夜驾驶"）
- 无需实车采集即可生成多样化边缘案例
- 获 SoftBank、Microsoft、NVIDIA 支持

### 17.3 三种蒸馏模式

| 模式 | 描述 | 代表工作 |
|------|------|---------|
| **数据增强** | World model 生成多样场景补充真实训练数据 | DriveDreamer, DriveDreamer-2 |
| **闭环训练** | World model 提供交互式反馈用于规划器训练 | UniSim, Think2Drive |
| **知识压缩** | 大型 world model 蒸馏为轻量可部署规划器 | MILE |

### 17.4 解决长尾问题

World Model 作为数据引擎的最大价值在于解决**长尾分布**问题：
- 99% 的驾驶场景是常规的，但安全关键的 1% 极难收集
- CTG++ 等工作专门生成**安全关键场景**（紧急制动、行人突然横穿等）
- World model 可以系统性地探索场景空间，发现传统测试无法覆盖的 corner case

---

## 18. 关键挑战与开放问题

### 18.1 几何 / 3D 一致性

视频生成型 world model 最大的痛点：
- 生成内容常违反 3D 几何约束（物体漂浮、透视错误）
- 6 相机环视的多视角一致性仍然困难
- 深度关系和空间关系不够精确

### 18.2 时间一致性与长时域稳定性

- 帧间闪烁、物体消失/重现
- 自回归 rollout 的误差累积问题
- 长时间维持物体身份一致性尚未解决

### 18.3 Sim-to-Real 鸿沟

- World model 生成的场景在物理层面仍与真实世界有差距
- 视觉逼真 ≠ 物理逼真
- 在 world model 仿真中训练的策略迁移到真车仍有挑战

### 18.4 动作可控性

- 许多 world model 生成看似合理的未来，但对特定动作输入的响应**不忠实**
- ACT-Bench 暴露了这一问题
- World model 能否作为可靠的"想象引擎"用于规划仍存争议

### 18.5 评测标准不统一

| 指标 | 评测内容 | 局限 |
|------|---------|------|
| FVD (Frechet Video Distance) | 视频质量与时间一致性 | 不评估物理合理性 |
| FID (Frechet Inception Distance) | 单帧质量 | 不评估时间动态 |
| LPIPS | 感知相似度 | 像素级，非语义级 |
| L2 误差 | 规划轨迹偏差 | 单一维度 |
| 碰撞率 | 安全性 | 依赖仿真环境 |

**共识**: 单纯的视觉质量指标 (FVD/FID) 不足以评估驾驶 world model。需要结合**下游规划性能**和**物理合理性**的综合评测。

### 18.6 计算规模与实时推理

- 训练高保真 world model 需要巨大算力（Cosmos: 2000万小时视频）
- 车载实时推理极具挑战
- 潜在空间 world model (Think2Drive, MILE) 以保真度换取效率

### 18.7 稀有 / 长尾事件

- 常见场景训练的 world model 难以处理罕见边缘案例
- 安全关键事件（近失事故、异常障碍物）在训练数据中严重不足
- 如何可靠地生成多样化 corner case 仍是开放问题

### 18.8 感知-预测-规划的紧密集成

- 大多数 world model 仍然是独立的预测/生成系统
- 与下游规划模块的紧密集成仍处于早期
- 在像素空间、潜在空间还是占用空间中规划？尚无定论

---

## 19. 评测基准与数据集

### 19.1 感知 / 场景理解数据集

| 数据集 | 提供者 | 关键特点 |
|--------|-------|---------|
| **nuScenes** | Motional | 1000 场景, 1.4M 相机图, 390K LiDAR 扫描 |
| **Waymo Open Dataset** | Waymo | 最大最多样化之一；高质量 LiDAR + 相机 |
| **KITTI** | KIT/Toyota | 经典基准；3D 检测、跟踪、场景流 |
| **Argoverse 2** | Argo AI | 多城市；运动预测、LiDAR 场景流 |
| **OpenScene** | nuScenes 生态 | 大规模占用预测和 3D 场景理解 |

### 19.2 规划 / 仿真基准

| 基准 | 关键特点 |
|------|---------|
| **nuPlan** (Motional) | 全球首个大规模规划基准；闭环仿真 |
| **CARLA Leaderboard** | 开源仿真器；路线完成率、违规分、驾驶分 |
| **Bench2Drive** | 基于 CARLA v2；端到端 AD 闭环评估 |
| **NAVSIM** | 视觉驾驶策略非反应式评估 |
| **Waymo WOSAC** | 多智能体交通仿真基准 |
| **WorldSimBench** | World model 作为仿真器的评测（AAAI 2025） |
| **ACT-Bench** | 评估动作条件化准确性 |

---

## 20. 总结与展望

### 20.1 领域现状总结

自动驾驶 World Model 经历了从 2023 年**概念验证** (GAIA-1) 到 2025 年**工业平台** (Cosmos) 的快速演进。当前格局：

1. **扩散模型主导生成**: 大多数视频/场景生成 world model 采用扩散架构
2. **自回归 token 预测紧随其后**: GAIA-1/2, OccWorld, SMART 等展示了 LLM 范式在 world model 中的潜力
3. **3DGS 正在取代 NeRF**: 在 3D 表示方面，高斯溅射因其渲染速度优势快速渗透
4. **闭环 > 开环**: 从单纯视频生成向可交互闭环仿真演进
5. **学术-工业高度联动**: Wayve、Waabi、Waymo 的学术论文直接推动工业产品

### 20.2 未来展望

**短期 (2025—2026)**:
- NVIDIA Cosmos 生态将推动 world model 的标准化和普及
- 4D world model (3DGS/占用) 将成为主流
- 闭环仿真 world model 将逐步替代传统仿真器
- Action-conditioned 世界模型的可靠性将显著提升

**中期 (2026—2028)**:
- World model 将成为端到端自动驾驶系统的**标准组件**
- 统一的感知-预测-规划 world model 架构趋于收敛
- 大规模开源 world model 将降低自动驾驶研发门槛
- World model 驱动的数据飞轮（生成数据 → 训练 → 更好的生成）将加速迭代

**长期愿景**:
- 通用世界模型（AGI 级别的环境理解）
- World model 不仅服务于自动驾驶，还将赋能具身智能、机器人等领域
- 从"模仿世界的外观"到"理解世界的因果结构"

### 20.3 关键结论

> **World Model 正在从自动驾驶的"锦上添花"转变为"不可或缺"。** 当前最大的挑战不再是"能否生成逼真的驾驶视频"，而是"world model 是否真正提升了自动驾驶的规划安全性"——这一问题仍然大程度开放。

---

## 21. 附录：全量论文索引表

### A. 视频生成型

| # | 论文 | 机构 | 年份 | 技术 | 会议 |
|---|------|------|------|------|------|
| 1 | GAIA-1 | Wayve | 2023 | 自回归 Transformer + VQ-VAE | arXiv |
| 2 | GAIA-2 | Wayve | 2025 | GAIA-1 升级版 | — |
| 3 | DriveDreamer | GigaAI, CASIA | 2024 | 视频扩散 + 结构化条件 | ECCV |
| 4 | DriveDreamer-2 | GigaAI, CASIA | 2025 | LLM + 多视角扩散 | AAAI |
| 5 | DriveDreamer4D | GigaAI, Tsinghua | 2025 | World model + 4D GS | CVPR |
| 6 | Drive-WM | Tsinghua | 2024 | 多视角扩散 | CVPR |
| 7 | GenAD | NJU, Shanghai AI Lab | 2024 | 扩散 + 结构先验 | CVPR |
| 8 | MagicDrive | CUHK, Huawei | 2024 | 多条件潜在扩散 | ICLR |
| 9 | MagicDrive3D | CUHK, Huawei | 2024 | 可变形 GS + 扩散 | arXiv |
| 10 | Panacea | Li Auto, USTC, MSRA | 2024 | 全景多视角扩散 | CVPR |
| 11 | Vista | MIT, Waymo | 2025 | 通用可控 world model | CVPR |
| 12 | WoVoGen | — | 2024 | World Volume + 扩散 | arXiv |
| 13 | DriveScape | — | 2024 | 高分辨率多视角扩散 | arXiv |
| 14 | WorldDreamer | Tsinghua | 2024 | 掩码 token 预测 | arXiv |
| 15 | SubjectDrive | — | 2025 | 主体一致性生成 | CVPR |
| 16 | InfinityDrive | — | 2025 | 无限长度视频生成 | arXiv |
| 17 | Delphi | — | 2024 | 长距离预测蒸馏 | arXiv |
| 18 | DOSE | — | 2025 | Diffusion Dropout + EMD | CVPR |

### B. 3D 占用 / 神经场景型

| # | 论文 | 机构 | 年份 | 技术 | 会议 |
|---|------|------|------|------|------|
| 1 | OccWorld | Tsinghua | 2024 | GPT 式占用预测 | ECCV |
| 2 | Copilot4D | Waabi, UToronto | 2024 | VQ-VAE + 离散扩散 (LiDAR) | ICLR |
| 3 | SelfOcc | Tsinghua | 2024 | 自监督占用 (NeRF+SDF) | CVPR |
| 4 | UnO | Waabi, UToronto | 2024 | 无监督占用场 | CVPR |
| 5 | OccSora | Tsinghua | 2024 | DiT 4D 占用生成 | arXiv |
| 6 | RenderWorld | — | 2024 | 3DGS 自监督 | arXiv |
| 7 | GaussianWorld | Tsinghua | 2024 | 3DGS 流式占用 | arXiv |

### C. 端到端自动驾驶集成型

| # | 论文 | 机构 | 年份 | 技术 | 会议 |
|---|------|------|------|------|------|
| 1 | MILE | Wayve | 2022 | World model + 模仿学习 | NeurIPS |
| 2 | UniWorld | NUDT | 2023 | World model 预训练 | arXiv |
| 3 | TrafficBots | ETH Zurich | 2023 | 多智能体个性化策略 | ICRA |
| 4 | GameFormer | NTU | 2023 | 博弈论交互建模 | NeurIPS |
| 5 | GUMP | Waymo | 2024 | 生成式运动规划 | — |
| 6 | Think2Drive | — | 2024 | 潜在 world model + RL | ECCV/AAAI |
| 7 | DriveWorld | Beihang | 2024 | Memory SSM 4D 预训练 | CVPR |
| 8 | LAW | Shanghai AI Lab | 2024 | 可迁移 world model | arXiv |
| 9 | Doe-1 | — | 2025 | 闭环大型 world model | AAAI |
| 10 | EMMA | Waymo | 2024 | Gemini 多模态 LLM | arXiv |

### D. 仿真导向型

| # | 论文 | 机构 | 年份 | 技术 | 会议 |
|---|------|------|------|------|------|
| 1 | UniSim | Waabi, UToronto | 2023 | 神经闭环多传感器仿真 | CVPR |
| 2 | VISTA | MIT CSAIL | 2024 | 数据驱动神经渲染仿真 | — |
| 3 | DriveArena | ECNU, Shanghai AI Lab | 2024 | 模块化闭环生成式仿真 | arXiv |
| 4 | LidarDM | UIUC, MIT | 2024 | 扩散 LiDAR 生成 | CVPR |
| 5 | SMART | — | 2024 | Next-token 多智能体仿真 | — |
| 6 | CTG++ | Columbia, NVIDIA, Stanford | 2024 | 引导式交通场景扩散 | — |
| 7 | DriveGAN | NVIDIA | 2021 | VAE + GAN 神经仿真器 | — |

### E. 基础/工业模型

| # | 模型 | 组织 | 年份 | 类型 |
|---|------|------|------|------|
| 1 | GAIA-1/2 | Wayve | 2023/2025 | 自回归 world model |
| 2 | Cosmos | NVIDIA | 2025 | 开源 WFM 平台 |
| 3 | EMMA | Waymo | 2024 | 多模态 LLM 驾驶 |
| 4 | PRISM-1 | Waabi | 2024 | 生成式传感器仿真 |
| 5 | Tesla FSD | Tesla | 2024-25 | 端到端隐式 world model |

### F. NeRF / 3DGS 神经场景重建型

| # | 论文 | 机构 | 年份 | 技术 | 会议 |
|---|------|------|------|------|------|
| 1 | EmerNeRF | NVIDIA | 2024 | 自监督时空分解 NeRF | ICLR |
| 2 | NeuRAD | Zenseact, Chalmers | 2024 | 多传感器统一神经渲染 | CVPR |
| 3 | Street Gaussians | — | 2024 | 3DGS 动态城市场景 | ECCV |
| 4 | UrbanGIRAFFE | Zhejiang Univ. | 2023 | 组合式 3D-aware 生成 | ICCV |
| 5 | S-NeRF | Fudan Univ. | 2023 | 街景 NeRF | ICLR |
| 6 | MARS | — | 2024 | 模块化实例感知 NeRF | — |
| 7 | READ | — | 2024 | 大规模户外神经渲染 | AAAI |
| 8 | NeuralAD | — | 2024 | 闭环神经渲染仿真 | CVPR |
| 9 | LiDAR4D | — | 2024 | 4D LiDAR 神经辐射场 | CVPR |
| 10 | MagicDrive3D | CUHK, Huawei | 2024 | 可变形 GS + 场景生成 | arXiv |
| 11 | RenderWorld | — | 2024 | 3DGS 自监督 world model | arXiv |
| 12 | GaussianWorld | Tsinghua | 2024 | 3DGS 流式占用预测 | arXiv |
| 13 | DriveDreamer4D | GigaAI, Tsinghua | 2025 | WM 视频 + 4D GS | CVPR |

### G. World Model + LLM/VLM 融合型

| # | 论文 | 机构 | 年份 | 技术 | 会议 |
|---|------|------|------|------|------|
| 1 | ADriver-I | MEGVII | 2024 | MLLM 交替预测动作+视频 | arXiv |
| 2 | DriveVLM | Tsinghua | 2024 | VLM + CoT 推理 → 层次规划 | — |
| 3 | LMDrive | CUHK, UToronto | 2024 | LLM 闭环端到端驾驶 | CVPR |
| 4 | DriveGPT4 | HKU, Huawei | 2023 | 多帧视频 + 文本 → 可解释控制 | arXiv |
| 5 | LanguageMPC | PKU, UC Berkeley | 2023 | LLM 决策 → MPC 执行 | arXiv |
| 6 | Lingo-1/2 | Wayve | 2023/2024 | 视觉-语言-动作模型 (VLAM) | — |
| 7 | DriveDreamer-2 | GigaAI | 2025 | LLM 场景布局 → 扩散视频 | AAAI |
| 8 | EMMA | Waymo | 2024 | Gemini 多模态 LLM 统一驾驶 | arXiv |

### H. 经典 World Model 基础

| # | 论文 | 机构 | 年份 | 技术 | 会议 |
|---|------|------|------|------|------|
| 1 | World Models | Google Brain | 2018 | VAE + MDN-RNN + Controller | NeurIPS |
| 2 | PlaNet | DeepMind | 2019 | RSSM + CEM 潜在规划 | ICML |
| 3 | DreamerV1 | DeepMind | 2020 | 潜在 actor-critic | ICLR |
| 4 | DreamerV2 | DeepMind | 2021 | 离散分类潜在表示 | ICLR |
| 5 | DreamerV3 | DeepMind | 2023 | 固定超参数 150+ 任务 | ICLR |
| 6 | IRIS | IDIAP | 2023 | VQ-VAE + Transformer | ICML |
| 7 | DIAMOND | — | 2024 | 扩散 world model RL | — |
| 8 | Genie / Genie 2 | Google DeepMind | 2024 | 11B 交互环境生成 | — |
| 9 | V-JEPA | Meta AI | 2024 | 联合嵌入预测 (非生成) | — |

---

> **参考综述论文**:
> 1. Yanchen Guan et al., "World Models for Autonomous Driving: An Initial Survey", IEEE T-IV, 2025
> 2. Teng Zhong et al., "A Survey on World Models for Autonomous Driving", 2024
> 3. Shu Yang et al., "Driving into the Future: A Comprehensive Survey on World Models in Autonomous Driving", 2024
> 4. "A Survey on Occupancy World Models for Autonomous Driving", 2024
> 5. "On the Promises and Challenges of Video Generation Models as World Simulators", 2024

---

*本综述覆盖 **80+ 篇核心论文**，涵盖 2018—2026 年学术界与工业界主要工作，横跨 8 大类别（视频生成、3D 占用、端到端 AD、仿真、基础模型、NeRF/3DGS、LLM/VLM 融合、经典基础）。自动驾驶 World Model 是当前 AI 研究最活跃的方向之一，视频生成 (Sora 范式)、3D 神经渲染 (3DGS)、基础模型和自动驾驶四大领域的交汇正在创造前所未有的研究机遇。核心争论——"生成逼真视频"与"理解世界因果结构"——将决定这一领域的长期演进方向。*
