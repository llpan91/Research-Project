---
title: "采样加速：从DDPM到一步生成的算法演进详解"
date: 2026-05-11
tags: ["diffusion-models", "理论基础"]
summary: "来自 扩散模型 研究笔记"
draft: false
---

# 采样加速：从 DDPM 到一步生成的算法演进详解

> **定位**：本文是对《扩散模型：从生成理论到决策智能》第 5 章的展开解读。针对原文中较为精炼的公式和结论，结合 **6 篇核心论文**，从直觉、数学推导、算法设计三个层面进行详细解读，帮助建立"为什么能加速"以及"每种方法到底改了什么"的清晰认知。
>
> **前置知识**：假设读者已理解 DDPM 的前向过程（闭式加噪）、逆向过程（逐步去噪）、以及噪声预测网络 $\varepsilon_\theta(x_t, t)$ 的基本训练流程（对应原文 §1-§4）。

---

## 目录

1. [问题的本质：为什么 DDPM 慢？](#1-问题的本质为什么-ddpm-慢)
2. [DDIM：从随机游走到确定性滑行](#2-ddim从随机游走到确定性滑行)
3. [DPM-Solver：用高阶 ODE 求解器进一步压缩步数](#3-dpm-solver用高阶-ode-求解器进一步压缩步数)
4. [Progressive Distillation：让学生学会"跳步"](#4-progressive-distillation让学生学会跳步)
5. [Consistency Models：一步到位的数学保证](#5-consistency-models一步到位的数学保证)
6. [Flow Matching：拉直传输路径](#6-flow-matching拉直传输路径)
7. [Latent Diffusion：降维打击](#7-latent-diffusion降维打击)
8. [全景对比与选择指南](#8-全景对比与选择指南)

---

## 1. 问题的本质：为什么 DDPM 慢？

### 1.1 DDPM 采样回顾

> 📄 **Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020**

DDPM 的推理（采样）过程是一条**马尔可夫链**：

$$x_T \sim \mathcal{N}(0, I) \;\xrightarrow{\text{step } T}\; x_{T-1} \;\xrightarrow{\text{step } T{-}1}\; \cdots \;\xrightarrow{\text{step } 1}\; x_0$$

每一步的更新公式为：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$

| 符号 | 含义 |
|------|------|
| $\alpha_t = 1 - \beta_t$ | 单步信号保留率 |
| $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | 累积信号保留率 |
| $\varepsilon_\theta(x_t, t)$ | 网络预测的噪声 |
| $\sigma_t$ | 注入的随机噪声标准差（DDPM 中取 $\sigma_t = \sqrt{\beta_t}$ 或 $\sqrt{\tilde{\beta}_t}$） |
| $z$ | 标准高斯随机噪声 |

### 1.2 慢在哪里？

**根本原因**：DDPM 的逆向过程是一条 **SDE（随机微分方程）的离散近似**，每步只走很小的"一步"，需要 $T=1000$ 步才能从纯噪声走回数据分布。

用一个物理比喻来理解：

```
DDPM 采样 ≈ 在浓雾中摸索回家

你站在一片浓雾里（纯高斯噪声 x_T），想回到家（真实数据 x_0）。

每一步你做两件事：
  1. 用"方向感"（ε_θ 网络）判断家的大致方向 → 朝那个方向迈一小步
  2. 雾让你随机偏移一点 → 加噪声 σ_t·z

因为每步只迈一小步 + 还被随机偏移，你需要走 1000 步才能到家。

核心瓶颈：
  - 每步都要调用一次神经网络（≈ 一次前向传播）
  - 1000 步 = 1000 次网络推理
  - 对于 512×512 图像，每次推理约 50ms → 总计 50 秒
```

**三个技术层面的瓶颈**：

| 瓶颈 | 原因 | 后果 |
|------|------|------|
| **步数多** | SDE 离散化需要小步长保证精度 | 1000 次网络调用 |
| **每步有随机性** | $\sigma_t z$ 项引入随机游走 | 不能跳步（跳了会偏） |
| **像素空间操作** | 直接在 $H \times W \times C$ 维空间扩散 | 每次网络前向传播计算量大 |

后续所有加速方法都在攻克这三个瓶颈中的一个或多个。

### 1.3 加速方法的分类框架

```
                        采样加速方法
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
    减少采样步数        降低单步开销        改变传输路径
          │                 │                  │
    ┌─────┼─────┐           │                  │
    │     │     │           │                  │
  无训练  蒸馏  重新        潜空间            最优传输
  加速    加速  定义问题    扩散              视角
    │     │     │           │                  │
  DDIM   PD   Consistency  Latent          Flow
  DPM-        Models       Diffusion       Matching
  Solver
```

---

## 2. DDIM：从随机游走到确定性滑行

> 📄 **Song et al., "Denoising Diffusion Implicit Models", ICLR 2021**

### 2.1 核心洞察：去掉随机性，就能跳步

DDIM 的出发点是一个极其深刻的观察：

> DDPM 的逆向过程中，**随机噪声项 $\sigma_t z$ 是导致不能跳步的根本原因**。如果我们去掉随机性，逆向过程就变成一条**确定性的轨迹**（ODE），沿着这条轨迹，我们可以选择任意稀疏的时间步子集来"跳着走"。

**直觉理解**：

```
DDPM（随机 SDE）：
  x_1000 →(+噪声)→ x_999 →(+噪声)→ x_998 → ... → x_0
  像醉汉回家：每步都有随机偏移，必须走小步才不会走丢

DDIM（确定性 ODE）：
  x_1000 ────────→ x_950 ────────→ x_900 → ... → x_0
  像清醒的人回家：每步走向确定的方向，可以大步跨越

  同一个网络 ε_θ，不需要重新训练！
```

### 2.2 数学推导：DDIM 是怎么得到的

**DDPM 的逆向过程源自贝叶斯后验** $q(x_{t-1} | x_t, x_0)$（见原文式 (5)-(7)）。但 DDPM 的作者只考虑了**马尔可夫**的逆向过程。

DDIM 的关键创新是：**定义一个更一般的、非马尔可夫的逆向过程**，它与 DDPM 共享相同的边缘分布 $q(x_t | x_0)$，但**逆向步骤之间的转移概率不同**。

具体地，DDIM 定义了一族参数化的逆向分布：

$$q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}},\; \sigma_t^2 I \right)$$

| 符号 | 含义 |
|------|------|
| $q_\sigma(x_{t-1} \mid x_t, x_0)$ | DDIM 定义的参数化逆向分布（由 $\sigma_t$ 控制） |
| $\sqrt{\bar{\alpha}_{t-1}} x_0$ | 均值中的"信号分量"：将 $x_0$ 估计缩放到 $t{-}1$ 的信号水平 |
| $\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}}$ | 从 $x_t$ 中提取的"噪声方向"（归一化后的噪声分量） |
| $\sigma_t$ | 控制随机性的自由参数：$\sigma_t = 0$ 时完全确定性 |

**这个分布的关键性质**：无论 $\sigma_t$ 取什么值，**前向边缘分布 $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$ 保持不变**。这意味着：
- 训练好的 $\varepsilon_\theta$ 可以直接复用，无需重新训练
- $\sigma_t$ 是一个**推理时的自由度**，可以在部署时调整

### 2.3 DDIM 采样公式的逐项拆解

将上式中的 $x_0$ 替换为网络估计 $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$，得到完整采样公式：

$$\boxed{x_{t-1} = \underbrace{\sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0}_{\text{(A) 预测信号}} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \varepsilon_\theta(x_t, t)}_{\text{(B) 预测噪声方向}} + \underbrace{\sigma_t \cdot \epsilon}_{\text{(C) 随机噪声}}}$$

**逐项理解**：

| 项 | 直觉含义 | 类比 |
|----|---------|------|
| **(A)** $\sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0$ | 网络对"原始图像"的最佳估计，缩放到 $t{-}1$ 的信号水平 | "我猜家在这里" |
| **(B)** $\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \varepsilon_\theta$ | 沿预测噪声方向保留一部分，控制从"当前状态"到"预测 $x_0$"的插值程度 | "从当前位置出发的惯性" |
| **(C)** $\sigma_t \cdot \epsilon$ | 额外随机性注入 | "随机探索" |

**$\sigma_t$ 的两个极端**：

| $\sigma_t$ 取值 | 效果 | 名称 |
|----------------|------|------|
| $\sigma_t = \sqrt{\frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)} \beta_t}$ | 退化为 DDPM | 完全随机 |
| $\sigma_t = 0$ | **完全确定性采样** | **DDIM** |

### 2.4 为什么 $\sigma_t = 0$ 就能跳步？

当 $\sigma_t = 0$ 时，采样公式变为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \varepsilon_\theta(x_t, t)$$

这是一个**确定性映射**：给定 $x_t$，$x_{t-1}$ 完全由网络输出决定，没有任何随机性。

确定性意味着**存在一条从 $x_T$ 到 $x_0$ 的唯一轨迹**。这条轨迹实际上是一个 **ODE（常微分方程）** 的数值解：

$$\frac{dx}{dt} = f(x, t) \quad \text{（概率流 ODE / Probability Flow ODE）}$$

ODE 的数值求解有一个重要性质：**你可以选择任意时间步网格来离散化**。不必逐步走 $1000, 999, 998, ..., 1, 0$，而是可以选择一个子序列，比如：

$$\tau = [1000, 950, 900, ..., 100, 50, 0] \quad \text{（仅 20 步）}$$

```
DDPM（1000 步）：  ●●●●●●●●●●●●●●●●●●●●●●●●●●...●● → x₀
                   逐步走，每步都有随机扰动

DDIM（50 步）：    ●──────●──────●──────●──────●──●  → x₀
                   跳着走，确定性轨迹，精度损失可控

DDIM（10 步）：    ●────────────────●────────────●    → x₀
                   大步跳，质量开始下降但仍可接受
```

**跳步的代价**：步长越大，ODE 数值求解的截断误差越大 → 生成质量下降。但实验表明 50 步的 DDIM 就能达到接近 DDPM 1000 步的质量。

### 2.5 DDIM 的额外优势：可逆编码

由于 DDIM 是确定性的，从 $x_0$ 到 $x_T$ 的映射也是确定性且可逆的。这意味着：

- **每张真实图像有一个唯一的"潜编码" $x_T$**
- 可以在潜空间做**语义插值**：对两张图像的 $x_T$ 做线性插值 → 解码 → 得到语义上平滑过渡的中间图像
- 这是 DDPM 做不到的（DDPM 的 $x_T$ 是随机的，没有"编码"功能）

```
图像 A → DDIM 编码 → x_T^A ─┐
                               ├── 插值 → x_T^{mix} → DDIM 解码 → 混合图像
图像 B → DDIM 编码 → x_T^B ─┘
```

### 2.6 DDIM 小结

| 维度 | DDPM | DDIM |
|------|------|------|
| 逆向过程类型 | 随机（SDE） | 确定性（ODE）或可控随机 |
| 典型步数 | 1000 | 20-50 |
| 加速比 | 1× | 20-50× |
| 是否需要重训 | — | **不需要** |
| 额外能力 | — | 可逆编码、潜空间插值 |
| 论文 | Ho et al., NeurIPS 2020 | Song et al., ICLR 2021 |

---

## 3. DPM-Solver：用高阶 ODE 求解器进一步压缩步数

> 📄 **Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps", NeurIPS 2022**
>
> 📄 **Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", 2022**

### 3.1 DDIM 的局限：一阶欧拉法

DDIM 本质上是用**一阶欧拉法（Euler method）** 求解概率流 ODE。回忆 ODE 数值求解的基本知识：

```
ODE：  dx/dt = f(x, t)

欧拉法（一阶）：  x_{t-Δt} = x_t + f(x_t, t) · (-Δt)     误差 O(Δt²)
中点法（二阶）：  先算中间点，再用中间点的斜率更新          误差 O(Δt³)
Runge-Kutta（四阶）：经典 RK4，四次函数求值               误差 O(Δt⁵)
```

DDIM 用欧拉法 → 每步截断误差 $O(\Delta t^2)$ → 要保持精度就不能步长太大 → 步数不能压得太低。

**DPM-Solver 的核心思想**：既然瓶颈在 ODE 求解器的阶数，那就设计一个**专门为扩散 ODE 定制的高阶求解器**。

### 3.2 扩散 ODE 的特殊结构

通用 ODE 求解器（如 RK45）把 $f(x, t)$ 当作黑箱。但扩散模型的概率流 ODE 有一个非常特殊的半线性结构：

$$\frac{dx}{dt} = \underbrace{f(t) \cdot x}_{\text{线性项（精确可解）}} + \underbrace{g(t) \cdot \varepsilon_\theta(x, t)}_{\text{非线性项（需要近似）}}$$

其中 $f(t)$ 和 $g(t)$ 是由噪声调度表决定的已知函数。

**关键洞察**：线性项可以精确求解（指数积分），只需要对非线性项做数值近似。这大大降低了求解难度。

### 3.3 DPM-Solver 的算法思想

DPM-Solver 使用**对数信噪比 $\lambda_t = \log(\bar{\alpha}_t / (1 - \bar{\alpha}_t))$** 作为时间变量（而非原始时间步 $t$），在这个变量下展开高阶 Taylor 展开：

**DPM-Solver-1（等价于 DDIM）**：
$$x_{t-1} \approx \text{精确线性项} + g(t) \cdot \varepsilon_\theta(x_t, t) \cdot \Delta\lambda$$

**DPM-Solver-2**：
- 先用一阶估计算一个中间点 $x_s$（$s$ 在 $t$ 和 $t{-}1$ 之间）
- 用 $\varepsilon_\theta(x_s, s)$ 修正斜率
- 用修正后的斜率更新到 $x_{t-1}$
- 每步需要 **2 次** 网络评估，但精度提升到二阶

**DPM-Solver-3**：
- 类似地，用两个中间点做三阶修正
- 每步 **3 次** 网络评估，三阶精度

```
DPM-Solver 各阶的步数 vs 质量权衡：

一阶（= DDIM）：    20 步 → FID ≈ 15    网络调用 20 次
二阶：              10 步 → FID ≈ 5     网络调用 20 次（每步 2 次）
三阶：              10 步 → FID ≈ 3.5   网络调用 30 次（每步 3 次）

注意：二阶 10 步和一阶 20 步的网络调用次数相同，但二阶质量远更好！
```

### 3.4 DPM-Solver++ 的改进

DPM-Solver++ 进一步改进了两个方面：

1. **数据预测（data prediction）** 而非噪声预测：在高引导强度（large guidance scale）下更稳定
2. **多步法（multistep method）**：复用前几步的网络输出，无需额外的中间点评估 → 每步仅 1 次网络调用即可达到高阶精度

```
DPM-Solver++（多步法）：

步骤 1:  ε₁ = ε_θ(x_t₁, t₁)         → 缓存
步骤 2:  ε₂ = ε_θ(x_t₂, t₂)         → 用 ε₁, ε₂ 做二阶更新
步骤 3:  ε₃ = ε_θ(x_t₃, t₃)         → 用 ε₁, ε₂, ε₃ 做三阶更新
  ...

每步只需 1 次网络调用（因为复用了历史值），
但通过多步信息获得高阶精度！
```

### 3.5 DPM-Solver 小结

| 维度 | DDIM | DPM-Solver-2 | DPM-Solver++ |
|------|------|-------------|--------------|
| ODE 求解阶数 | 1 阶 | 2 阶 | 2-3 阶 |
| 典型步数 | 50 | 10-20 | **10-20** |
| 每步网络调用 | 1 | 2 | **1**（多步法） |
| 总网络调用 | 50 | 20-40 | **10-20** |
| 是否需重训 | 否 | 否 | **否** |
| 论文 | Song, ICLR 2021 | Lu, NeurIPS 2022 | Lu, 2022 |

**重要意义**：DPM-Solver 系列证明了 **10-20 步的无训练加速** 是可行的，这已经让扩散模型在实际应用中可用（如 Stable Diffusion 默认使用 DPM-Solver++ 20 步）。

---

## 4. Progressive Distillation：让学生学会"跳步"

> 📄 **Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models", ICLR 2022**

### 4.1 核心思想

前面的方法（DDIM、DPM-Solver）都是**不修改模型**，只改采样算法。Progressive Distillation 走了另一条路：**训练一个新模型，让它用更少的步数达到相同质量**。

比喻：

```
DDIM/DPM-Solver ≈ 给同一辆车换更好的导航系统（更聪明的路径规划）
Progressive Distillation ≈ 训练一辆新车，让它一步能跨更远（更强的"腿"）
```

### 4.2 逐步减半的蒸馏策略

算法的核心循环极其优雅：

```
初始：教师模型使用 1024 步采样

第 1 轮蒸馏：
  教师：用 2 步（t → t-1 → t-2）生成 x_{t-2}
  学生：用 1 步（t → t-2）直接生成 x_{t-2}
  训练目标：让学生的 1 步输出 ≈ 教师的 2 步输出
  → 学生学会了 512 步采样

第 2 轮蒸馏：
  上一轮的学生变成新的教师
  重复同样的过程
  → 新学生学会了 256 步采样

...反复减半...

第 N 轮蒸馏：
  → 最终学生可以 4 步（甚至 1-2 步）采样
```

**数学表述**：

设教师模型为 $\varepsilon_\eta$，学生模型为 $\varepsilon_\theta$。对于时间步子序列 $t_1 > t_2 > t_3$（其中 $t_2$ 是 $t_1$ 和 $t_3$ 的中点）：

$$\text{教师的 2 步结果：}\; \tilde{x} = \text{DDIM}(\text{DDIM}(x_{t_1}, t_1 \to t_2, \varepsilon_\eta), t_2 \to t_3, \varepsilon_\eta)$$

$$\text{学生的 1 步结果：}\; \hat{x} = \text{DDIM}(x_{t_1}, t_1 \to t_3, \varepsilon_\theta)$$

$$\text{蒸馏损失：}\; L = \mathbb{E}\left[\| \tilde{x} - \hat{x} \|^2 \right]$$

或等价地，将损失表达为噪声预测空间中的 MSE：让学生预测的 $\varepsilon_\theta$ 等于教师两步 DDIM 隐含的"等效噪声"。

### 4.3 为什么逐步减半而不是直接蒸馏到 1 步？

```
直接蒸馏（1024 步 → 1 步）：
  教师用 1024 步慢慢走到终点
  学生要一步跨过去
  → 差距太大，学不好（任务过于困难）

逐步蒸馏（1024 → 512 → 256 → ... → 4）：
  每轮只要求学生学会"把两步合成一步"
  → 每轮的学习难度都很低
  → 累积起来实现大幅加速
```

这本质上是一种**课程学习（curriculum learning）**：先学简单的合并，再逐步学更大步的合并。

### 4.4 Progressive Distillation 小结

| 维度 | 详情 |
|------|------|
| 最终步数 | 2-4 步 |
| 是否需要重新训练 | **是**（需要多轮蒸馏，训练成本不低） |
| 优势 | 极少步数即可高质量生成 |
| 劣势 | 蒸馏过程本身耗时；每轮都需要完整训练 |
| 论文 | Salimans & Ho, ICLR 2022 |

---

## 5. Consistency Models：一步到位的数学保证

> 📄 **Song et al., "Consistency Models", ICML 2023**

### 5.1 核心思想：自一致性

Consistency Models（CM）提出了一个全新的角度来实现少步生成。它不是"蒸馏一个更快的去噪器"，而是**直接学习一个新函数 $f_\theta$，这个函数满足一个特殊的数学性质——自一致性**。

回顾概率流 ODE：从任意噪声水平 $t$ 出发，沿 ODE 轨迹前进，都会到达同一个终点 $x_0$。

```
ODE 轨迹：

         x_T ─────── x_{t₁} ─────── x_{t₂} ─────── x_ε ≈ x_0
           \            |              |              /
            \           |              |             /
             └──────────┴──────────────┴────────────┘
                        ↓              ↓
                    所有这些点经过 ODE 求解到终点，都到达同一个 x_0
```

**自一致性（Self-Consistency Property）**：

$$\boxed{f_\theta(x_t, t) = f_\theta(x_{t'}, t'), \quad \forall\, t, t' \in [\epsilon, T] \text{ 且 } x_t, x_{t'} \text{ 在同一条 ODE 轨迹上}}$$

即：一致性函数 $f_\theta$ 将同一条 ODE 轨迹上的**任意点**都映射到**同一个输出**——即该轨迹对应的 $x_0$。

**直觉**：如果你站在一条从纯噪声到数据的"河流"上的任意位置，一致性函数都能直接告诉你这条河最终流向哪里（$x_0$），而不需要你一步步沿河走下去。

### 5.2 边界条件

为了保证 $f_\theta$ 的输出确实是 $x_0$ 的估计，需要一个**边界条件**：

$$f_\theta(x_\epsilon, \epsilon) = x_\epsilon$$

即：在 $t = \epsilon$（接近 $t=0$，几乎没有噪声）时，输出就是输入本身。

实践中通过**参数化设计**来硬编码这个边界条件：

$$f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(x, t)$$

其中 $c_{\text{skip}}(\epsilon) = 1, c_{\text{out}}(\epsilon) = 0$，保证边界条件自动满足。$F_\theta$ 是一个可以是任意架构的神经网络（如 U-Net）。

### 5.3 两种训练方式

#### 方式一：Consistency Distillation（CD）

从一个**预训练的扩散模型**蒸馏。核心损失函数：

$$L_{\text{CD}} = \mathbb{E}\left[d\!\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat{x}_{t_n}, t_n)\right)\right]$$

| 符号 | 含义 |
|------|------|
| $f_\theta$ | 正在训练的一致性模型（在线网络） |
| $f_{\theta^-}$ | EMA 版本的一致性模型（目标网络，类似 DQN 的 target network） |
| $x_{t_{n+1}}$ | 含噪样本 |
| $\hat{x}_{t_n}$ | 用预训练扩散模型做一步 ODE 更新得到的结果 |
| $d(\cdot, \cdot)$ | 度量函数（如 L2、LPIPS 等） |

```
训练过程：

1. 采样 x₀ ~ 数据, ε ~ N(0,I), 计算 x_{t_{n+1}} = √ᾱ · x₀ + √(1-ᾱ) · ε
2. 用预训练扩散模型 ε_φ 做一步 ODE，得到 x̂_{t_n}
3. 损失 = d( f_θ(x_{t_{n+1}}, t_{n+1}),  f_{θ⁻}(x̂_{t_n}, t_n) )
              ↑ 在线网络                    ↑ 目标网络（EMA）
4. 这个损失惩罚的是"同一条轨迹上相邻两点的输出不一致"
   → 逼迫 f_θ 学会自一致性
```

**直觉**：如果相邻点的输出一致，通过传递性，整条轨迹上所有点的输出都一致 → $f_\theta(x_t, t)$ 就是 $x_0$ 的准确估计 → 一步生成！

#### 方式二：Consistency Training（CT）

不需要预训练扩散模型，从头训练。用 $x_0$ 本身代替 ODE 求解步骤：

$$L_{\text{CT}} = \mathbb{E}\left[d\!\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(x_{t_n}, t_n)\right)\right]$$

其中 $x_{t_{n+1}}$ 和 $x_{t_n}$ 都是从同一个 $x_0$ 出发用闭式公式加噪得到的（它们天然在同一条 ODE 轨迹上，因为共享同一个 $x_0$ 和 $\varepsilon$）。

CT 不需要预训练教师，但通常质量略低于 CD。

### 5.4 多步采样：质量提升

虽然 CM 支持一步生成，但也可以用 2-4 步进一步提升质量：

```
一步生成：
  x_T → f_θ(x_T, T) = x̂₀              直接输出

两步生成：
  x_T → f_θ(x_T, T) = x̂₀              第一步：粗估计
      → 重新加噪到 x_{t_mid}           在 x̂₀ 上加中间水平的噪声
      → f_θ(x_{t_mid}, t_mid) = x̂₀'    第二步：精修

四步生成：类似地，交替"估计 → 加噪 → 估计 → 加噪 → 估计"
```

### 5.5 与机器人实时控制的联系

这是 Consistency Models 对本项目（UMI）最重要的价值：

| Diffusion Policy | Consistency Policy |
|------------------|--------------------|
| 10-50 步去噪 | 1-2 步 |
| ~100ms 推理 | **< 10ms 推理** |
| 适合离线规划 | **适合实时闭环控制** |

> 📄 **Prasad et al., "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation", RSS 2024**

Consistency Policy 将 Diffusion Policy 蒸馏为 Consistency Model 形式，实现 1-2 步动作生成，使得机器人控制频率可达 100+ Hz。

### 5.6 Consistency Models 小结

| 维度 | Consistency Distillation | Consistency Training |
|------|------------------------|---------------------|
| 是否需要预训练扩散模型 | 是 | 否 |
| 训练复杂度 | 中等 | 较高 |
| 1 步生成质量 | 好 | 略低 |
| 4 步生成质量 | 接近扩散模型 | 接近 CD |
| 论文 | Song et al., ICML 2023 | 同上 |

---

## 6. Flow Matching：拉直传输路径

> 📄 **Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023**
>
> 📄 **Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023**

### 6.1 从一个直觉出发：扩散路径太弯了

回顾扩散模型的"路径"——从数据 $x_0$ 到噪声 $x_T$：

```
扩散模型的前向过程：
  x₀ → x₁ → x₂ → ... → x_T

这条路径是什么形状？

想象在高维空间中：
- x₀ 在数据流形上（比如一张猫的图片的像素向量）
- x_T 在标准高斯球上（纯噪声）
- 中间的路径由 q(x_t | x_0) = N(√ᾱ_t · x₀, (1-ᾱ_t)I) 决定

这条路径其实是弯曲的！
因为信号衰减 √ᾱ_t 和噪声增长 √(1-ᾱ_t) 的速率不是线性的（由 β_t 调度表决定）
```

**弯曲的路径 → 需要更多的离散化步数来准确追踪**。这正是扩散模型需要很多步的数学根源之一。

**Flow Matching 的核心思想**：如果我们**直接训练一条从噪声到数据的直线路径**，那么离散化步数可以大幅减少。

### 6.2 最优传输（Optimal Transport）视角

Flow Matching 的理论根基是**最优传输理论**：给定两个分布（噪声分布 $p_0 = \mathcal{N}(0, I)$ 和数据分布 $p_1 = p_{\text{data}}$），找到一个从 $p_0$ 到 $p_1$ 的"最短路径"（在 Wasserstein 距离意义下）。

在条件最优传输下，给定一对 $(x_0, x_1)$（噪声样本和数据样本），最优路径就是**直线**：

$$\psi_t(x_0) = (1 - t) x_0 + t \cdot x_1, \quad t \in [0, 1]$$

| 符号 | 含义 |
|------|------|
| $x_0 \sim \mathcal{N}(0, I)$ | 噪声起点（注意：这里的 $x_0$ 是噪声，$x_1$ 是数据，与 DDPM 的记号**相反**） |
| $x_1 \sim p_{\text{data}}$ | 数据终点 |
| $\psi_t(x_0)$ | $t$ 时刻的位置：从噪声 $x_0$ 到数据 $x_1$ 的线性插值 |
| $t \in [0, 1]$ | 归一化时间（0 = 纯噪声，1 = 纯数据） |

对应的**速度场（velocity field）**：

$$u_t(\psi_t) = x_1 - x_0$$

即沿着直线的恒定速度（方向和大小都不随 $t$ 变化）。

### 6.3 Flow Matching 的训练

训练一个神经网络 $v_\theta(x, t)$ 来预测速度场：

$$L_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(\psi_t, t) - (x_1 - x_0)\|^2\right]$$

| 符号 | 含义 |
|------|------|
| $v_\theta(x, t)$ | 网络预测的速度场 |
| $\psi_t = (1-t) x_0 + t x_1$ | $t$ 时刻的插值点 |
| $x_1 - x_0$ | 真实速度（常量向量，方向从噪声指向数据） |

**与扩散模型训练的对比**：

| | 扩散模型 | Flow Matching |
|--|---------|--------------|
| 训练目标 | 预测噪声 $\varepsilon$ | 预测速度 $v = x_1 - x_0$ |
| 插值公式 | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \varepsilon$ | $\psi_t = (1-t) x_0 + t x_1$ |
| 路径形状 | 弯曲（由 $\bar{\alpha}_t$ 控制） | **直线** |
| 时间范围 | $t \in [0, T]$（离散步） | $t \in [0, 1]$（连续） |

### 6.4 为什么直线路径更好？

```
弯曲路径（扩散模型）：                    直线路径（Flow Matching）：

  噪声 ●                                  噪声 ●
        \                                        \
         \                                        \
          ╲                                        \
           ╲                                        \
            ╲  弯曲！                                \  直线！
             ╲                                        \
              ╲                                        \
               ╲                                        \
                ●  数据                                  ● 数据

用 5 步离散化这条弯曲路径：              用 5 步离散化这条直线：
  误差较大（弦代替弧）                     误差很小（直线的离散化就是直线本身）
```

**数学上**：

- 弯曲路径的曲率 → ODE 求解器需要更小的步长
- 直线路径的曲率为零 → **理论上一步就能精确求解**（如果 $v_\theta$ 完美学会了速度场）

实际中由于 $v_\theta$ 不完美（训练误差）、以及边缘速度场（对所有 $x_0$ 求平均后）不再是完美直线，仍需要一些步数，但比扩散模型少得多。

### 6.5 采样过程

推理时只需求解 ODE：

$$x_0 \sim \mathcal{N}(0, I), \quad \frac{dx}{dt} = v_\theta(x, t), \quad x_1 = x_0 + \int_0^1 v_\theta(x_t, t) \, dt$$

用任意 ODE 求解器（欧拉法、中点法等）离散化：

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

典型只需 **5-20 步**即可达到高质量。

### 6.6 Rectified Flow：用 Reflow 进一步拉直

> 📄 **Liu et al., "Flow Straight and Fast", ICLR 2023**

Rectified Flow 提出了**Reflow**操作：

```
第 1 轮训练：
  用 (x₀, x₁) 对训练 → 得到 v_θ₁
  v_θ₁ 定义的路径已经大致是直线

Reflow（第 2 轮）：
  从 x₀ ~ N(0,I) 出发，用 v_θ₁ 求解 ODE 得到 x₁'
  现在 (x₀, x₁') 是一对"ODE 耦合"的点
  用新的 (x₀, x₁') 对重新训练 → 得到 v_θ₂
  v_θ₂ 定义的路径更直！

每次 Reflow 都让路径更直 → 更少步数即可
```

### 6.7 Flow Matching 与扩散模型的统一

一个重要的理论结果：**扩散模型的概率流 ODE 是 Flow Matching 的特例**，只是路径选择不同。

| 框架 | 路径 $\psi_t$ | 速度场 |
|------|-------------|--------|
| VP-SDE（DDPM 系列） | $\sqrt{\bar{\alpha}_t} x_1 + \sqrt{1-\bar{\alpha}_t} x_0$ | 弯曲 |
| VE-SDE（NCSN 系列） | $x_1 + \sigma_t x_0$ | 弯曲 |
| **Optimal Transport FM** | $(1-t) x_0 + t x_1$ | **直线** |

Stable Diffusion 3、Flux 等最新模型已采用 Flow Matching 框架，验证了其在大规模生成任务上的优势。

### 6.8 Flow Matching 小结

| 维度 | 详情 |
|------|------|
| 核心创新 | 直线传输路径 + 速度场预测 |
| 典型步数 | 5-20 步 |
| 是否需重训 | 是（新框架，需从头训练） |
| 数学复杂度 | **比扩散模型更简洁** |
| 代表应用 | Stable Diffusion 3, Flux |
| 论文 | Lipman et al., ICLR 2023; Liu et al., ICLR 2023 |

---

## 7. Latent Diffusion：降维打击

> 📄 **Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022**

### 7.1 攻克另一个瓶颈：单步计算量

前面的方法（DDIM、DPM-Solver、CM、FM）都在减少**步数**。但还有一个正交的加速维度：**减少每一步的计算量**。

像素空间的问题：

```
512×512 RGB 图像 = 512 × 512 × 3 = 786,432 维向量
→ U-Net 要在 786K 维空间做卷积
→ 每次前向传播约 50ms（A100 GPU）

如果能在更低维的空间做扩散...
```

### 7.2 核心架构：感知压缩 + 潜空间扩散

```
┌─────────────────────────────────────────────────────────┐
│ 阶段 1：感知压缩（预训练，一次性）                         │
│                                                          │
│   图像 x ∈ R^{512×512×3}                                 │
│       ↓ VAE Encoder E                                    │
│   潜表示 z = E(x) ∈ R^{64×64×4}        ← 压缩 48×       │
│       ↓ VAE Decoder D                                    │
│   重建 x̂ = D(z) ≈ x                                     │
│                                                          │
│   训练：重建损失 + KL 正则 + 感知损失（LPIPS）             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 阶段 2：潜空间扩散（核心训练）                             │
│                                                          │
│   在 z 空间而非 x 空间做扩散：                             │
│                                                          │
│   前向加噪：z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε            │
│   训练：     min_θ ||ε - ε_θ(z_t, t, c)||²              │
│   采样：     z_T ~ N(0,I) → 逐步去噪 → z₀ → D(z₀) = x₀ │
│                                                          │
│   条件注入：通过 Cross-Attention 将文本/图像条件注入 U-Net  │
└─────────────────────────────────────────────────────────┘
```

### 7.3 加速效果

| 维度 | 像素空间扩散 | 潜空间扩散（LDM） |
|------|-----------|-----------------|
| 扩散空间维度 | $512 \times 512 \times 3 = 786K$ | $64 \times 64 \times 4 = 16K$ |
| 单步计算量 | ~50ms | **~3ms**（约 16× 加速） |
| 50 步总耗时 | ~2.5s | **~0.15s** |
| 图像质量 | 高 | 高（VAE 重建几乎无损） |
| 可扩展性 | 难以高分辨率 | **可扩展到 1024×1024+** |

**关键洞察**：图像的高频细节（像素级精度）对语义没有贡献，VAE 的编码器去除了这些**感知冗余**。在去除冗余后的低维空间做扩散，既保留了语义信息，又大幅降低了计算量。

### 7.4 与其他加速方法的组合

Latent Diffusion 与前述方法**完全正交**，可以叠加使用：

```
组合加速（Stable Diffusion 的实际配置）：

  Latent Diffusion（降维 48×）
  + DPM-Solver++（20 步 ODE）
  + FP16 精度
  + Flash Attention
  ─────────────────────────
  = 512×512 图像在消费级 GPU 上 2-5 秒生成
```

---

## 8. 全景对比与选择指南

### 8.1 所有方法的统一对比

| 方法 | 论文/年份 | 核心思想 | 步数 | 需重训？ | 加速类型 |
|------|----------|---------|------|---------|---------|
| **DDPM** | Ho, NeurIPS 2020 | 基线 | 1000 | — | — |
| **DDIM** | Song, ICLR 2021 | SDE→ODE，去随机性 | 20-50 | 否 | 减步数 |
| **DPM-Solver** | Lu, NeurIPS 2022 | 高阶 ODE 求解器 | 10-20 | 否 | 减步数 |
| **Prog. Distill.** | Salimans, ICLR 2022 | 逐步减半蒸馏 | 2-4 | 是 | 减步数 |
| **Consistency** | Song, ICML 2023 | 自一致性映射 | 1-4 | 是 | 减步数 |
| **Flow Matching** | Lipman, ICLR 2023 | 直线传输路径 | 5-20 | 是（新框架） | 改路径 |
| **Latent Diff.** | Rombach, CVPR 2022 | 潜空间扩散 | 不变 | 是（两阶段） | 降维度 |

### 8.2 发展的内在逻辑

```
一切的起点：DDPM 需要 1000 步太慢了

思路 1："能不能不改模型，只改采样算法？"
  → DDIM：去掉随机性，确定性 ODE 可以跳步（50 步）
  → DPM-Solver：更好的 ODE 求解器（10-20 步）
  → 极限：无训练方法的极限大约在 10 步左右

思路 2："能不能训练一个更快的模型？"
  → Progressive Distillation：教学生用更少步数模仿教师（2-4 步）
  → Consistency Models：直接学习一步到位的映射（1-4 步）

思路 3："能不能从根本上改变问题的定义？"
  → Flow Matching：用直线路径代替弯曲路径，本质上减少所需步数（5-20 步）
  → Latent Diffusion：在低维空间做扩散，降低每步计算量（单步 16× 快）

实际应用中三种思路通常组合使用：
  Stable Diffusion = Latent Diffusion + DPM-Solver++
  Stable Diffusion 3 = Latent Diffusion + Flow Matching
  Consistency Policy = Diffusion Policy + Consistency Distillation
```

### 8.3 场景选择指南

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| **已有预训练扩散模型，想直接加速** | DDIM / DPM-Solver++ | 零训练成本，即插即用 |
| **对质量要求极高，步数不太敏感** | DPM-Solver++ 20 步 | 最佳质量-速度平衡 |
| **需要 1-4 步极速生成** | Consistency Distillation | 1-4 步仍保持高质量 |
| **从头训练新模型** | Flow Matching | 更简洁的数学，更好的收敛 |
| **高分辨率图像生成** | Latent Diffusion + 上述任一 | 降维是高分辨率的必要条件 |
| **机器人实时控制（< 10ms）** | Consistency Policy | 1-2 步推理满足实时性 |

### 8.4 历史演进时间线

```
2020.06  DDPM            ────  扩散模型可以生成高质量图像（但需要 1000 步）
2020.10  DDIM            ────  去掉随机性 → 50 步
2021.01  Score SDE       ────  统一框架：SDE/ODE/分数匹配
2022.02  Prog. Distill.  ────  蒸馏 → 4 步
2022.04  DPM-Solver      ────  高阶求解器 → 10-20 步
2022.06  Latent Diff.    ────  潜空间扩散，单步 16× 快
2022.12  DPM-Solver++    ────  无训练加速极限 → 10 步
2023.03  Consistency     ────  一步生成成为可能
2023.06  Flow Matching   ────  直线路径，更少步数
2023.12  SD3 / Flux      ────  Flow Matching 大规模验证
2024.xx  Consistency Policy ── 机器人实时控制 < 10ms
```

---

## 附录：关键论文列表

| 论文 | 会议/年份 | 核心贡献 |
|------|----------|---------|
| DDPM (Ho et al.) | NeurIPS 2020 | 扩散模型基础，1000 步采样 |
| DDIM (Song et al.) | ICLR 2021 | 确定性 ODE 采样，可跳步 |
| Score SDE (Song et al.) | ICLR 2021 | SDE/ODE 统一框架 |
| DPM-Solver (Lu et al.) | NeurIPS 2022 | 专用高阶 ODE 求解器，10 步高质量 |
| DPM-Solver++ (Lu et al.) | arXiv 2022 | 多步法，每步仅 1 次网络调用 |
| Progressive Distillation (Salimans & Ho) | ICLR 2022 | 逐步减半蒸馏，4 步生成 |
| Latent Diffusion / Stable Diffusion (Rombach et al.) | CVPR 2022 | 潜空间扩散，单步计算量降 16× |
| Consistency Models (Song et al.) | ICML 2023 | 自一致性映射，1 步生成 |
| Flow Matching (Lipman et al.) | ICLR 2023 | 直线传输路径 |
| Rectified Flow (Liu et al.) | ICLR 2023 | Reflow 拉直路径 |
| Consistency Policy (Prasad et al.) | RSS 2024 | 机器人 1-2 步动作生成 |
