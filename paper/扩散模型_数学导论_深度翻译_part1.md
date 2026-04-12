# 扩散模型：数学导论

**原文：Diffusion Models: A Mathematical Introduction** \
**作者：Sepehr Maleki (林肯大学 Lincoln AI Lab) 和 Negar Pourmoazemi (Trainline, 伦敦)** \
**arXiv:2511.11746v1, 2025年11月**

---

## 摘要

本文对基于扩散的生成模型进行了简洁、自洽的推导。从高斯分布的基本性质（密度函数、二次型期望、重参数化、乘积以及 KL 散度）出发，我们从第一性原理构建了去噪扩散概率模型。这包括前向加噪过程、其封闭形式的边际分布、精确的离散逆后验分布，以及相关的变分下界。该下界最终简化为实践中常用的标准噪声预测目标。随后我们讨论了似然估计与加速采样，涵盖 DDIM、对抗学习的逆动力学（DDGAN）以及多尺度变体（如嵌套扩散和潜在扩散），并以 Stable Diffusion 作为典型示例。接着给出连续时间公式：从扩散 SDE 出发，经由连续性方程和 Fokker-Planck 方程导出概率流 ODE，引入流匹配方法，并展示整流（rectified）流如何在时间重参数化的意义下恢复 DDIM。最后讨论引导扩散：将分类器引导解释为后验得分的修正，将无分类器引导解释为条件和无条件得分之间的有原则插值。全文聚焦于透明的代数推导、明确的中间步骤以及一致的符号体系，使读者既能理解理论，也能实现相应的算法。

---

## 目录

1. [引言](#1-引言)
2. [预备知识](#2-预备知识)
   - 2.1 [各向同性高斯的密度函数](#21-各向同性高斯的密度函数)
   - 2.2 [二次型的期望](#22-二次型的期望)
   - 2.3 [高斯采样的重参数化](#23-高斯采样的重参数化)
   - 2.4 [两个高斯分布的乘积](#24-两个高斯分布的乘积)
   - 2.5 [高斯分布之间的 KL 散度](#25-高斯分布之间的-kl-散度)
3. [扩散模型](#3-扩散模型)
   - 3.1 [前向过程](#31-前向过程)
   - 3.2 [逆过程与真实后验（DDPM 后验）](#32-逆过程与真实后验ddpm-后验)
   - 3.3 [损失函数](#33-损失函数)
   - 3.4 [计算 p_θ(x_0)](#34-计算-p_θx_0)
   - 3.5 [补充证明：变分下界推导全解析](#35-补充证明变分下界推导全解析)
4. [加速方法](#4-加速方法)
   - 4.1 [去噪扩散隐式模型（DDIM）](#41-去噪扩散隐式模型ddim)
   - 4.2 [DDGAN：对抗学习的逆动力学](#42-ddgan对抗学习的逆动力学)
   - 4.3 [嵌套扩散模型](#43-嵌套扩散模型)
   - 4.4 [Stable Diffusion](#44-stable-diffusion)
5. [流匹配](#5-流匹配)
   - 5.1 [概率流 ODE 与连续性方程](#51-概率流-ode-与连续性方程)
   - 5.2 [流匹配目标：边际与条件](#52-流匹配目标边际与条件)
   - 5.3 [线性流与整流流（直线耦合）](#53-线性流与整流流直线耦合)
   - 5.4 [DDIM 作为流匹配（时间重参数化）](#54-ddim-作为流匹配时间重参数化)
   - 5.5 [离散化与实践训练注意事项](#55-离散化与实践训练注意事项)
6. [引导扩散](#6-引导扩散)
   - 6.1 [基于分类器的引导：后验得分修正](#61-基于分类器的引导后验得分修正)
   - 6.2 [无分类器引导：条件-无条件混合](#62-无分类器引导条件-无条件混合)
   - 6.3 [时变引导调度](#63-时变引导调度)
   - 6.4 [稳定性：失效模式与补救措施](#64-稳定性失效模式与补救措施)
   - 6.5 [引导蒸馏](#65-引导蒸馏)
7. [结论](#7-结论)

---

## 1 引言

去噪扩散模型已成为深度生成建模的核心范式。它们构建一个易于处理的**前向**（扩散）过程，通过高斯噪声逐步破坏数据，然后学习一个**逆向**（生成）过程，逐步去噪恢复数据。这一简单的思路——噪声输入、数据输出——具有显著的可扩展性，在图像合成、文本到图像生成、音频等领域取得了最先进的成果。

然而，其背后的数学初看之下可能显得晦涩难懂。人们需要跟踪耦合的马尔可夫链及其高斯条件分布，推导变分（ELBO）目标函数，选择和理解噪声调度，并理解条件控制机制（基于分类器的引导和无分类器的引导）如何改变逆向动力学。连续时间的视角进一步将离散模型与随机微分方程（SDE）及其关联的概率流常微分方程（ODE）联系起来，而近期的工作又将采样重新表述为通过流匹配学习一个速度场。

本教程从**第一性原理**出发，以统一的符号和完整的中间步骤来发展这些核心要素。

我们首先介绍高斯分布的预备知识（密度函数、仿射变换、乘积、KL 散度），以确定全文使用的恒等式。然后仔细构建去噪扩散概率模型（DDPM）：前向链和逆向链、精确的 DDPM 后验分布和变分损失，展示标准训练目标如何在 $\epsilon$-预测参数化下自然出现。在此过程中，我们明确了 $\epsilon$-参数化、$\hat{\mathbf{x}}_0$-参数化和速度参数化之间的关系，阐明了逆向均值系数 $\beta_t/\sqrt{1-\bar{\alpha}_t}$ 的含义，并给出了后验方差 $\tilde{\beta}_t$ 的角色。接着，我们将离散扩散与基于得分（score）和概率流的视角联系起来，解释这些视角如何激发 DDIM 等加速采样器。然后我们介绍实际的加速方法和潜在空间方法（如 Stable Diffusion），并详细讨论引导扩散——推导分类器引导作为后验得分的修正，以及无分类器引导作为条件与无条件得分之间有原则的混合，同时涉及时变引导调度、数值稳定性和常见失效模式。最后，我们介绍流匹配，包括条件目标和边际目标以及直线（整流）流，以简洁的证明保持内容的可读性和严谨性。

本教程面向希望获得扩散生成器工作原理的数学理解的研究人员和学生。需要概率论的基础知识、高斯分布的基本线性代数知识；深度学习基础有帮助但非严格必需。文本是自包含的，证明尽可能保持基础。我们的目标是用透明的代数和直接可用的公式来取代经验性的启发式方法。

### 读者路线图

如果你是扩散模型的初学者，请从高斯预备知识和 DDPM 推导开始，它们建立了符号体系和变分目标。主要关注快速采样的读者可以直接跳到加速材料（DDIM 及相关方法）和 Stable Diffusion 的潜在空间部分。关注可控性的读者应查看引导扩散部分（分类器引导和无分类器引导、调度和注意事项）。流匹配部分统一了 ODE 视角并提供了一条替代的训练路线。

### 符号

所有随机向量默认在 $\mathbb{R}^d$ 中，除非另行说明。粗体符号表示向量/矩阵（如 $\mathbf{x}, \mathbf{z}, \boldsymbol{\Sigma}$）；单位矩阵为 $\mathbf{I}_d$（维度明确时简记为 $\mathbf{I}$）。欧几里得范数为 $\|\cdot\|_2$，内积为 $\langle\cdot,\cdot\rangle$，迹为 $\text{tr}(\cdot)$，行列式为 $\det(\cdot)$。均值为 $\boldsymbol{\mu}$、协方差为 $\boldsymbol{\Sigma}$ 的高斯分布记为 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$。期望为 $\mathbb{E}[\cdot]$，方差为 $\text{Var}[\cdot]$，$\text{KL}(P\|Q)$ 表示 Kullback-Leibler 散度。$\text{law}(\mathbf{X})$ 表示随机向量 $\mathbf{X}$ 的分布。

数据为 $\mathbf{x}_0 \sim p_\text{data}$；加噪状态为 $\mathbf{x}_t$，对应离散时间 $t \in \{1,\ldots,T\}$。模型分布为 $p_\theta$，前向（加噪）分布为 $q$。前向链为：

$$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1}), \quad q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\big(\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I}\big) \tag{Eq.N-1}$$

其中调度参数 $\alpha_t \in (0,1)$，$\beta_t := 1 - \alpha_t$。我们使用累积乘积：

$$\bar{\alpha}_t := \prod_{i=1}^{t} \alpha_i, \quad \text{SNR}_t := \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}, \quad \ell_t := \log \text{SNR}_t \tag{Eq.N-2}$$

封闭形式的边际分布和重参数化为：

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\big(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I}\big), \quad \mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \;\; \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{Eq.N-3}$$

逆向（去噪）条件分布为：

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\big(\mu_\theta(\mathbf{x}_t, t),\, \sigma_t^2 \mathbf{I}\big), \quad \mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\bigg(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)\bigg) \tag{Eq.N-4}$$

其中 $\hat{\boldsymbol{\epsilon}}_\theta$ 是学习到的噪声预测器，后验方差为：

$$\tilde{\beta}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\,\beta_t, \quad \sigma_t^2 \in \{\tilde{\beta}_t,\, 0\} \;\;\text{分别对应 DDPM/DDIM} \tag{Eq.N-5}$$

$\hat{\mathbf{x}}_0(\mathbf{x}_t, t)$ 表示 $x_0$-预测，$s_t(\mathbf{x}) := \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 表示（真实或学习的）得分。

对于连续时间符号，$\mathbf{X}_t$ 表示 $t \in [0,T]$ 时的状态，布朗运动为 $\mathbf{W}_t$，扩散 SDE 为：

$$\mathrm{d}\mathbf{X}_t = \mathbf{f}(\mathbf{X}_t, t)\,\mathrm{d}t + g(t)\,\mathrm{d}\mathbf{W}_t \tag{Eq.N-6}$$

概率流 ODE 速度为 $\mathbf{v}(\mathbf{x}, t)$，产生相同的时间边际分布。在流匹配部分，$\mathbf{v}_\theta(\mathbf{x}, t)$ 为学习的速度，$\psi_t(\cdot)$ 为端点间的插值。条件变量为 $\mathbf{c}$；无分类器引导使用缩放因子 $\lambda \geq 0$ 和混合预测器：

$$\hat{\boldsymbol{\epsilon}}_\lambda = \hat{\boldsymbol{\epsilon}}_\text{uncond} + \lambda\big(\hat{\boldsymbol{\epsilon}}_\text{cond} - \hat{\boldsymbol{\epsilon}}_\text{uncond}\big) \tag{Eq.N-7}$$

对应的得分形式为 $s_t^{(\lambda)}(\mathbf{x}) = -(1/\sqrt{1-\bar{\alpha}_t})\,\hat{\boldsymbol{\epsilon}}_\lambda(\mathbf{x}, t, \mathbf{c})$。在讨论潜在空间方法时，潜变量记为 $\mathbf{z}_t$，被当作高斯密度内部的向量处理（隐式展平）。

---

## 2 预备知识

### 2.1 各向同性高斯的密度函数

设 $\mathbf{x} \in \mathbb{R}^d$，考虑 $d$ 维高斯分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$。其概率密度函数（PDF）为：

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}\det(\boldsymbol{\Sigma})^{1/2}} \exp\Big(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\Big) \tag{1}$$

其中 $\boldsymbol{\mu} \in \mathbb{R}^d$，$\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ 是对称正定矩阵。

**各向同性**高斯是协方差 $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$（$\sigma > 0$）的特殊情况，即方差在每个方向上相同（等价地，分布是旋转不变的）。此时：

$$\det(\boldsymbol{\Sigma}) = \sigma^{2d}, \quad \boldsymbol{\Sigma}^{-1} = \frac{1}{\sigma^2}\mathbf{I}$$

二次型（马氏距离）项化简为：

$$(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) = \frac{1}{\sigma^2}(\mathbf{x}-\boldsymbol{\mu})^\top(\mathbf{x}-\boldsymbol{\mu}) = \frac{1}{\sigma^2}\|\mathbf{x}-\boldsymbol{\mu}\|_2^2$$

因此：

$$p(\mathbf{x}) = \frac{1}{(2\pi\sigma^2)^{d/2}}\exp\Big(-\frac{1}{2\sigma^2}\|\mathbf{x}-\boldsymbol{\mu}\|_2^2\Big) \tag{2}$$

### 2.2 二次型的期望

在推导高斯 KL 散度等结果时，我们经常遇到高斯分布下形如 $(\mathbf{x}-\boldsymbol{\mu}_p)^\top \mathbf{A}(\mathbf{x}-\boldsymbol{\mu}_p)$ 的二次型表达式。本节给出相关恒等式及简短证明。

**定义 2.1**（迹）。对矩阵 $\mathbf{A} \in \mathbb{R}^{d \times d}$，迹是其对角元素之和：

$$\text{tr}(\mathbf{A}) = \sum_{i=1}^{d} A_{ii}$$

**引理 2.2**（迹的恒等式）。设 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{d \times d}$，$\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$，则：

1. **线性性**：对所有标量 $\alpha, \beta \in \mathbb{R}$，$\text{tr}(\alpha\mathbf{A}+\beta\mathbf{B}) = \alpha\,\text{tr}(\mathbf{A}) + \beta\,\text{tr}(\mathbf{B})$
2. **轮换性**：只要乘积有定义，$\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})$
3. **秩一情形**：$\text{tr}(\mathbf{u}\mathbf{v}^\top) = \mathbf{v}^\top \mathbf{u}$

**引理 2.3**（迹技巧）。对任意 $\mathbf{A} \in \mathbb{R}^{d \times d}$ 和 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$：

$$\mathbf{v}^\top \mathbf{A}\,\mathbf{u} = \text{tr}(\mathbf{A}\,\mathbf{u}\mathbf{v}^\top) \tag{Eq.2-1}$$

> **证明**：从右侧出发。由引理 2.2 的 (ii) 和 (iii)：$\text{tr}(\mathbf{A}\mathbf{u}\mathbf{v}^\top) = \text{tr}(\mathbf{u}\mathbf{v}^\top\mathbf{A}) = \text{tr}(\mathbf{u}(\mathbf{A}^\top\mathbf{v})^\top) = \mathbf{v}^\top\mathbf{A}\,\mathbf{u}$

**命题 2.4**（中心二次型的期望）。设 $p$ 为 $\mathbb{R}^d$ 上的高斯 $\mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$，$\mathbf{x} \sim p$。对任意 $\mathbf{A} \in \mathbb{R}^{d \times d}$：

$$\mathbb{E}_p\big[(\mathbf{x}-\boldsymbol{\mu}_p)^\top \mathbf{A}\,(\mathbf{x}-\boldsymbol{\mu}_p)\big] = \text{tr}(\mathbf{A}\,\boldsymbol{\Sigma}_p) \tag{3}$$

> **证明**：令 $\mathbf{z} := \mathbf{x} - \boldsymbol{\mu}_p \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_p)$。由引理 2.3 以及迹和期望的线性性：
> $$\mathbb{E}_p[\mathbf{z}^\top\mathbf{A}\mathbf{z}] = \mathbb{E}_p[\text{tr}(\mathbf{A}\,\mathbf{z}\mathbf{z}^\top)] = \text{tr}(\mathbf{A}\,\mathbb{E}_p[\mathbf{z}\mathbf{z}^\top]) = \text{tr}(\mathbf{A}\,\boldsymbol{\Sigma}_p)$$


### 2.3 高斯采样的重参数化

我们希望找到一种简单、可微分的方法来从多元高斯 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \subset \mathbb{R}^d$ 中采样。关键思想是将样本表示为标准正态的仿射函数。

**定义 2.5**（仿射重参数化）。设 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$。对 $\boldsymbol{\mu} \in \mathbb{R}^d$ 和 $\mathbf{A} \in \mathbb{R}^{d \times d}$，定义：

$$\mathbf{x} := \boldsymbol{\mu} + \mathbf{A}\mathbf{z}$$

**命题 2.6**（标准高斯仿射变换的分布）。若 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，则：

$$\mathbf{x} = \boldsymbol{\mu} + \mathbf{A}\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{A}\mathbf{A}^\top) \tag{Eq.2-2}$$

> **证明**：写 $\mathbf{x} = \boldsymbol{\mu} + \mathbf{A}\mathbf{z}$，其中 $\mathbb{E}[\mathbf{z}] = \mathbf{0}$，$\text{Cov}[\mathbf{z}] = \mathbf{I}_d$。均值显然为：
> $$\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu} + \mathbf{A}\,\mathbb{E}[\mathbf{z}] = \boldsymbol{\mu}$$
> 
> 对于协方差，利用定义 $\text{Cov}[\mathbf{x}] = \mathbb{E}[(\mathbf{x}-\mathbb{E}[\mathbf{x}])(\mathbf{x}-\mathbb{E}[\mathbf{x}])^\top]$：
> $$\text{Cov}[\mathbf{x}] = \mathbb{E}[(\mathbf{A}\mathbf{z})(\mathbf{A}\mathbf{z})^\top] = \mathbb{E}[\mathbf{A}\,\mathbf{z}\mathbf{z}^\top\mathbf{A}^\top] = \mathbf{A}\,\mathbb{E}[\mathbf{z}\mathbf{z}^\top]\,\mathbf{A}^\top = \mathbf{A}\,\mathbf{I}_d\,\mathbf{A}^\top = \mathbf{A}\mathbf{A}^\top$$
> 
> 其中 $\mathbb{E}[\mathbf{z}\mathbf{z}^\top] = \mathbf{I}_d$ 可以显式验证：写 $\mathbf{z} = (z_1,\ldots,z_d)^\top$，各 $z_i \sim \mathcal{N}(0,1)$ 且不相关，则 $[\mathbb{E}[\mathbf{z}\mathbf{z}^\top]]_{ij} = \mathbb{E}[z_i z_j]$，当 $i \neq j$ 时为 0，当 $i = j$ 时为 1。

**推论 2.7**（采样 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$）。若 $\boldsymbol{\Sigma}$ 对称半正定且 $\mathbf{A}$ 满足 $\mathbf{A}\mathbf{A}^\top = \boldsymbol{\Sigma}$（如当 $\boldsymbol{\Sigma} \succ \mathbf{0}$ 时取 Cholesky 分解），则对 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$：

$$\mathbf{x} = \boldsymbol{\mu} + \mathbf{A}\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \tag{Eq.2-3}$$

**推论 2.8**（各向同性情形）。对 $\sigma > 0$，$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$：

$$\mathbf{x} = \boldsymbol{\mu} + \sigma\,\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \sigma^2\mathbf{I}_d) \tag{Eq.2-4}$$

**命题 2.9**（期望的重参数化恒等式）。设 $f: \mathbb{R}^d \to \mathbb{R}$ 在 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 下可积。若 $\boldsymbol{\Sigma} = \mathbf{A}\mathbf{A}^\top$ 且 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，则：

$$\mathbb{E}_{\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})}[f(\mathbf{x})] = \mathbb{E}_{\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_d)}[f(\boldsymbol{\mu}+\mathbf{A}\mathbf{z})] \tag{Eq.2-5}$$

### 2.4 两个高斯分布的乘积

我们经常需要将同一变量上的两个高斯因子组合在一起（例如，合并来自两个来源的信息）。关键事实是：乘积仍然（正比于）一个高斯。为了便于读出结果的均值和协方差，我们先记录一个标准的"配方"恒等式。

**引理 2.10**（配方）。设 $\boldsymbol{\Lambda} \in \mathbb{R}^{d \times d}$ 对称正定，$\boldsymbol{\eta} \in \mathbb{R}^d$。则对所有 $\mathbf{x} \in \mathbb{R}^d$：

$$-\tfrac{1}{2}\mathbf{x}^\top\boldsymbol{\Lambda}\mathbf{x} + \mathbf{x}^\top\boldsymbol{\eta} = -\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Lambda}(\mathbf{x}-\boldsymbol{\mu}) + \tfrac{1}{2}\boldsymbol{\mu}^\top\boldsymbol{\Lambda}\boldsymbol{\mu}, \quad \text{其中}\;\boldsymbol{\mu} = \boldsymbol{\Lambda}^{-1}\boldsymbol{\eta} \tag{Eq.2-6}$$

**命题 2.11**（两个高斯的乘积）。设 $p(\mathbf{x}) = \mathcal{N}(\mathbf{x};\boldsymbol{\mu}_p,\boldsymbol{\Sigma}_p)$，$q(\mathbf{x}) = \mathcal{N}(\mathbf{x};\boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q)$，均具有对称正定协方差。则它们的逐点乘积正比于一个高斯：

$$p(\mathbf{x})\,q(\mathbf{x}) \propto \mathcal{N}(\mathbf{x};\boldsymbol{\mu},\boldsymbol{\Sigma}) \tag{Eq.2-7}$$

其中：

$$\boldsymbol{\Sigma} = \big(\boldsymbol{\Sigma}_p^{-1} + \boldsymbol{\Sigma}_q^{-1}\big)^{-1}, \quad \boldsymbol{\mu} = \boldsymbol{\Sigma}\big(\boldsymbol{\Sigma}_p^{-1}\boldsymbol{\mu}_p + \boldsymbol{\Sigma}_q^{-1}\boldsymbol{\mu}_q\big) \tag{Eq.2-8}$$

即**精度**（逆方差）相加，均值为精度加权平均。

> **证明**：将每个高斯写成指数（"自然"）形式，丢弃不依赖于 $\mathbf{x}$ 的常数：
> $$p(\mathbf{x}) \propto \exp\big(-\tfrac{1}{2}\mathbf{x}^\top\boldsymbol{\Sigma}_p^{-1}\mathbf{x} + \mathbf{x}^\top\boldsymbol{\Sigma}_p^{-1}\boldsymbol{\mu}_p\big)$$
> $q(\mathbf{x})$ 类似。相乘并合并同类项得：
> $$p(\mathbf{x})\,q(\mathbf{x}) \propto \exp\Big(-\tfrac{1}{2}\mathbf{x}^\top\underbrace{(\boldsymbol{\Sigma}_p^{-1}+\boldsymbol{\Sigma}_q^{-1})}_{\boldsymbol{\Lambda}}\mathbf{x} + \mathbf{x}^\top\underbrace{(\boldsymbol{\Sigma}_p^{-1}\boldsymbol{\mu}_p+\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\mu}_q)}_{\boldsymbol{\eta}}\Big)$$
> 
> 应用引理 2.10，将指数改写为 $-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Lambda}(\mathbf{x}-\boldsymbol{\mu}) + \text{const}$，其中 $\boldsymbol{\mu} = \boldsymbol{\Lambda}^{-1}\boldsymbol{\eta}$。识别出 $\boldsymbol{\Sigma} = \boldsymbol{\Lambda}^{-1}$ 即得所述均值和协方差。

**推论 2.12**（各向同性特殊情形）。若 $\boldsymbol{\Sigma}_p = \sigma_p^2 \mathbf{I}_d$，$\boldsymbol{\Sigma}_q = \sigma_q^2 \mathbf{I}_d$，则：

$$\boldsymbol{\Sigma} = \Big(\frac{1}{\sigma_p^2}+\frac{1}{\sigma_q^2}\Big)^{-1}\mathbf{I}_d, \quad \boldsymbol{\mu} = \Big(\frac{1}{\sigma_p^2}+\frac{1}{\sigma_q^2}\Big)^{-1}\Big(\frac{\boldsymbol{\mu}_p}{\sigma_p^2}+\frac{\boldsymbol{\mu}_q}{\sigma_q^2}\Big) \tag{Eq.2-9}$$

该代数可推广到任意有限个高斯因子：若 $p_k(\mathbf{x}) = \mathcal{N}(\mathbf{x};\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$，$k=1,\ldots,n$，则其乘积正比于精度为所有精度之和、均值为精度加权平均的高斯：

$$\boldsymbol{\Sigma}^{-1} = \sum_{k=1}^{n}\boldsymbol{\Sigma}_k^{-1}, \quad \boldsymbol{\mu} = \boldsymbol{\Sigma}\sum_{k=1}^{n}\boldsymbol{\Sigma}_k^{-1}\boldsymbol{\mu}_k \tag{Eq.2-10}$$

直觉上：精度相加，均值按各分量精度的比例被拉向各分量均值。

### 2.5 高斯分布之间的 KL 散度

现在我们计算两个多元高斯之间的 Kullback-Leibler 散度：

$$P = \mathcal{N}(\boldsymbol{\mu}_P, \boldsymbol{\Sigma}_P), \quad Q = \mathcal{N}(\boldsymbol{\mu}_Q, \boldsymbol{\Sigma}_Q)$$

二者均在 $\mathbb{R}^d$ 上，协方差对称正定。根据定义：

$$\text{KL}(P\|Q) = \mathbb{E}_P\Big[\log\frac{p(\mathbf{x})}{q(\mathbf{x})}\Big] = \mathbb{E}_P\big[\log p(\mathbf{x}) - \log q(\mathbf{x})\big]$$

**命题 2.13**（高斯的 KL 散度）。对上述 $P$ 和 $Q$：

$$\text{KL}(P\|Q) = \tfrac{1}{2}\Big(\text{tr}(\boldsymbol{\Sigma}_Q^{-1}\boldsymbol{\Sigma}_P) + (\boldsymbol{\mu}_Q-\boldsymbol{\mu}_P)^\top\boldsymbol{\Sigma}_Q^{-1}(\boldsymbol{\mu}_Q-\boldsymbol{\mu}_P) - d + \log\frac{\det(\boldsymbol{\Sigma}_Q)}{\det(\boldsymbol{\Sigma}_P)}\Big) \tag{4}$$

> **证明**：从 $p(\mathbf{x})$ 的密度出发取对数：
> $$\log p(\mathbf{x}) = -\tfrac{d}{2}\log(2\pi) - \tfrac{1}{2}\log\det(\boldsymbol{\Sigma}_P) - \tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_P)^\top\boldsymbol{\Sigma}_P^{-1}(\mathbf{x}-\boldsymbol{\mu}_P)$$
> 
> $\log q(\mathbf{x})$ 类似（用 $\boldsymbol{\mu}_Q, \boldsymbol{\Sigma}_Q$）。因此：
> $$\log p(\mathbf{x}) - \log q(\mathbf{x}) = \tfrac{1}{2}\log\frac{\det(\boldsymbol{\Sigma}_Q)}{\det(\boldsymbol{\Sigma}_P)} - \tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_P)^\top\boldsymbol{\Sigma}_P^{-1}(\mathbf{x}-\boldsymbol{\mu}_P) + \tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_Q)^\top\boldsymbol{\Sigma}_Q^{-1}(\mathbf{x}-\boldsymbol{\mu}_Q)$$
> 
> 两侧取 $\mathbb{E}_P[\cdot]$。现在利用前一小节的二次型恒等式计算两个期望。
> 
> 首先：$\mathbb{E}_P[(\mathbf{x}-\boldsymbol{\mu}_P)^\top\boldsymbol{\Sigma}_P^{-1}(\mathbf{x}-\boldsymbol{\mu}_P)] = \text{tr}(\boldsymbol{\Sigma}_P^{-1}\boldsymbol{\Sigma}_P) = d$
> 
> 对于第二个，写 $\mathbf{x}-\boldsymbol{\mu}_Q = (\mathbf{x}-\boldsymbol{\mu}_P)+(\boldsymbol{\mu}_P-\boldsymbol{\mu}_Q)$ 并展开：
> $$\mathbb{E}_P[(\mathbf{x}-\boldsymbol{\mu}_Q)^\top\boldsymbol{\Sigma}_Q^{-1}(\mathbf{x}-\boldsymbol{\mu}_Q)] = \text{tr}(\boldsymbol{\Sigma}_Q^{-1}\boldsymbol{\Sigma}_P) + (\boldsymbol{\mu}_P-\boldsymbol{\mu}_Q)^\top\boldsymbol{\Sigma}_Q^{-1}(\boldsymbol{\mu}_P-\boldsymbol{\mu}_Q)$$
> 
> 中间的交叉项消失因为 $\mathbb{E}_P[\mathbf{x}-\boldsymbol{\mu}_P] = \mathbf{0}$。将两个期望代回并化简即得公式 (4)。$\square$

**特殊情形**：若两个协方差都是对角的，公式 (4) 按分量简化为：

$$\text{KL}(P\|Q) = \tfrac{1}{2}\sum_{i=1}^{d}\bigg(\frac{\sigma_{P,i}^2}{\sigma_{Q,i}^2} + \frac{(\mu_{P,i}-\mu_{Q,i})^2}{\sigma_{Q,i}^2} - 1 + \log\frac{\sigma_{Q,i}^2}{\sigma_{P,i}^2}\bigg) \tag{Eq.2-11}$$

若二者都是各向同性且方差相同，$\boldsymbol{\Sigma}_P = \boldsymbol{\Sigma}_Q = \sigma^2\mathbf{I}_d$，则 $\text{KL}(P\|Q) = \frac{1}{2\sigma^2}\|\boldsymbol{\mu}_P-\boldsymbol{\mu}_Q\|_2^2$。

若二者都是各向同性但方差不同：

$$\text{KL}(P\|Q) = \tfrac{1}{2}\bigg(d\Big(\frac{\sigma_P^2}{\sigma_Q^2}-1-\log\frac{\sigma_P^2}{\sigma_Q^2}\Big) + \frac{1}{\sigma_Q^2}\|\boldsymbol{\mu}_P-\boldsymbol{\mu}_Q\|_2^2\bigg) \tag{Eq.2-12}$$

---

## 3 扩散模型

扩散模型有两个耦合的随机过程，作用于数据向量 $\mathbf{x} \in \mathbb{R}^d$：

- 一个**前向**（加噪）过程 $q$，通过在 $T$ 个小步骤中添加高斯噪声来逐步破坏结构；以及
- 一个**逆向**（去噪）过程 $p_\theta$，学习反转这些步骤以从噪声中恢复数据。

### 3.1 前向过程

前向过程整条路径的联合概率遵循马尔可夫性质：

$$q(\mathbf{x}_T, \mathbf{x}_{T-1}, \ldots, \mathbf{x}_1 \mid \mathbf{x}_0) = q(\mathbf{x}_T \mid \mathbf{x}_{T-1}) \cdots q(\mathbf{x}_2 \mid \mathbf{x}_1)\,q(\mathbf{x}_1 \mid \mathbf{x}_0) \tag{Eq.3-0}$$

记为 $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$。高斯转移核 $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ 定义了前向扩散：

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\big(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t\mathbf{I}\big) \tag{5}$$

其中 $\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}$ 为均值，$\beta_t\mathbf{I}$ 为协方差。$\mathbf{I}$ 是单位矩阵，$\beta_t \in (0,1)$ 是预设噪声调度中的噪声水平参数：

$$\beta_1 < \beta_2 < \cdots < \beta_T$$

前向过程被设计为使终态（近似地）成为标准高斯。定义：

$$\alpha_t := 1 - \beta_t, \quad \bar{\alpha}_t := \prod_{i=1}^{t}\alpha_i \tag{6}$$

则随着 $T$ 增大，累积乘积 $\bar{\alpha}_T = \prod_{i=1}^{T}\alpha_i$ 趋近于 0，边际分布 $q(\mathbf{x}_T \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_T}\,\mathbf{x}_0,\,(1-\bar{\alpha}_T)\mathbf{I})$ 趋近于 $\mathcal{N}(\mathbf{0},\mathbf{I})$，因此我们设 $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{0},\mathbf{I})$。利用此设定，公式 (5) 可改写为：

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\big(\mathbf{x}_t;\, \sqrt{\alpha_t}\,\mathbf{x}_{t-1},\, (1-\alpha_t)\mathbf{I}\big) \tag{7}$$

我们将证明，要生成 $\mathbf{x}_t$，只需 $\mathbf{x}_0$（加上新的高斯噪声）。利用公式 (7) 和推论 2.7 的重参数化技巧，我们写出：

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}, \quad \mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\,\boldsymbol{\epsilon}, \quad \ldots \tag{8}$$

其中每个 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ 是独立抽取的。将 $\mathbf{x}_{t-1}$ 代入第一行得：

$$\mathbf{x}_t = \sqrt{\alpha_t}\big(\sqrt{\alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\,\boldsymbol{\epsilon}\big) + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}$$
$$= \sqrt{\alpha_t\alpha_{t-1}}\,\mathbf{x}_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\boldsymbol{\epsilon}}_{\sim\mathcal{N}(\mathbf{0},\,\alpha_t(1-\alpha_{t-1})\mathbf{I})} + \underbrace{\sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}}_{\sim\mathcal{N}(\mathbf{0},\,(1-\alpha_t)\mathbf{I})} \tag{9}$$

最后两项是独立高斯，因此其和也是高斯，方差为方差之和：

$$\sqrt{\alpha_t(1-\alpha_{t-1})}\,\boldsymbol{\epsilon} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon} \sim \mathcal{N}\big(\mathbf{0},\,(1-\alpha_t\alpha_{t-1})\mathbf{I}\big)$$

因此：

$$\mathbf{x}_t = \sqrt{\alpha_t\alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\boldsymbol{\epsilon} \tag{10}$$

继续递推替换 $\mathbf{x}_{t-2}$，然后 $\mathbf{x}_{t-3}$，依此类推。每一步方差合并的模式相同。通过归纳法，直到 $\mathbf{x}_t$ 用 $\mathbf{x}_0$ 表示，得到：

$$\mathbf{x}_t = \sqrt{\prod_{i=1}^{t}\alpha_i}\,\mathbf{x}_0 + \sqrt{1-\prod_{i=1}^{t}\alpha_i}\,\boldsymbol{\epsilon}$$

利用公式 (6) 中的记号，任意时间 $t$ 的封闭形式采样方程为：

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon} \tag{12}$$

因此，由于 $\mathbf{x}_0$ 在条件下被视为给定的（非随机的），$\mathbf{x}_t$ 的分布以 $\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0$ 为均值、$(1-\bar{\alpha}_t)\mathbf{I}$ 为协方差：

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\big(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I}\big) \tag{13}$$

### 3.2 逆过程与真实后验（DDPM 后验）

整个逆马尔可夫过程的联合概率为：

$$p_\theta(\mathbf{x}_T, \mathbf{x}_{T-1}, \ldots, \mathbf{x}_1, \mathbf{x}_0) = p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1) \cdots p_\theta(\mathbf{x}_{T-1} \mid \mathbf{x}_T)\,p_\theta(\mathbf{x}_T)$$

记为 $p_\theta(\mathbf{x}_{0:T})$。

真实后验（DDPM 后验）可以写成高斯形式：

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\big(\mathbf{x}_{t-1};\, \hat{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0),\, \hat{\boldsymbol{\Sigma}}(\mathbf{x}_t, \mathbf{x}_0)\big) \tag{14}$$

其中 $\hat{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0) \in \mathbb{R}^d$ 和 $\hat{\boldsymbol{\Sigma}}(\mathbf{x}_t, \mathbf{x}_0) \in \mathbb{R}^{d \times d}$ 分别是均值和协方差。由于前向噪声是各向同性的，**真实后验协方差也是各向同性的**：

$$\hat{\boldsymbol{\Sigma}}(\mathbf{x}_t, \mathbf{x}_0) = \hat{\sigma}_t^2\,\mathbf{I}_d, \quad \hat{\sigma}_t^2 = \tilde{\beta}_t := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t \tag{Eq.3-1}$$

**推导 $\hat{\boldsymbol{\mu}}$ 和 $\hat{\sigma}_t^2$**：由贝叶斯规则：

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)\,q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}$$

由于前向过程是马尔可夫的，$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$。利用三个高斯密度的显式形式（均为各向同性），丢弃不依赖于 $\mathbf{x}_{t-1}$ 的项后，将指数中的项收集成关于 $\mathbf{x}_{t-1}$ 的二次型：

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \propto \exp\Big(-\tfrac{1}{2}\Big[\Big(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}\Big)\mathbf{x}_{t-1}^\top\mathbf{x}_{t-1} - 2\Big(\frac{\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}_0\Big)^\top\mathbf{x}_{t-1} + C(\mathbf{x}_t,\mathbf{x}_0)\Big]\Big) \tag{15}$$

其中 $C(\mathbf{x}_t,\mathbf{x}_0)$ 不含 $\mathbf{x}_{t-1}$ 项。定义：

$$A := \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}, \quad \mathbf{b} := \frac{\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}_0$$

配方得：

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \propto \exp\Big(-\tfrac{1}{2}A\big\|\mathbf{x}_{t-1} - \tfrac{\mathbf{b}}{A}\big\|_2^2 + \tfrac{\mathbf{b}^\top\mathbf{b}}{2A}\Big)$$

因此**后验均值**为：

$$\hat{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0) = \frac{\mathbf{b}}{A} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\mathbf{x}_0 \tag{16}$$

而**后验协方差**是标量精度 $A$ 乘以 $\mathbf{I}_d$ 的逆：

$$\hat{\boldsymbol{\Sigma}}(\mathbf{x}_t, \mathbf{x}_0) = A^{-1}\mathbf{I}_d, \quad A^{-1} = \frac{1}{\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}} = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{\alpha_t(1-\bar{\alpha}_{t-1})+\beta_t} = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} = \tilde{\beta}_t$$

因此 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\big(\hat{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0),\, \tilde{\beta}_t\,\mathbf{I}_d\big)$。

由公式 (12) 可以写出：

$$\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\big(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}\big)$$

代入公式 (16)，得到方便的 **$\boldsymbol{\epsilon}$-形式**：

$$\hat{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0) = \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}\Big) \tag{17}$$

### 3.3 损失函数

我们关心的是 $p_\theta(\mathbf{x}_0)$，即学习到的模型赋予生成样本 $\mathbf{x}_0$ 的概率。起点是：

$$p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\,p_\theta(\mathbf{x}_0) = p_\theta(\mathbf{x}_{0:T})$$

虽然这看起来是反因果的，但恒等式在代数上成立。我们的目标是通过非负的 KL 散度使 $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ 和 $p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ 接近：

$$0 \leq \text{KL}\big(q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\big) = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}$$

**化简期望**。利用 $p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)$，逐步展开：

$$\text{KL}\big(q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\big) = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}$$

$$= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\,p_\theta(\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})}$$

$$= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)$$

由于 $\log p_\theta(\mathbf{x}_0)$ 不依赖于 $\mathbf{x}_{1:T}$，所以：

$$\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0) = \log p_\theta(\mathbf{x}_0)$$

因此：

$$\text{KL}\big(q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\big) = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\log\frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \geq 0$$

$$\implies \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\log\frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \geq -\log p_\theta(\mathbf{x}_0)$$

对 $q(\mathbf{x}_0)$ 取期望：

$$\mathbb{E}_{q(\mathbf{x}_{0:T})}\log\frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \geq -\mathbb{E}_{q(\mathbf{x}_0)}\log p_\theta(\mathbf{x}_0)$$

右侧与 $\text{KL}(q(\mathbf{x}_0)\|p_\theta(\mathbf{x}_0)) = \text{const} - \mathbb{E}_{q(\mathbf{x}_0)}\log p_\theta(\mathbf{x}_0)$ 相关，这是我们希望最小化的量，但 $\log p_\theta(\mathbf{x}_0)$ 难以计算。因此我们转而最小化左侧的**变分界（VB）**（与 VAE 中的证据下界 ELBO 密切相关）。

**展开 VB**。将前向和逆向的马尔可夫分解代入：

$$\mathbb{E}_{q(\mathbf{x}_{0:T})}\log\frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} = \mathbb{E}_{q(\mathbf{x}_{0:T})}\log\frac{q(\mathbf{x}_T \mid \mathbf{x}_{T-1})\cdots q(\mathbf{x}_2 \mid \mathbf{x}_1)\,q(\mathbf{x}_1 \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)\cdots p_\theta(\mathbf{x}_{T-1} \mid \mathbf{x}_T)\,p_\theta(\mathbf{x}_T)}$$

$$= \mathbb{E}_{q(\mathbf{x}_{0:T})}\log\frac{q(\mathbf{x}_1 \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)} - \mathbb{E}_{q(\mathbf{x}_{0:T})}\log p_\theta(\mathbf{x}_T) + \mathbb{E}_{q(\mathbf{x}_{0:T})}\sum_{t=2}^{T}\log\frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}$$

由于前向过程是马尔可夫的，$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)$，故上式等价于：

$$= \mathbb{E}_{q(\mathbf{x}_{0:T})}\log\frac{q(\mathbf{x}_1 \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)} - \mathbb{E}_{q(\mathbf{x}_{0:T})}\log p_\theta(\mathbf{x}_T) + \mathbb{E}_{q(\mathbf{x}_{0:T})}\sum_{t=2}^{T}\log\frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}$$

**应用贝叶斯定理**。对前向转移应用贝叶斯定理 $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) = \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)\,q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}$，代入得：

$$\text{VB} = \mathbb{E}_{q(\mathbf{x}_{0:T})}\log\frac{q(\mathbf{x}_1 \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)} - \mathbb{E}_{q(\mathbf{x}_{0:T})}\log p_\theta(\mathbf{x}_T) + \mathbb{E}_{q(\mathbf{x}_{0:T})}\sum_{t=2}^{T}\log\frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} + \mathbb{E}_{q(\mathbf{x}_{0:T})}\sum_{t=2}^{T}\log\frac{q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}$$

**伸缩求和化简**。最后一项是伸缩和（telescoping sum）：

$$\mathbb{E}_{q(\mathbf{x}_{0:T})}\sum_{t=2}^{T}\log\frac{q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)} = \mathbb{E}_{q(\mathbf{x}_{0:T})}\log\frac{q(\mathbf{x}_T \mid \mathbf{x}_0)}{q(\mathbf{x}_1 \mid \mathbf{x}_0)}$$

将此与第一项 $\log\frac{q(\mathbf{x}_1 \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}$ 和 $-\log p_\theta(\mathbf{x}_T)$ 合并，$q(\mathbf{x}_1 \mid \mathbf{x}_0)$ 消去，得到最终的**三项分解**：

$$\text{VB} = \underbrace{\mathbb{E}_{q(\mathbf{x}_{0:T})}\log\frac{q(\mathbf{x}_T \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)}}_{\textcircled{I}} - \underbrace{\mathbb{E}_{q(\mathbf{x}_{0:T})}\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}_{\textcircled{II}} + \underbrace{\mathbb{E}_{q(\mathbf{x}_{0:T})}\sum_{t=2}^{T}\log\frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}}_{\textcircled{III}}$$

- **$\textcircled{I}$**：$\mathbb{E}_{q(\mathbf{x}_{0:T})}\log\frac{q(\mathbf{x}_T \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)}$——当 $q(\mathbf{x}_T \mid \mathbf{x}_0) = p_\theta(\mathbf{x}_T)$ 时为 0；否则当 $p_\theta(\mathbf{x}_T) = \mathcal{N}(\mathbf{0},\mathbf{I})$ 且 $T$ 足够大时为一个小常数。
- **$\textcircled{II}$**：$\mathbb{E}_{q(\mathbf{x}_{0:T})}\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$——在实践中通常被忽略（充当 $t{=}1$ 的解码器项）。
- **$\textcircled{III}$**：$\mathbb{E}_{q(\mathbf{x}_{0:T})}\sum_{t=2}^{T}\log\frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}$——归结为各向同性高斯之间 KL 散度之和（封闭形式）。

因此优化鼓励 $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 去匹配 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$，对 $t = 2,\ldots,T$：

$$\text{VB} = \mathbb{E}_{q(\mathbf{x}_{0:T})}\sum_{t=2}^{T}\log\frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} \cong \sum_{t=2}^{T}\text{KL}\big\{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big\} \tag{18}$$

这两个高斯分别是：

- $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\Big(\mathbf{x}_{t-1};\, \frac{1}{\sqrt{\alpha_t}}\big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_t\big),\, \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t\,\mathbf{I}\Big)$

- $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\big(\mathbf{x}_{t-1};\, \mu_\theta(\mathbf{x}_t, t),\, \sigma_t^2\,\mathbf{I}\big)$

设 $\sigma_t^2 = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t := \tilde{\beta}_t$。则：

$$\text{KL}\big\{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big\} = \frac{1}{2\sigma_t^2}\Big\|\frac{1}{\sqrt{1-\beta_t}}\big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_t\big) - \mu_\theta(\mathbf{x}_t, t)\Big\|_2^2$$

$$= \frac{1-\bar{\alpha}_t}{2\beta_t(1-\bar{\alpha}_{t-1})}\Big\|\frac{1}{\sqrt{1-\beta_t}}\big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_t\big) - \mu_\theta(\mathbf{x}_t, t)\Big\|_2^2$$

利用参数化 [6]：

$$\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\boldsymbol{\epsilon}}_t\Big)$$

得到简单的加权噪声匹配目标：

$$\text{KL}\big\{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big\} = \frac{\beta_t}{2(1-\beta_t)(1-\bar{\alpha}_{t-1})}\|\boldsymbol{\epsilon}_t - \hat{\boldsymbol{\epsilon}}_t\|_2^2 \tag{Eq.3-2}$$

因此：

$$\text{VB} = \sum_{t=2}^{T}\frac{\beta_t}{2(1-\beta_t)(1-\bar{\alpha}_{t-1})}\|\boldsymbol{\epsilon}_t - \hat{\boldsymbol{\epsilon}}_t\|_2^2 \tag{Eq.3-3}$$

> **核心直觉**：训练目标本质上就是让网络学会预测在每一步中添加的噪声 $\boldsymbol{\epsilon}$。这就是实践中广泛使用的**噪声预测**损失。

### 3.4 计算 $p_\theta(\mathbf{x}_0)$

利用马尔可夫链写出完整逆向路径的联合概率：

$$p_\theta(\mathbf{x}_{0:T}) = p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)\,p_\theta(\mathbf{x}_1 \mid \mathbf{x}_2) \cdots p_\theta(\mathbf{x}_{T-1} \mid \mathbf{x}_T)\,p_\theta(\mathbf{x}_T) \tag{19}$$

则 $p_\theta(\mathbf{x}_0)$ 通过对 $\mathbf{x}_{1:T}$ 边际化得到：

$$p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T})\,\mathrm{d}\mathbf{x}_{1:T} \tag{20}$$

直接计算代价高昂，因此引入**重要性采样**，给积分乘以恒等于 1 的比值 $\frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}$，不改变积分结果，但将积分测度换成易于采样的前向分布 $q$：

$$p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T})\,\frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\,\mathrm{d}\mathbf{x}_{1:T} = \int q(\mathbf{x}_{1:T}|\mathbf{x}_0)\,\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\,\mathrm{d}\mathbf{x}_{1:T}$$

将 $p_\theta(\mathbf{x}_{0:T})$（式19）与前向联合分布 $q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_{t-1})$ 各自展开后，分子分母逐步对消，得到：

$$p_\theta(\mathbf{x}_0) = \int q(\mathbf{x}_{1:T}|\mathbf{x}_0)\,p_\theta(\mathbf{x}_T)\,\frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)\,p_\theta(\mathbf{x}_1|\mathbf{x}_2)\cdots p_\theta(\mathbf{x}_{T-1}|\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})\cdots q(\mathbf{x}_2|\mathbf{x}_1)\,q(\mathbf{x}_1|\mathbf{x}_0)}\,\mathrm{d}\mathbf{x}_{1:T}$$

根据期望的定义，上式等价于：

$$p_\theta(\mathbf{x}_0) = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\bigg[p_\theta(\mathbf{x}_T)\,\frac{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)\,p_\theta(\mathbf{x}_1 \mid \mathbf{x}_2) \cdots p_\theta(\mathbf{x}_{T-1} \mid \mathbf{x}_T)}{q(\mathbf{x}_T \mid \mathbf{x}_{T-1}) \cdots q(\mathbf{x}_2 \mid \mathbf{x}_1)\,q(\mathbf{x}_1 \mid \mathbf{x}_0)}\bigg] \tag{21}$$

实际使用的逆时间条件分布为：

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\bigg(\mathbf{x}_{t-1};\, \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\boldsymbol{\epsilon}}_t\Big),\, \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t\,\mathbf{I}\bigg)$$

### 3.5 补充说明：计算 $p_\theta(\mathbf{x}_0)$ 的公式推导详细解读

这部分是**扩散模型（Diffusion Model）** 中，从完整反向路径的联合概率，推导初始样本 $\mathbf{x}_0$ 的边缘概率 $p_\theta(\mathbf{x}_0)$ 的核心过程，本质是用**重要性采样（Importance Sampling）** 解决高维积分的计算难题。下面我们逐行拆解每一步的逻辑、原理和意义。

---

#### 背景铺垫：扩散模型的反向马尔可夫链

扩散模型的核心是两个过程：
1.  **前向过程（加噪）**：从真实数据 $\mathbf{x}_0$ 逐步加噪，得到 $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T$，最终 $\mathbf{x}_T$ 近似标准高斯分布。
2.  **反向过程（去噪）**：学习一个马尔可夫链 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$，从 $\mathbf{x}_T$ 逐步去噪，还原出 $\mathbf{x}_0$。

这部分推导的目标，就是**从反向过程的联合概率，计算初始样本 $\mathbf{x}_0$ 的边缘概率 $p_\theta(\mathbf{x}_0)$**，也就是模型对真实数据的拟合概率。

---

#### 逐公式拆解推导

**1. 公式(19)：完整反向路径的联合概率**

$$
p_\theta(\mathbf{x}_{0:T}) = p_\theta(\mathbf{x}_0 | \mathbf{x}_1) p_\theta(\mathbf{x}_1 | \mathbf{x}_2) \cdots p_\theta(\mathbf{x}_{T-1} | \mathbf{x}_T) p_\theta(\mathbf{x}_T)
$$

- **逻辑**：反向过程是**马尔可夫链**，满足马尔可夫性：未来状态只依赖当前状态，与历史无关。
- **展开**：联合概率 = 初始噪声分布 $p_\theta(\mathbf{x}_T)$（通常取标准高斯 $\mathcal{N}(\mathbf{0}, \mathbf{I})$） × 每一步的反向转移概率 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 的连乘。
- **意义**：完整描述了从 $\mathbf{x}_T$ 到 $\mathbf{x}_0$ 的整个去噪路径的概率。

---

**2. 公式(20)：边缘概率的定义（高维积分）**

$$
p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}
$$

- **逻辑**：边缘概率的定义：要得到 $\mathbf{x}_0$ 的概率，需要对所有中间变量 $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T$ 积分（也就是"边缘化"掉中间变量）。
- **问题**：$\mathbf{x}_{1:T}$ 是高维向量（比如图像是 $H\times W\times 3$ 维，$T$ 通常取1000），直接计算这个高维积分**计算量爆炸，完全不可行**，因此需要用重要性采样来近似。

---

**3. 重要性采样的引入：公式变形**

直接计算积分太贵，因此引入**重要性采样**：给积分乘上一个恒为1的比值 $\frac{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}$，其中 $q(\cdot)$ 是**前向过程的分布**（已知、可采样，不需要学习）。

$$
p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) \frac{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} d\mathbf{x}_{1:T} = \int q(\mathbf{x}_{1:T} | \mathbf{x}_0) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} d\mathbf{x}_{1:T}
$$

- **原理**：重要性采样的核心思想：
  $$
  \mathbb{E}_p[f(X)] = \int p(x) f(x) dx = \int q(x) \cdot \frac{p(x)}{q(x)} f(x) dx = \mathbb{E}_q\left[ \frac{p(x)}{q(x)} f(x) \right]
  $$
  这里我们把积分从对 $p_\theta$ 积分，转化为对**易采样的前向分布 $q$** 积分，用权重 $\frac{p_\theta}{q}$ 修正分布差异。
- **意义**：把不可计算的高维积分，转化为了对前向分布 $q$ 的期望，而 $q$ 是已知的，可以通过采样来近似期望。

---

**4. 展开项：公式进一步化简**

接下来把 $p_\theta(\mathbf{x}_{0:T})$ 和 $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ 展开：
- $p_\theta(\mathbf{x}_{0:T})$ 就是公式(19)：$p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$
- $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ 是前向过程的联合概率，同样满足马尔可夫性：$q(\mathbf{x}_1 | \mathbf{x}_0) q(\mathbf{x}_2 | \mathbf{x}_1) \cdots q(\mathbf{x}_T | \mathbf{x}_{T-1})$

代入后得到：

$$
p_\theta(\mathbf{x}_0) = \int q(\mathbf{x}_{1:T} | \mathbf{x}_0) p_\theta(\mathbf{x}_T) \frac{p_\theta(\mathbf{x}_0 | \mathbf{x}_1) p_\theta(\mathbf{x}_1 | \mathbf{x}_2) \cdots p_\theta(\mathbf{x}_{T-1} | \mathbf{x}_T)}{q(\mathbf{x}_T | \mathbf{x}_{T-1}) \cdots q(\mathbf{x}_3 | \mathbf{x}_2) q(\mathbf{x}_2 | \mathbf{x}_1) q(\mathbf{x}_1 | \mathbf{x}_0)} d\mathbf{x}_{1:T}
$$

- **逻辑**：分子是反向过程的转移概率连乘，分母是前向过程的转移概率连乘，把比值拆成了每一步的概率比。
- **意义**：把联合概率的比值，拆成了每一步去噪/加噪的概率比，为后续转化为期望做准备。

---

**5. 公式(21)：转化为期望形式**

根据期望的定义：$\int q(x) f(x) dx = \mathbb{E}_q[f(x)]$，上面的积分可以直接写成对 $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ 的期望：

$$
p_\theta(\mathbf{x}_0) = \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ p_\theta(\mathbf{x}_T) \frac{p_\theta(\mathbf{x}_0 | \mathbf{x}_1) p_\theta(\mathbf{x}_1 | \mathbf{x}_2) \cdots p_\theta(\mathbf{x}_{T-1} | \mathbf{x}_T)}{q(\mathbf{x}_T | \mathbf{x}_{T-1}) \cdots q(\mathbf{x}_2 | \mathbf{x}_1) q(\mathbf{x}_1 | \mathbf{x}_0)} \right]
$$

- **核心突破**：把高维积分转化为了**蒙特卡洛可计算的期望**！
  - 我们可以从 $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ 中采样多条前向路径（也就是给 $\mathbf{x}_0$ 加噪得到 $\mathbf{x}_{1:T}$），然后计算每条路径的权重，取平均就可以近似 $p_\theta(\mathbf{x}_0)$。
- **意义**：这是扩散模型中**变分下界（ELBO）**推导的核心步骤，后续的损失函数就是基于这个期望的对数来构造的。

---

**6. 最后一行：实际使用的反向转移概率**

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}\left( \mathbf{x}_{t-1} ; \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_t \right), \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t \mathbf{I} \right)
$$

- **符号说明**：
  - $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$（前向过程的累积系数，超参数）
  - $\hat{\epsilon}_t$ 是模型预测的噪声（扩散模型的核心学习目标：预测每一步加的噪声）
  - $\mathcal{N}(\cdot ; \mu, \Sigma)$ 表示高斯分布，均值为 $\mu$，协方差为 $\Sigma$
- **逻辑**：这是扩散模型中**实际训练的反向转移概率**，均值由模型预测的噪声 $\hat{\epsilon}_t$ 决定，协方差是固定的超参数（由前向过程的 $\beta_t$ 决定）。
- **意义**：给出了公式(19)中 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 的具体形式，是模型训练和采样的基础。

---

#### 整个推导的核心逻辑总结

1.  **目标**：计算初始数据 $\mathbf{x}_0$ 的概率 $p_\theta(\mathbf{x}_0)$，衡量模型对真实数据的拟合能力。
2.  **问题**：直接边缘化中间变量的高维积分不可计算。
3.  **解决方案**：用**重要性采样**，把对反向分布 $p_\theta$ 的积分，转化为对已知的前向分布 $q$ 的期望。
4.  **结果**：得到了可通过蒙特卡洛采样近似的期望形式，为扩散模型的**变分下界（ELBO）损失**提供了理论基础。
5.  **落地**：给出了实际使用的反向转移概率的高斯分布形式，明确了模型的学习目标（预测噪声 $\hat{\epsilon}_t$）。

---

#### 补充：与扩散模型训练的关联

这个推导是扩散模型**损失函数**的源头：
- 我们的训练目标是最大化 $\log p_\theta(\mathbf{x}_0)$，也就是最大化公式(21)中期望的对数。
- 通过Jensen不等式，可以得到对数似然的变分下界（ELBO）：
  $$
  \log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_q\left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \right]
  $$
- 最大化这个下界，就等价于最小化扩散模型的损失函数（通常简化为噪声预测的MSE损失）。

---

#### 关键概念补充

| 概念 | 含义 |
|------|------|
| 马尔可夫链 | 未来状态仅依赖当前状态，与历史无关，是扩散模型前后向过程的核心假设 |
| 边缘化（Marginalization） | 对联合概率中的无关变量积分，得到目标变量的边缘概率 |
| 重要性采样 | 用易采样的分布 $q$ 近似难采样的分布 $p$，通过权重修正分布差异，解决高维积分问题 |
| 变分下界（ELBO） | 对数似然的下界，是扩散模型训练的核心优化目标 |
| 反向转移概率 $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ | 模型学习的去噪步骤，从 $\mathbf{x}_t$ 预测 $\mathbf{x}_{t-1}$ 的分布 |
| 前向转移概率 $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ | 已知的加噪步骤，不需要学习，由超参数 $\beta_t$ 决定 |

### 3.6 补充证明：变分下界推导全解析

本节是对 3.3 节损失函数推导的教学性补充，从动机到最终结果逐步拆解逻辑，帮助彻底理解变分扩散模型（VDM）的变分下界（VB）推导。本质是用 KL 散度构造对数似然的下界，最终把训练目标转化为可计算的噪声匹配损失。

#### 核心目标：为什么要做这个推导？

**核心问题**。扩散模型的本质是：
- **前向过程（加噪）**：从干净数据 $\mathbf{x}_0$ 逐步加噪，得到 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$（$\mathbf{x}_T$ 近似标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$）
- **反向过程（去噪）**：从 $\mathbf{x}_T$ 逐步去噪，还原 $\mathbf{x}_0$，对应分布 $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$

我们的目标是**最大化对数似然 $\log p_\theta(\mathbf{x}_0)$**，但直接计算 $\log p_\theta(\mathbf{x}_0)$ 是不可行的（需要积分所有中间变量 $\mathbf{x}_{1:T}$），因此用**变分推断**构造一个**证据下界（ELBO，也就是这里的 VB）**：

$$\log p_\theta(\mathbf{x}_0) \geq \text{VB}$$

最大化 VB 等价于最大化对数似然的下界，从而间接优化模型。

**核心工具：KL 散度**。KL 散度衡量两个分布的"距离"，非负，当且仅当两个分布完全相同时为 0：

$$\text{KL}(q \,\|\, p) = \mathbb{E}_q\!\left[\log \frac{q}{p}\right] \geq 0$$

这是整个推导的起点。

#### 从 KL 散度到变分下界

**第一步：KL 散度的定义与展开**。用近似后验 $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$（前向加噪过程，已知、可计算）去逼近真实后验 $p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$（反向去噪过程，待优化），定义 KL 散度：

$$0 \leq \text{KL}\big(q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\big) = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right]$$

**第二步：贝叶斯定理拆分 $p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$**。根据贝叶斯定理：

$$p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \frac{p_\theta(\mathbf{x}_{1:T}, \mathbf{x}_0)}{p_\theta(\mathbf{x}_0)} = \frac{p_\theta(\mathbf{x}_{0:T})}{p_\theta(\mathbf{x}_0)}$$

代入 KL 散度：

$$\text{KL}\big(q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\big) = \mathbb{E}_{q} \left[ \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \cdot p_\theta(\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right] = \mathbb{E}_{q} \left[ \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right] + \mathbb{E}_{q} \left[ \log p_\theta(\mathbf{x}_0) \right]$$

**第三步：简化期望项**。注意 $\log p_\theta(\mathbf{x}_0)$ 不依赖 $\mathbf{x}_{1:T}$，因此对 $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ 求期望等于自身：

$$\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0) \right] = \log p_\theta(\mathbf{x}_0)$$

代入后得到：

$$\text{KL}\big(q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\big) = \mathbb{E}_{q} \left[ \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right] + \log p_\theta(\mathbf{x}_0) \geq 0$$

移项得到对数似然的下界：

$$\log p_\theta(\mathbf{x}_0) \geq - \mathbb{E}_{q} \left[ \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right]$$

**第四步：对 $q(\mathbf{x}_0)$ 取期望**。我们最终要优化的是**数据分布上的期望**，因此对 $q(\mathbf{x}_0)$（真实数据分布）取期望：

$$\mathbb{E}_{q(\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0) \right] \geq - \mathbb{E}_{q(\mathbf{x}_{0:T})} \left[ \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right]$$

右边就是**变分下界 VB**，我们的目标就是最小化 VB（等价于最大化对数似然）。

#### VB 的展开与化简

**第一步：展开马尔可夫链结构**。扩散模型的前向/反向过程都是马尔可夫链，因此：
- 前向：$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = q(\mathbf{x}_1 \mid \mathbf{x}_0)\,q(\mathbf{x}_2 \mid \mathbf{x}_1) \cdots q(\mathbf{x}_T \mid \mathbf{x}_{T-1})$
- 反向：$p_\theta(\mathbf{x}_{0:T}) = p_\theta(\mathbf{x}_T)\,p_\theta(\mathbf{x}_{T-1} \mid \mathbf{x}_T) \cdots p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$

代入 VB：

$$\mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} = \mathbb{E}_{q} \log \frac{q(\mathbf{x}_1 \mid \mathbf{x}_0) \prod_{t=2}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1) \prod_{t=2}^{T} p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \cdot p_\theta(\mathbf{x}_T)}$$

$$= \mathbb{E}_{q} \log \frac{q(\mathbf{x}_1 \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)} - \mathbb{E}_{q} \log p_\theta(\mathbf{x}_T) + \mathbb{E}_{q} \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}$$

**第二步：再次用贝叶斯定理拆分 $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)$**。对任意 $t \geq 2$，前向过程的后验满足：

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)\,q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}$$

因此：

$$\log \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} = \log \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)\,q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)\,p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} = \log \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} + \log \frac{q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}$$

**第三步：伸缩求和（Telescoping Sum）**。对 $t=2$ 到 $T$ 求和时，$\log \frac{q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}$ 形成伸缩和，中间项全部抵消：

$$\sum_{t=2}^{T} \log \frac{q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)} = \log \frac{q(\mathbf{x}_T \mid \mathbf{x}_0)}{q(\mathbf{x}_1 \mid \mathbf{x}_0)}$$

**第四步：合并得到最终 VB 分解**。把所有项合并，$q(\mathbf{x}_1 \mid \mathbf{x}_0)$ 消去，最终 VB 被拆分为 3 个可解释的部分：

$$\text{VB} = \underbrace{\mathbb{E}_{q} \log \frac{q(\mathbf{x}_T \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)}}_{\text{项 I}} - \underbrace{\mathbb{E}_{q} \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}_{\text{项 II}} + \underbrace{\mathbb{E}_{q} \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}}_{\text{项 III}}$$

#### 三个项的物理意义与简化

**项 I**：$\mathbb{E}_{q} \log \frac{q(\mathbf{x}_T \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)}$
- **意义**：衡量加噪后的 $\mathbf{x}_T$ 分布 $q(\mathbf{x}_T \mid \mathbf{x}_0)$ 与预设的先验 $p_\theta(\mathbf{x}_T)$（通常设为 $\mathcal{N}(\mathbf{0}, \mathbf{I})$）的 KL 散度。
- **简化**：当 $T$ 足够大时，$q(\mathbf{x}_T \mid \mathbf{x}_0)$ 会收敛到 $\mathcal{N}(\mathbf{0}, \mathbf{I})$，因此项 I 近似为 0，训练中可以忽略。

**项 II**：$-\mathbb{E}_{q} \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$
- **意义**：对应 $t=1$ 步的去噪损失，衡量从 $\mathbf{x}_1$ 还原 $\mathbf{x}_0$ 的重构误差。
- **简化**：在实践中通常被忽略（或合并到项 III），因为扩散模型的核心是中间步的去噪，最后一步的重构误差影响很小。

**项 III**：$\mathbb{E}_{q} \sum_{t=2}^{T} \log \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}$
- **意义**：**核心训练项**，每一项都是 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$（前向过程的真实后验，已知）和 $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$（模型预测的去噪分布，待优化）的 KL 散度。
- **优化目标**：最小化项 III，等价于让模型预测的去噪分布 $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 尽可能逼近真实后验 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$。

#### 高斯分布下的 KL 散度闭式解与最终损失

**核心前提：两个分布都是各向同性高斯**。扩散模型的前向/反向过程都假设为各向同性高斯分布，因此：

- 真实后验：$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\!\left( \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_t \right),\; \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t \,\mathbf{I} \right)$

  其中 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，$\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是加的噪声。

- 模型预测：$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\!\left( \mu_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \,\mathbf{I} \right)$

**高斯 KL 散度的闭式解**。对于两个各向同性高斯 $\mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$ 和 $\mathcal{N}(\boldsymbol{\mu}_2, \sigma_2^2 \mathbf{I})$，KL 散度的闭式解为：

$$\text{KL}\big(\mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I}) \,\|\, \mathcal{N}(\boldsymbol{\mu}_2, \sigma_2^2 \mathbf{I})\big) = \frac{1}{2\sigma_2^2} \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|_2^2 + \text{常数项（仅与方差有关）}$$

代入两个分布的均值和方差，化简后常数项被抵消，最终得到：

$$\text{KL}\big(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big) = \frac{\beta_t}{2(1-\beta_t)(1-\bar{\alpha}_{t-1})} \|\boldsymbol{\epsilon}_t - \hat{\boldsymbol{\epsilon}}_t\|_2^2$$

其中 $\hat{\boldsymbol{\epsilon}}_t$ 是模型预测的噪声（$\mu_\theta(\mathbf{x}_t, t)$ 由 $\hat{\boldsymbol{\epsilon}}_t$ 参数化）。

**最终训练目标：噪声匹配损失**。把所有 $t$ 的 KL 散度求和，忽略项 I 和项 II，最终 VB 简化为：

$$\text{VB} = \sum_{t=2}^{T} \frac{\beta_t}{2(1-\beta_t)(1-\bar{\alpha}_{t-1})} \|\boldsymbol{\epsilon}_t - \hat{\boldsymbol{\epsilon}}_t\|_2^2$$

这就是扩散模型的**核心训练目标**：让模型预测的噪声 $\hat{\boldsymbol{\epsilon}}_t$ 尽可能逼近真实加的噪声 $\boldsymbol{\epsilon}_t$，也就是"噪声匹配"。

#### 关键术语与符号对照表

| 符号 | 含义 |
|------|------|
| $\mathbf{x}_0$ | 干净数据（原始输入） |
| $\mathbf{x}_t$ | 第 $t$ 步加噪后的数据 |
| $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ | 前向加噪过程（已知，固定） |
| $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ | 反向去噪过程（待优化，模型） |
| $\beta_t$ | 加噪率（超参数，$0 < \beta_t < 1$） |
| $\alpha_t = 1-\beta_t$ | 加噪率的补 |
| $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ | 累计加噪率 |
| $\boldsymbol{\epsilon}_t$ | 真实加的噪声（$\mathcal{N}(\mathbf{0}, \mathbf{I})$） |
| $\hat{\boldsymbol{\epsilon}}_t$ | 模型预测的噪声 |
| $\text{KL}(q \,\|\, p)$ | KL 散度，衡量两个分布的距离 |
| VB | 变分下界（Variational Bound），对数似然的下界 |
| ELBO | 证据下界（Evidence Lower Bound），与 VB 等价 |

#### 推导的核心逻辑总结

1. **问题转化**：把"最大化对数似然"转化为"最小化变分下界 VB"，解决直接计算对数似然不可行的问题。
2. **结构拆分**：利用马尔可夫链和贝叶斯定理，把 VB 拆分为 3 个可解释的项，核心是中间步的 KL 散度项。
3. **高斯简化**：利用高斯分布 KL 散度的闭式解，把抽象的分布匹配转化为具体的"噪声预测 MSE 损失"。
4. **最终目标**：训练模型让预测的噪声 $\hat{\boldsymbol{\epsilon}}_t$ 逼近真实噪声 $\boldsymbol{\epsilon}_t$，完成去噪能力的学习。

#### 与 DDPM 的关系

我们熟知的 DDPM（Denoising Diffusion Probabilistic Models）就是这个推导的**简化版本**：
- DDPM 直接把 VB 中的项 I、项 II 忽略，只保留项 III 的噪声匹配损失。
- DDPM 把 $\frac{\beta_t}{2(1-\beta_t)(1-\bar{\alpha}_{t-1})}$ 简化为固定权重，最终损失为 $\mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}_t}\!\left[ \|\boldsymbol{\epsilon}_t - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)\|_2^2 \right]$，和这里的推导完全一致。

#### 常见问题

**为什么叫"变分"下界？** 因为我们用了一个近似分布 $q$（变分分布）去逼近真实后验 $p_\theta$，从而构造了对数似然的下界，这是变分推断的核心思想。

**为什么 KL 散度非负很重要？** KL 散度非负保证了我们构造的 VB 一定是对数似然的下界，因此最小化 VB 一定能提升对数似然，不会出现优化方向错误的问题。

**为什么最终变成了噪声预测？** 因为扩散模型的前向过程是加噪，反向过程本质是"预测加的噪声并去除"，因此用噪声匹配作为损失，是最直观、最容易优化的目标。

---

## 4 加速方法

DDPM 模型的一个关键缺点是需要很多次迭代才能产生高质量样本。

### 4.1 去噪扩散隐式模型（DDIM）

我们从去噪扩散隐式模型（DDIMs）开始。推导的起点是上一节得到的精确DDPM单步后验分布：

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_{t-1} ; \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon} \right), \sigma_t^2 \mathbf{I}_d \right), \quad \boldsymbol{\epsilon} = \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}} \tag{22}$$

其中 $\alpha_t=1-\beta_t$，$\bar{\alpha}_t=\prod_{i=1}^t\alpha_i$。当取 $\sigma_t^2=\tilde{\beta}_t:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ 时，就还原为标准DDPM的反向条件分布及其真实后验方差。利用2.3节中前向边缘分布的重参数化形式：

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon} \quad\Rightarrow\quad \boldsymbol{\epsilon}=\frac{\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}}$$

将该 $\boldsymbol{\epsilon}$ 代入式(22)中DDPM后验的均值部分：

$$\begin{align*}
\hat{\boldsymbol{\mu}}_t^{(\text{DDPM})}(\mathbf{x}_t,\mathbf{x}_0)
&= \frac{1}{\sqrt{\alpha_t}}\left(
\mathbf{x}_t
-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\cdot\frac{\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}}
\right)\\
&= \frac{1}{\sqrt{\alpha_t}}\left(
\mathbf{x}_t-\frac{\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_t
+\frac{\beta_t\sqrt{\bar{\alpha}_t}}{1-\bar{\alpha}_t}\mathbf{x}_0
\right)\\
&= \underbrace{\frac{1}{\sqrt{\alpha_t}}\left(1-\frac{\beta_t}{1-\bar{\alpha}_t}\right)}_{c_t}\mathbf{x}_t
+\underbrace{\frac{1}{\sqrt{\alpha_t}}\cdot\frac{\beta_t\sqrt{\bar{\alpha}_t}}{1-\bar{\alpha}_t}}_{d_t}\mathbf{x}_0
\end{align*}$$

现在用 $\alpha_t,\bar{\alpha}_{t-1},\bar{\alpha}_t$ 重新表示系数 $c_t$ 和 $d_t$。由于 $\bar{\alpha}_t=\alpha_t\bar{\alpha}_{t-1}$，有：

$$1-\bar{\alpha}_t
=1-\alpha_t\bar{\alpha}_{t-1}
=(1-\bar{\alpha}_{t-1})+\bar{\alpha}_{t-1}(1-\alpha_t)
=(1-\bar{\alpha}_{t-1})+\bar{\alpha}_{t-1}\beta_t$$

于是：

$$1-\frac{\beta_t}{1-\bar{\alpha}_t}
=\frac{1-\bar{\alpha}_t-\beta_t}{1-\bar{\alpha}_t}
=\frac{\alpha_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$

因此

$$c_t=\frac{\sqrt{\bar{\alpha}_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t},\quad
d_t=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}$$

这与我们之前通过配方法得到的闭式结果完全一致：

$$\hat{\boldsymbol{\mu}}_t^{(\text{DDPM})}(\mathbf{x}_t,\mathbf{x}_0)
=\frac{\sqrt{\bar{\alpha}_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
+\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0$$

DDIM定义了一组替代的反向条件分布，它保留了与DDPM完全相同的单步边缘分布 $q(\mathbf{x}_t\mid\mathbf{x}_0)$，同时允许对每一步的方差 $\sigma_t^2$ 进行控制：

$$q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)
=\mathcal{N}\left(
\mathbf{x}_{t-1};
\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0
+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\frac{\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}},
\sigma_t^2\mathbf{I}_d
\right) \tag{23}$$

$$q(\mathbf{x}_T\mid\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_T;\sqrt{\bar{\alpha}_T}\mathbf{x}_0,(1-\bar{\alpha}_T)\mathbf{I}_d)$$

式(23)中的均值是直线 $\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ 两个端点在几何意义下的凸组合：它保留了与 $\mathbf{x}_0$ 对齐的分量 $\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0$，并将正交的噪声分量 $(\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0)/\sqrt{1-\bar{\alpha}_t}$ 按因子 $\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}$ 缩放，以匹配时刻 $t-1$ 处的目标方差。

若取 $\sigma_t^2=\tilde{\beta}_t:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$，则

$$1-\bar{\alpha}_{t-1}-\sigma_t^2
=(1-\bar{\alpha}_{t-1})\left(1-\frac{\beta_t}{1-\bar{\alpha}_t}\right)
=(1-\bar{\alpha}_{t-1})\cdot\frac{\alpha_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
=\frac{\alpha_t(1-\bar{\alpha}_{t-1})^2}{1-\bar{\alpha}_t}$$

因此

$$\frac{\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}}{\sqrt{1-\bar{\alpha}_t}}
=\frac{\sqrt{\bar{\alpha}_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$

此时式(23)的均值变为

$$\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0
+\frac{\sqrt{\bar{\alpha}_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}(\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0)
=\frac{\sqrt{\bar{\alpha}_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
+\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0$$

与前面DDPM的后验均值完全一致。因此，DDPM是DDIM族中取 $\sigma_t^2=\tilde{\beta}_t$ 的特例。

**命题4.1（DDIM保留DDPM边缘分布）**
对式(23)中任意满足 $\sigma_t^2\in[0,1-\bar{\alpha}_{t-1}]$ 的取值，单步边缘分布始终为

$$q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I}_d),\quad \forall t \tag{24}$$

**证明**：我们对 $t$ 进行反向归纳。
- **基例**：当 $t=T$ 时，由式(23)第二行直接成立。
- **归纳假设**：假设命题在时刻 $t$ 成立，考虑时刻 $t-1$ 的边缘分布：

$$q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)
=\int q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)q(\mathbf{x}_t\mid\mathbf{x}_0)d\mathbf{x}_t$$

由于两项均为高斯分布，该积分仍是高斯分布。其均值为

$$\boldsymbol{\mu}_{t-1}
=\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}\left[
\mathbb{E}_{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}[\mathbf{x}_{t-1}]
\right]
=\mathbb{E}\left[
\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0
+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\frac{\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}}
\right]$$

由归纳假设，$\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}[\mathbf{x}_t]=\sqrt{\bar{\alpha}_t}\mathbf{x}_0$，因此第二项期望为0，最终得到 $\boldsymbol{\mu}_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0$。

对于协方差，使用全方差公式：

$$\operatorname{Var}[\mathbf{x}_{t-1}\mid\mathbf{x}_0]
=\underbrace{\mathbb{E}[\operatorname{Var}(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)]}_{\sigma_t^2\mathbf{I}_d}
+\underbrace{\operatorname{Var}(\mathbb{E}[\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0])}_{\left(\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\big/\sqrt{1-\bar{\alpha}_t}\right)^2\operatorname{Var}(\mathbf{x}_t\mid\mathbf{x}_0)}$$

由归纳假设，$\operatorname{Var}(\mathbf{x}_t\mid\mathbf{x}_0)=(1-\bar{\alpha}_t)\mathbf{I}_d$，因此

$$\operatorname{Var}[\mathbf{x}_{t-1}\mid\mathbf{x}_0]
=\sigma_t^2\mathbf{I}_d+(1-\bar{\alpha}_{t-1}-\sigma_t^2)\mathbf{I}_d
=(1-\bar{\alpha}_{t-1})\mathbf{I}_d$$

故 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)=\mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0,(1-\bar{\alpha}_{t-1})\mathbf{I}_d)$，归纳成立。证毕。

在采样阶段，我们用模型的反向核去近似 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$。式(23)清晰地表明，随机性由 $\sigma_t^2$ 控制。
- 若取确定性采样 $\sigma_t^2=0$，更新规则退化为仅含均值项；
- 若取随机采样 $\sigma_t^2>0$，则从对应高斯分布中采样。

当 $\sigma_t^2=\tilde{\beta}_t$ 时，完全退化为DDPM采样。

DDIM的一个核心实用优势是可以在简化的时间网格上快速采样。选取原时间步的一个子集，仅在这些时间步上应用式(23)，即可大幅减少采样步数，同时命题4.1保证边缘分布始终保持一致，因此整个简化后的链是自洽的。

### 4.2 DDGAN：对抗学习得到的反向动力学

本文提出了一种替代训练策略，该策略直接学习反向转移过程，从而使生成过程能够在极少的迭代次数内完成。我们保留"预备知识"部分中的前向过程与符号定义：$\beta_t$、$\alpha_t = 1 - \beta_t$、$\bar{\alpha}_t = \prod_{s \leq t}\alpha_s$，且

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, \, (1 - \bar{\alpha}_t)\mathbf{I}_d), \quad \mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\varepsilon}$$

对于每个 $t \in \{1, \dots, T\}$，我们考虑前向链下连续隐变量的分布，即

$$q_t^{\text{pair}}(\mathbf{x}_{t-1}, \mathbf{x}_t) := q(\mathbf{x}_t)q(\mathbf{x}_{t-1} \mid \mathbf{x}_t), \quad q(\mathbf{x}_t) = \int q(\mathbf{x}_t \mid \mathbf{x}_0)p_{\text{data}}(\mathbf{x}_0)d\mathbf{x}_0$$

带时间条件的生成器 $G_\theta$ 会生成反向步 $\hat{\mathbf{x}}_{t-1} = G_\theta(\mathbf{x}_t, t, \mathbf{z})$，其中 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$；而判别器 $D_\phi(\cdot, \cdot, t)$ 用于区分**真实样本对** $(\mathbf{x}_{t-1}, \mathbf{x}_t) \sim q_t^{\text{pair}}$ 与**生成样本对** $(\hat{\mathbf{x}}_{t-1}, \mathbf{x}_t)$。通常不同时间步 $t$ 会共享参数，并通过显式时间嵌入（time embeddings）实现对时间的条件编码。

**定义 4.2（离散时间下的对抗目标）** 一种简单的逻辑回归（logistic）损失函数用于训练生成器 $G_\theta$ 和判别器 $D_\phi$，目标函数为

$$\min_{\theta} \max_{\phi} \sum_{t=1}^{T}\bigg\{ \mathbb{E}_{(\mathbf{x}_{t-1}, \mathbf{x}_t) \sim q_t^{\text{pair}}}\big[\log D_\phi(\mathbf{x}_{t-1}, \mathbf{x}_t, t)\big] + \mathbb{E}_{\mathbf{x}_t \sim q(\mathbf{x}_t), \, \mathbf{z} \sim \mathcal{N}} \big[\log\big(1 - D_\phi(G_\theta(\mathbf{x}_t, t, \mathbf{z}), \mathbf{x}_t, t)\big)\big] \bigg\}$$

也可采用其他判别器损失函数，但对时间 $t$ 的条件编码方式保持不变。

**命题 4.3（学习到的反向转移的一致性）**
对任意 $t \in \{1, \dots, T\}$，在定义 4.2 的目标函数下，若时间步 $t$ 处的判别器达到最优状态，则由其诱导出的生成器更新会推动模型联合分布

$$p_t^\theta(\mathbf{x}_{t-1}, \mathbf{x}_t) := q(\mathbf{x}_t)p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$

向**真实联合分布**

$$q_t^{\text{pair}}(\mathbf{x}_{t-1}, \mathbf{x}_t) := q(\mathbf{x}_t)q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$

收敛。特别地，当学习到的反向转移 $p_t^\theta$ 与前向样本对分布 $q_t^{\text{pair}}$ 相等（即 $p_t^\theta = q_t^{\text{pair}}$）时，时间步 $t$ 处的博弈达到平稳点。

**证明**：在逻辑损失函数下，时间步 $t$ 处的最优判别器为

$$D_t^*(\mathbf{x}_{t-1}, \mathbf{x}_t) = \frac{q_t^{\text{pair}}(\mathbf{x}_{t-1}, \mathbf{x}_t)}{q_t^{\text{pair}}(\mathbf{x}_{t-1}, \mathbf{x}_t) + p_t^\theta(\mathbf{x}_{t-1}, \mathbf{x}_t)}$$

将 $D_t^*$ 代入单时间步目标函数可得

$$\mathcal{L}_t(\theta) = -\log 4 + \text{KL}\left(q_t^{\text{pair}} \bigg\Vert \frac{q_t^{\text{pair}} + p_t^\theta}{2}\right) + \text{KL}\left(p_t^\theta \bigg\Vert \frac{q_t^{\text{pair}} + p_t^\theta}{2}\right)$$

其中引入 $m = (q_t^{\text{pair}} + p_t^\theta)/2$ 并展开对数项即可得到该式。每个 KL 散度项均非负，且仅当两个输入分布相同时才取零值，因此 $\mathcal{L}_t(\theta) \geq -\log 4$，且等号成立当且仅当 $p_t^\theta = q_t^{\text{pair}}$。因此，生成器的更新会持续减小两个联合分布之间的差异，直至二者完全一致。

**采样与训练细节**
从 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ 开始采样，并仅执行少量迭代步骤 $\mathbf{x}_{t-1} \leftarrow G_\theta(\mathbf{x}_t, t, \mathbf{z})$ 即可完成生成。实际训练中，我们在**粗时间网格** $\{t_k\}_{k=0}^K$ 上进行训练（满足 $K \ll T$），由此学习到的反向映射支持**大步跳跃**（large jumps）；而在推理阶段，同样使用该粗网格。通常会输出相对于 $\mathbf{x}_t$ 的残差（或采用等价的重参数化方式），而非直接预测 $\hat{\mathbf{x}}_{t-1}$。对判别器 $D_\phi$ 施加适度的正则化可提升训练稳定性；同时可引入**小噪声一致性项**（small-noise consistency terms）以鼓励与前文的高斯反向均值保持一致，且这一操作不会改变对抗训练的本质。

**方法总结**
从实际的**wall-clock采样耗时**角度来看，该方法属于**加速策略**（通过减少采样步数实现加速）；其核心是**修改训练准则**，而非改变极大似然训练模型的更新规则。典型的引导（guidance）条件约束通常通过生成器 $G_\theta$ 的输入引入（若有需要，也可对判别器 $D_\phi$ 加入条件约束）。

### 4.3 嵌套扩散模型

嵌套扩散模型通过**组合多个层级化结构的扩散链**，对单一扩散过程进行扩展。每一层都会对前一层的表征进行**优化或增强**，这些层级共同构成了一个联合生成模型。本文针对从正向加噪到反向生成的全过程，展开了相关构建方法的研究。

设 $\mathbf{x}_0 \in \mathcal{X}$ 为一个服从数据分布 $p_{\text{data}}(\mathbf{x}_0)$ 的样本。单一扩散过程的定义如下：

$$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1}), \quad p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$

其中包含高斯正向步骤与经学习得到的反向模型。嵌套扩散模型引入了 **$K$ 个层级**，每个层级在表征空间中都拥有独立的扩散链，并通过**确定性变换**或**经学习得到的变换**将各层级连接起来。为便于说明，我们考虑如下映射路径：

$$(\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_T^1) \to (\mathbf{z}_0, \mathbf{z}_1, \dots, \mathbf{z}_T^2) \to \dots \to (\mathbf{u}_0, \mathbf{u}_1, \dots, \mathbf{u}_T^K)$$

其中 $\mathbf{x}_0$ 为原始数据，$\mathbf{u}_T^K$ 为噪声水平最高的深层状态；中间层级的链可通过**确定性映射**（例如下采样、编码器或其他变换）得到，例如：

$$\mathbf{z}_0 = f_1(\mathbf{x}_0), \quad \mathbf{u}_0 = f_2(\mathbf{z}_0), \quad \dots$$

对于第 $k \in \{1, \dots, K\}$ 个层级，我们将步骤 $t$ 处的状态记为 $\mathbf{y}_t^k$（因此 $\mathbf{y}_t^1 = \mathbf{x}_t$，$\mathbf{y}_t^2 = \mathbf{z}_t$ 等）。每个层级都拥有独立的前向过程：

$$q_k(\mathbf{y}_t^k \mid \mathbf{y}_{t-1}^k) = \mathcal{N}\left(\mathbf{y}_t^k; \sqrt{\alpha_t^{(k)}} \mathbf{y}_{t-1}^k, (1 - \alpha_t^{(k)}) \mathbf{I}\right), \quad \alpha_t^{(k)} = 1 - \beta_t^{(k)}, \quad \bar{\alpha}_t^{(k)} = \prod_{i=1}^t \alpha_i^{(k)}$$

层级间的嵌套连接通过**初始化规则**实现，例如：

$$\mathbf{y}_0^k = g_k(\mathbf{Y}^{k-1}) \quad \text{或} \quad \mathbf{y}_0^k = g_k(\mathbf{y}_0^{k-1})$$

其中 $\mathbf{Y}^{k-1} = (\mathbf{y}_0^{k-1}, \dots, \mathbf{y}_T^{k-1})$，$\mathbf{Y}^k = (\mathbf{y}_0^k, \dots, \mathbf{y}_T^k)$。

联合前向分布可进行如下因式分解：

$$q(\mathbf{Y}^1, \dots, \mathbf{Y}^K) = q_1(\mathbf{Y}^1) \prod_{k=2}^K q_k(\mathbf{Y}^k \mid \mathbf{Y}^{k-1})$$

其中

$$q_1(\mathbf{Y}^1) = p_{\text{data}}(\mathbf{x}_0) \prod_{t=1}^T q_1(\mathbf{x}_t \mid \mathbf{x}_{t-1}), \quad q_k(\mathbf{Y}^k \mid \mathbf{Y}^{k-1}) = \delta(\mathbf{y}_0^k - g_k(\mathbf{Y}^{k-1})) \prod_{t=1}^T q_k(\mathbf{y}_t^k \mid \mathbf{y}_{t-1}^k)$$

这里 $\delta(\cdot)$ 函数强制第 $k$ 层的初始化由第 $k-1$ 层决定。与单层级情况类似，每个层级都存在闭式边缘分布：

$$q_k(\mathbf{y}_t^k \mid \mathbf{y}_0^k) = \mathcal{N}\left(\mathbf{y}_t^k; \sqrt{\bar{\alpha}_t^{(k)}} \mathbf{y}_0^k, (1 - \bar{\alpha}_t^{(k)}) \mathbf{I}\right) \tag{25}$$

以及DDPM后验分布：

$$q_k(\mathbf{y}_{t-1}^k \mid \mathbf{y}_t^k, \mathbf{y}_0^k) = \mathcal{N}\left(\mathbf{y}_{t-1}^k; \frac{1}{\sqrt{\alpha_t^{(k)}}}\left(\mathbf{y}_t^k - \frac{\beta_t^{(k)}}{\sqrt{1 - \bar{\alpha}_t^{(k)}}} \boldsymbol{\epsilon}_t^{(k)}\right), \tilde{\beta}_t^{(k)} \mathbf{I}\right), \quad \tilde{\beta}_t^{(k)} := \frac{1 - \bar{\alpha}_{t-1}^{(k)}}{1 - \bar{\alpha}_t^{(k)}} \beta_t^{(k)} \tag{26}$$

嵌套反向模型支持跨层级的联合生成。一种常见的**由粗到细**（coarse-to-fine）因式分解方式为：

$$p_\theta(\mathbf{Y}^1, \dots, \mathbf{Y}^K) = p(\mathbf{y}_T^K) \prod_{t=1}^T p_\theta(\mathbf{y}_{t-1}^K \mid \mathbf{y}_t^K, K, t) \prod_{k=K-1}^1 \prod_{t=1}^T p_\theta(\mathbf{y}_{t-1}^k \mid \mathbf{y}_t^k, \mathbf{y}_0^{k+1}, k, t) \tag{27}$$

其中 $p(\mathbf{y}_T^K) = \mathcal{N}(\mathbf{0}, \mathbf{I})$。跨层级项 $p_\theta(\mathbf{y}_0^k \mid \mathbf{y}_0^{k+1}, k)$ 起到**上采样器/解码器**的作用（可以是确定性的或经学习得到的）。每一层级的反向转换均为高斯分布，其参数化方式与前文一致，并通过噪声预测实现：

$$p_\theta(\mathbf{y}_{t-1}^k \mid \mathbf{y}_t^k, \cdot) = \mathcal{N}\left(\mathbf{y}_{t-1}^k; \mu_\theta^{(k)}(\mathbf{y}_t^k, \cdot), \sigma_{t,k}^2 \mathbf{I}\right) \tag{28}$$

其中

$$\mu_\theta^{(k)}(\mathbf{y}_t^k, \cdot) = \frac{1}{\sqrt{\alpha_t^{(k)}}}\left(\mathbf{y}_t^k - \frac{\beta_t^{(k)}}{\sqrt{1 - \bar{\alpha}_t^{(k)}}} \hat{\boldsymbol{\epsilon}}_\theta^{(k)}(\mathbf{y}_t^k, t, k, \mathbf{y}_0^{k+1})\right)$$

若取 $\sigma_{t,k}^2 = \tilde{\beta}_t^{(k)}$ 则可恢复DDPM情况，也可选择其他参数（例如DDIM风格）以实现加速采样。

**典型生成路径**：
1. 采样 $\mathbf{y}_T^K \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$；在第 $K$ 层执行反向扩散以得到 $\mathbf{y}_0^K$。
2. 通过 $\mathbf{y}_0^{K-1} \sim p_\theta(\mathbf{y}_0^{K-1} \mid \mathbf{y}_0^K, K-1)$ 生成下一层级的条件（若为确定性过程，则 $\mathbf{y}_0^{K-1} = g_{K-1}^{-1}(\mathbf{y}_0^K)$）。
3. 对第 $K-1$ 层进行噪声初始化（$\mathbf{y}_T^{K-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$），并在以 $\mathbf{y}_0^K$ 为条件的情况下执行反向扩散，得到 $\mathbf{y}_0^{K-1}$。
4. 重复上述步骤直至第1层，最终获得 $\mathbf{y}_0^1 = \mathbf{x}_0$。

也可采用其他顺序（如由细到粗）；核心设计在于**确定性/经学习得到的跨层级映射**的放置位置（位于每条扩散链的起点或终点）。

训练过程通过最小化**负对数似然的变分上界**实现：

$$\mathcal{L} = \text{KL}\left(q(\mathbf{Y}^1, \dots, \mathbf{Y}^K) \parallel p_\theta(\mathbf{Y}^1, \dots, \mathbf{Y}^K)\right)$$

与单层级情况类似，该目标可分解为各层级上每步KL散度的和，从而产生实用的**均值平方噪声匹配项**：

$$\mathcal{L} \cong \sum_{k=1}^K \sum_{t=1}^T w_{t,k} \mathbb{E}\left[\left\|\boldsymbol{\epsilon}_t^{(k)} - \hat{\boldsymbol{\epsilon}}_\theta^{(k)}(\mathbf{y}_t^k, t, k, \text{cond}_k)\right\|_2^2\right], \quad w_{t,k} = \frac{\beta_t^{(k)}}{2(1 - \beta_t^{(k)})(1 - \bar{\alpha}_{t-1}^{(k)})} \tag{29}$$

其中 $\boldsymbol{\epsilon}_t^{(k)}$ 为注入第 $k$ 层的真实高斯噪声，$\text{cond}_k$ 表示任意跨层级条件（例如 $\mathbf{y}_0^{k+1}$）。权重 $w_{t,k}$ 与前文一致，用于确保跨时间步的**尺度一致性**。

一种直观的视角是：每个层级处理不同的**尺度**或**潜表征**。以两层级为例，设：

$$\mathbf{z}_0 = f(\mathbf{x}_0)$$

其中 $f$ 为确定性下采样器/编码器。在粗空间 $\mathbf{z}_0$ 上运行扩散过程 $q_2(\mathbf{z}_{1:T} \mid \mathbf{z}_0)$，进行反向扩散以得到 $\mathbf{z}_0$，随后通过解码器（确定性或经学习得到）将 $\mathbf{z}_0$ 映射回数据空间。

接着对 $\mathbf{x}$ 运行/条件化第二条扩散链：

$$q_1(\mathbf{x}_{1:T} \mid \mathbf{x}_0), \quad q_2(\mathbf{z}_{1:T} \mid \mathbf{z}_0)$$

其中跨层级连接为 $\mathbf{z}_0 = f(\mathbf{x}_0)$ 且 $p_\theta(\mathbf{x}_0 \mid \mathbf{z}_0)$。

整体联合前向分布仍可按前文方式进行因式分解，每个尺度的轨迹都建立在前一尺度的状态之上。

跨层级条件分布 $p_\theta(\mathbf{y}_0^k \mid \mathbf{y}_0^{k+1})$ 可以是**确定性的**（例如上采样）或**经学习得到的**（例如条件解码器）。在采样/推导得到 $\mathbf{y}_0^k$ 后，可采用以下两种方式之一：
- (i) 设 $\mathbf{y}_T^k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，并运行反向链 $p_\theta(\mathbf{y}_{t-1}^k \mid \mathbf{y}_t^k, \mathbf{y}_0^{k+1})$ 直至得到 $\mathbf{y}_0^k$；
- (ii) 在第 $k$ 层使用DDIM风格的**降步采样器**，并采用相同的累积调度 $\bar{\alpha}_t^{(k)}$ 以提升效率。

在所有情况下，参数化方式（公式28）结合累积项 $\bar{\alpha}_t^{(k)}$ 都能保证符号与机制与前文提出的**单层级扩散模型保持一致**。

### 4.4 Stable Diffusion

Stable Diffusion 是一种**潜扩散（latent diffusion）方法**：它不再直接在像素空间中运行扩散过程，而是先将数据映射到更低维度的潜空间中，在潜空间中执行去噪的计算成本更低；采样完成后，再通过解码器将最终的潜变量映射回数据域。

设 $\mathbf{x}_0 \in \mathbb{R}^{H \times W \times C}$ 是从数据分布 $p_{\text{data}}(\mathbf{x}_0)$ 中采样得到的样本。一个训练好的编码器-解码器结构 $(E, D)$，其输出维度满足 $(h, w, c) \ll (H, W, C)$，定义如下：

$$E: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{h \times w \times c}, \quad D: \mathbb{R}^{h \times w \times c} \to \mathbb{R}^{H \times W \times C}, \quad \mathbf{z}_0 := E(\mathbf{x}_0), \quad \tilde{\mathbf{x}}_0 := D(\mathbf{z}_0)$$

我们将 $\mathbf{z}_0$ 作为扩散过程的"数据"。将其展平后，潜变量位于 $\mathbb{R}^d$ 空间中，其中 $d = hwc$。沿用前文相同的调度符号，令 $\alpha_t := 1 - \beta_t$，$\bar{\alpha}_t := \prod_{i=1}^t \alpha_i$，则潜空间的前向链为线性高斯分布：

$$q(\mathbf{z}_{1:T} \mid \mathbf{z}_0) = \prod_{t=1}^T q(\mathbf{z}_t \mid \mathbf{z}_{t-1}), \quad q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) = \mathcal{N}\left(\mathbf{z}_t; \sqrt{\alpha_t} \mathbf{z}_{t-1}, \beta_t \mathbf{I}_d\right) \tag{30}$$

由此可推导出闭式边缘分布及其重参数化形式：

$$q(\mathbf{z}_t \mid \mathbf{z}_0) = \mathcal{N}\left(\mathbf{z}_t; \sqrt{\bar{\alpha}_t} \mathbf{z}_0, (1 - \bar{\alpha}_t) \mathbf{I}_d\right), \quad \mathbf{z}_t = \sqrt{\bar{\alpha}_t} \mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d) \tag{31}$$

由于式(30)是线性高斯分布，因此精确的单步后验分布与像素空间的DDPM后验分布完全一致，仅作用域变为潜空间：

$$q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{z}_0) = \mathcal{N}\left(\mathbf{z}_{t-1}; \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t\right), \tilde{\beta}_t \mathbf{I}_d\right), \quad \tilde{\beta}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t, \quad \boldsymbol{\epsilon}_t = \frac{\mathbf{z}_t - \sqrt{\bar{\alpha}_t} \mathbf{z}_0}{\sqrt{1 - \bar{\alpha}_t}} \tag{32}$$

生成过程从终端噪声水平的高斯先验开始采样，通过学习得到的均值（基于噪声预测）和选定的每步方差，反向执行该链：

$$p_\theta(\mathbf{z}_{0:T}) = p(\mathbf{z}_T) \prod_{t=1}^T p_\theta(\mathbf{z}_{t-1} \mid \mathbf{z}_t), \quad p(\mathbf{z}_T) = \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$

$$p_\theta(\mathbf{z}_{t-1} \mid \mathbf{z}_t) = \mathcal{N}\left(\mathbf{z}_{t-1}; \underbrace{\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t, t)\right)}_{\mu_\theta(\mathbf{z}_t, t)}, \sigma_t^2 \mathbf{I}_d\right), \quad \sigma_t^2 \in \{\tilde{\beta}_t, 0\} \ (\text{DDPM 或 DDIM 采样}) \tag{33}$$

这正是前文参数化形式在将 $\mathbf{x} \mapsto \mathbf{z}$ 替换后的结果。如果使用条件生成（例如文本条件），只需将 $\hat{\boldsymbol{\epsilon}}_\theta$ 扩展为 $\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t, t, \cdot)$ 即可；相关机制将在后续第6节详细展开。

潜空间的训练遵循相同的ELBO/KL分解，结合式(31)的前向重参数化，得到加权噪声匹配损失：

$$\mathcal{L}_{\text{LDM}} \cong \mathbb{E}_{\mathbf{z}_0, t, \boldsymbol{\epsilon}} \left[ w_t \left\| \boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t, t) \right\|_2^2 \right], \quad \mathbf{z}_t = \sqrt{\bar{\alpha}_t} \mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad w_t = \frac{\beta_t}{2(1 - \beta_t)(1 - \bar{\alpha}_{t-1})} \tag{34}$$

任何不改变贝叶斯最优预测器的等价加权方式都是可接受的；上述形式是为了与前文像素空间的推导保持一致。

反向过程得到 $\hat{\mathbf{z}}_0$ 后，解码器将其重构为数据空间的输出：

$$\hat{\mathbf{x}}_0 = D(\hat{\mathbf{z}}_0) \tag{35}$$

在潜空间中运行，将计算集中在由 $(E, D)$ 学习到的语义结构化特征上，在完整保留扩散数学原理（前向边缘分布、真实后验、反向参数化）的同时，大幅降低了采样成本。

**核心原理**：
- 编码器 $E$ 把高维图像压缩到低维潜空间，大幅降低计算量
- 扩散模型仅在潜空间训练和采样，计算成本仅为像素空间DDPM的约1/8（以4倍下采样为例）
- 解码器 $D$ 把采样后的潜变量还原为高清图像，同时保留生成质量

**关键符号对照表**：

| 符号 | 含义 |
|------|------|
| $\mathbf{x}_0$ | 原始高维图像（像素空间） |
| $\mathbf{z}_0$ | 编码器输出的低维潜变量 |
| $E/D$ | 预训练的编码器/解码器 |
| $\hat{\boldsymbol{\epsilon}}_\theta$ | 噪声预测UNet模型 |
| $\sigma_t^2$ | 采样方差，$\sigma_t^2=\tilde{\beta}_t$ 对应DDPM，$\sigma_t^2=0$ 对应DDIM |
| $\mathcal{L}_{\text{LDM}}$ | 潜扩散模型的训练损失 |

**与传统DDPM的对比**：

| 特性 | 像素空间DDPM | Stable Diffusion（潜扩散） |
|------|--------------|---------------------------|
| 计算空间 | 原始像素（高维） | 压缩后的潜空间（低维） |
| 采样速度 | 慢（1000步，计算量大） | 快（50步，计算量小） |
| 显存占用 | 极高（需处理全分辨率图像） | 大幅降低（仅处理潜变量） |
| 生成质量 | 高 | 相当，同时支持高清生成 |
| 条件扩展 | 困难 | 易扩展（文本、图像等条件） |

---

## 5 流匹配（Flow Matching）

流匹配将生成建模重新表述为：学习一个**时间依赖的速度场**，使其**确定性地**将概率质量从一个平凡的源分布输运到数据分布。

在讨论扩散模型时，我们规定了一族中间噪声边际 $\{p_t\}_{t \in [0,T]}$（通过 $\alpha_t$ 和 $\bar{\alpha}_t$ 确定离散调度），并通过逆转随机前向过程来采样。流匹配则寻求一个速度场 $\mathbf{v}(\mathbf{x}, t)$，使其常微分方程（ODE）

$$\dot{\mathbf{X}}_t = \mathbf{v}(\mathbf{X}_t, t)$$

与扩散构造所定义的时间边际 $\{p_t\}$ 完全相同 [5]。换言之，我们寻找一种确定性输运，使其在每个时刻 $t$ 的边际分布与随机扩散在边际上无法区分。

**流匹配的两大优势：**
1. 用确定性 ODE 积分取代逐步随机模拟，消除轨迹噪声，并允许使用标准高阶 ODE 求解器。
2. 一旦速度场 $\mathbf{v}$ 被指定，就可以直接训练它来复现指定的边际 $p_{t \in [0,T]}$，而无需在训练过程中委身于某个特定的随机前向机制。

为使论述精确，我们借助两个标准分析工具：**连续性方程**（Liouville 方程，编码了概率在时变速度场下的守恒性，描述上述 ODE 的边际）和**Fokker-Planck 方程**（描述前向扩散的边际演化）。通过将 Fokker-Planck 方程中的扩散二阶（拉普拉斯）项改写为由得分 $s_t = \nabla_\mathbf{x} \log p_t$ 驱动的保守输运项，我们推导出**概率流 ODE**，其速度为

$$\mathbf{v}(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \tfrac{1}{2}g(t)^2\, s_t(\mathbf{x}),$$

该速度产生的边际与 SDE $\mathrm{d}\mathbf{X}_t = \mathbf{f}(\mathbf{X}_t, t)\,\mathrm{d}t + g(t)\,\mathrm{d}\mathbf{W}_t$ 相同。在前面使用的方差保持（VP）情形下，当真实得分被其学得的近似所替代时，这正好还原为熟悉的 DDIM 形式。

---

### 5.1 概率流 ODE 与连续性方程

时间索引密度族 $(p_t)_{t \in [0,T]}$ 与速度场 $\mathbf{v}(\cdot, t)$ 通过**连续性方程**相联系：

$$\partial_t p_t(\mathbf{x}) = -\nabla_\mathbf{x} \cdot \big(p_t(\mathbf{x})\,\mathbf{v}(\mathbf{x}, t)\big),$$

它表达了沿 $\mathbf{v}$ 所生成的流的概率守恒。第一步是将确定性 ODE 轨迹与这个偏微分方程联系起来 [12]。

**定义 5.1**（连续性方程，Liouville 方程）。若 $(p_t, \mathbf{v})$ 在 $[0,T]$ 上满足

$$\partial_t p_t = -\nabla_\mathbf{x} \cdot (p_t \mathbf{v}), \quad \text{且} \quad p_{t=0} = p_0,$$

则称其满足连续性方程。

**引理 5.2**（ODE 流的 Liouville 输运）。设 $\dot{\mathbf{X}}_t = \mathbf{v}(\mathbf{X}_t, t)$，$\mathbf{X}_0 \sim p_0$，其中 $\mathbf{v}$ 关于 $\mathbf{x}$ 局部 Lipschitz、关于 $t$ 可测，且解在 $[0,T]$ 上唯一存在且不爆破。则 $\mathbf{X}_t$ 的分布律 $p_t$ 满足以速度 $\mathbf{v}$ 的连续性方程。

> **证明概要**：对任意光滑紧支撑函数 $\varphi$，由链式法则 $\frac{\mathrm{d}}{\mathrm{d}t}\varphi(\mathbf{X}_t) = \nabla\varphi(\mathbf{X}_t)^\top \mathbf{v}(\mathbf{X}_t, t)$。取期望并利用 $\mathbf{X}_t$ 的分布律，再对 $\mathbf{x}$ 分部积分（边界项因紧支撑消失），即得 $\partial_t p_t = -\nabla_\mathbf{x} \cdot (p_t \mathbf{v})$。

接下来，我们回顾扩散模型中前向 SDE 的 Fokker-Planck 方程，并展示其与连续性方程的联系。

**引理 5.3**（各向同性、状态无关扩散的 Fokker-Planck 方程）。设 $\mathrm{d}\mathbf{X}_t = \mathbf{f}(\mathbf{X}_t, t)\,\mathrm{d}t + g(t)\,\mathrm{d}\mathbf{W}_t$，其中 $\mathbf{f}$ 可测且线性增长，$g$ 连续，$\mathbf{X}_t$ 有光滑密度 $p_t$。则

$$\partial_t p_t(\mathbf{x}) = -\nabla_\mathbf{x} \cdot \big(\mathbf{f}(\mathbf{x}, t)\,p_t(\mathbf{x})\big) + \frac{g(t)^2}{2}\Delta_\mathbf{x} p_t(\mathbf{x}).$$

> **证明概要**：对光滑紧支撑 $\varphi$ 应用 Itô 公式，取期望（鞅项均值为零），两次分部积分后得出结论。

**定理 5.4**（各向同性、状态无关扩散的概率流 ODE）。定义

$$\mathbf{v}(\mathbf{x}, t) := \mathbf{f}(\mathbf{x}, t) - \frac{g(t)^2}{2}s_t(\mathbf{x}) = \mathbf{f}(\mathbf{x}, t) - \frac{g(t)^2}{2}\nabla_\mathbf{x}\log p_t(\mathbf{x}).$$

设 $\dot{\mathbf{Y}}_t = \mathbf{v}(\mathbf{Y}_t, t)$，$\mathbf{Y}_0 \sim p_0$。则 $\mathbf{Y}_t$ 的时间边际与 SDE 完全相同，即对所有 $t \in [0, T]$ 有 $\mathbf{Y}_t \sim p_t$。

> **证明关键步骤**：由引理 5.3，$\partial_t p_t = -\nabla \cdot (\mathbf{f}p_t) + \frac{g^2}{2}\Delta p_t$。利用恒等式 $\Delta p_t = \nabla \cdot (p_t \nabla\log p_t)$，可得 $\partial_t p_t = -\nabla \cdot (p_t \mathbf{v})$，即 $p_t$ 满足以 $\mathbf{v}$ 为速度的连续性方程。由引理 5.2，ODE 解的分布律也满足同一连续性方程，故两者边际一致。

**特例（方差保持 VP 过程）**：在 VP 前向过程 $\mathrm{d}\mathbf{X}_t = -\frac{1}{2}\beta(t)\mathbf{X}_t\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}\mathbf{W}_t$ 中，$\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}$，$g(t)^2 = \beta(t)$，定理 5.4 给出：

$$\dot{\mathbf{x}} = -\tfrac{1}{2}\beta(t)\mathbf{x} - \tfrac{1}{2}\beta(t)\nabla_\mathbf{x}\log p_t(\mathbf{x}),$$

当得分被其学得的近似替代后，这正是 DDIM 的概率流 ODE。

**注 5.5**（范围与推广）：上述论证假设状态无关的各向同性扩散 $g(t)\mathbf{I}_d$，这是本教程全篇采用的情形。对于状态相关的扩散矩阵，Fokker-Planck 算子变为 $\partial_t p_t = -\nabla \cdot (\mathbf{f}p_t) + \frac{1}{2}\nabla \cdot (\mathbf{D}\nabla p_t)$，对应的概率流速度需要额外的 $-\frac{1}{2}\mathbf{D}\nabla\log p_t$ 以外的项，适定性也需要额外正则性。实践中标准 VP/VE 公式规避了这些复杂性。

---

### 5.2 流匹配目标：边际 vs. 条件

定理 5.4 的概率流 ODE 表明，时变速度场 $\mathbf{v}(\mathbf{x}, t)$ 通过连续性方程唯一确定了边际的演化。**流匹配**将此转化为监督学习问题：构造一族中间分布 $\{p_t\}_{t \in [0,T]}$ 以及输运 $p_t$ 的目标速度场 $\mathbf{u}_t(\mathbf{x})$，训练模型 $\mathbf{v}_\theta(\mathbf{x}, t)$ 回归 $\mathbf{u}_t(\mathbf{x})$。这避免了显式的得分估计；它只需要来自 $p_0$（数据）和简单终端分布 $p_T$（如 $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$）的样本，以及端点之间可微的路径插值。

**定义 5.6**（随机插值与耦合）。设 $p_0$ 和 $p_T$ 是 $\mathbb{R}^d$ 上的概率密度。**耦合**是边际分别为 $p_0$ 和 $p_T$ 的联合分布 $\pi(\mathbf{x}_0, \mathbf{x}_T)$。**随机插值**是可测映射

$$\psi: \mathbb{R}^d \times \mathbb{R}^d \times [0,T] \to \mathbb{R}^d, \quad (\mathbf{x}_0, \mathbf{x}_T, t) \mapsto \psi_t(\mathbf{x}_0, \mathbf{x}_T),$$

满足 $\psi_0(\mathbf{x}_0, \mathbf{x}_T) = \mathbf{x}_0$，$\psi_T(\mathbf{x}_0, \mathbf{x}_T) = \mathbf{x}_T$，且 $t \mapsto \psi_t(\mathbf{x}_0, \mathbf{x}_T)$ 连续可微。

设 $\mathbf{X}_0 \sim p_0$，$\mathbf{X}_T \sim p_T$，$(\mathbf{X}_0, \mathbf{X}_T) \sim \pi$，$\mathbf{X}_t := \psi_t(\mathbf{X}_0, \mathbf{X}_T)$，$p_t = \text{law}(\mathbf{X}_t)$，将条件（逐路径）速度记作 $\dot{\psi}_t(\mathbf{x}_0, \mathbf{x}_T) := \partial_t \psi_t(\mathbf{x}_0, \mathbf{x}_T)$。

**定义 5.7**（边际目标速度）。对 $t \in [0,T]$，定义

$$\mathbf{u}_t(\mathbf{x}) := \mathbb{E}\!\left[\dot{\psi}_t(\mathbf{X}_0, \mathbf{X}_T) \;\Big|\; \mathbf{X}_t = \mathbf{x}\right].$$

**命题 5.8**（插值连续性方程）。在定义 5.6 的正则性条件下，边际 $p_t$ 满足

$$\partial_t p_t(\mathbf{x}) = -\nabla_\mathbf{x} \cdot \big(p_t(\mathbf{x})\,\mathbf{u}_t(\mathbf{x})\big), \quad t \in [0,T].$$

因此，以 $\mathbf{Y}_0 \sim p_0$ 为初值的 ODE $\dot{\mathbf{Y}}_t = \mathbf{u}_t(\mathbf{Y}_t)$ 对所有 $t$ 满足 $\text{law}(\mathbf{Y}_t) = p_t$。

两种自然的回归目标：

- **条件流匹配（CFM）**目标在逐路径目标上训练：

$$\mathcal{L}_{\text{CFM}}(\theta) := \mathbb{E}\!\left[\left\|\mathbf{v}_\theta(\mathbf{Z}_t, t) - \mathbf{U}_t\right\|_2^2\right], \quad \mathbf{Z}_t := \psi_t(\mathbf{X}_0, \mathbf{X}_T),\quad \mathbf{U}_t := \dot{\psi}_t(\mathbf{X}_0, \mathbf{X}_T).$$

- **边际流匹配（MFM）**目标在边际速度 $\mathbf{u}_t$ 上直接回归：

$$\mathcal{L}_{\text{MFM}}(\theta) := \mathbb{E}\!\left[\left\|\mathbf{v}_\theta(\mathbf{X}_t, t) - \mathbf{u}_t(\mathbf{X}_t)\right\|_2^2\right], \quad \mathbf{X}_t \sim p_t.$$

**定理 5.9**（最优预测器与 CFM/MFM 等价性）。假设 $\mathbb{E}\|\mathbf{U}_t\|_2^2 < \infty$ 且对所有 $\theta$ 有 $\mathbb{E}\|\mathbf{v}_\theta(\mathbf{Z}_t, t)\|_2^2 < \infty$。则：

1. $\mathcal{L}_{\text{CFM}}$ 在可测 $\mathbf{v}$ 上的唯一 $L^2$ 最小化子为 $\mathbf{v}^*(\mathbf{x}, t) \equiv \mathbf{u}_t(\mathbf{x}) = \mathbb{E}[\mathbf{U}_t \mid \mathbf{Z}_t = \mathbf{x}]$。
2. 存在与 $\theta$ 无关的常数 $C$ 使得 $\mathcal{L}_{\text{CFM}}(\theta) = \mathcal{L}_{\text{MFM}}(\theta) + C$，即 CFM 和 MFM 共享同一组全局最小化子。

> **证明关键**：利用 $L^2$ 正交性原理以及全方差公式 $\mathbb{E}\|\mathbf{v}_\theta(Y) - U\|_2^2 = \mathbb{E}\|\mathbf{v}_\theta(Y) - \mathbb{E}[U|Y]\|_2^2 + \mathbb{E}\|U - \mathbb{E}[U|Y]\|_2^2$，其中第二项为常数。

**CFM 实践吸引力**：对于常见插值，解析形式的 $\mathbf{u}_t(\mathbf{x})$ 涉及难解的后验 $\pi(\mathbf{x}_0, \mathbf{x}_T \mid \mathbf{X}_t = \mathbf{x})$，而 $\dot{\psi}_t(\mathbf{X}_0, \mathbf{X}_T)$ 可直接从采样端点观察到。

对于直线路径 $\psi_t = (1-\rho(t))\mathbf{X}_0 + \rho(t)\mathbf{X}_T$，逐路径 CFM 目标为

$$\mathbf{U}_t = \dot{\psi}_t(\mathbf{X}_0, \mathbf{X}_T) = \dot{\rho}(t)(\mathbf{X}_T - \mathbf{X}_0),$$

因此 CFM 只需利用 $(\mathbf{X}_0, \mathbf{X}_T)$ 的样本来训练 $\mathbf{v}_\theta(\psi_t(\mathbf{X}_0, \mathbf{X}_T), t)$ 匹配 $\dot{\rho}(t)(\mathbf{X}_T - \mathbf{X}_0)$。

---

### 5.3 线性与整流流（直线耦合）

本节将流匹配框架特化到数据与高斯噪声之间的**直线插值**。我们通过缩放的 Tweedie 恒等式，以得分 $\nabla_\mathbf{x}\log p_t(\mathbf{x})$ 来给出封闭形式的边际速度，并分离出一个仅依赖时间的因子，由此构造出数值性质更好的**整流**（rectified）速度场。

设 $\mathbf{X}_0 \sim p_0$ 独立于 $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，$\rho: [0,T] \to [0,1]$ 是 $C^1$ 函数且 $\rho(0) = 0$，$\rho(T) = 1$，以及

$$\mathbf{X}_t = (1-\rho(t))\mathbf{X}_0 + \rho(t)\mathbf{Z}, \quad p_t = \text{law}(\mathbf{X}_t).$$

条件流匹配（CFM）目标 [1]（定义 5.6）为 $\dot{\psi}_t(\mathbf{X}_0, \mathbf{Z}) = \dot{\rho}(t)(\mathbf{Z} - \mathbf{X}_0)$。

**引理 5.10**（缩放 Tweedie 恒等式）。设 $\mathbf{Y} = \mathbf{U} + \sigma\mathbf{Z}$，其中 $\mathbf{U} \in \mathbb{R}^d$ 独立于 $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，$\sigma > 0$，设 $p_\mathbf{Y}$ 为 $\mathbf{Y}$ 的密度。则在 $p_\mathbf{Y}$ 可微的所有点 $\mathbf{y}$ 处：

$$\mathbb{E}[\mathbf{U} \mid \mathbf{Y} = \mathbf{y}] = \mathbf{y} + \sigma^2 \nabla_\mathbf{y}\log p_\mathbf{Y}(\mathbf{y}), \quad \mathbb{E}[\mathbf{Z} \mid \mathbf{Y} = \mathbf{y}] = -\sigma\nabla_\mathbf{y}\log p_\mathbf{Y}(\mathbf{y}).$$

**命题 5.11**（直线边际速度）。在 $\mathbf{X}_t = (1-\rho)\mathbf{X}_0 + \rho\mathbf{Z}$ 且 $\dot{\psi}_t = \dot{\rho}(\mathbf{Z} - \mathbf{X}_0)$ 的条件下：

$$\mathbf{u}_t(\mathbf{x}) = \mathbb{E}[\dot{\psi}_t \mid \mathbf{X}_t = \mathbf{x}] = -\frac{\dot{\rho}(t)}{1-\rho(t)}\!\left(\mathbf{x} + \rho(t)\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\right).$$

> **证明**：令 $\mathbf{U} = (1-\rho)\mathbf{X}_0$，$\sigma = \rho$，$\mathbf{Y} = \mathbf{X}_t = \mathbf{U} + \sigma\mathbf{Z}$。由引理 5.10：
> $\mathbb{E}[\mathbf{X}_0 \mid \mathbf{X}_t = \mathbf{x}] = \frac{1}{1-\rho}(\mathbf{x} + \rho^2\nabla_\mathbf{x}\log p_t(\mathbf{x}))$，以及 $\mathbb{E}[\mathbf{Z} \mid \mathbf{X}_t = \mathbf{x}] = -\rho\nabla_\mathbf{x}\log p_t(\mathbf{x})$，代入后化简得证。

命题 5.11 表明直线流匹配产生的速度具有如下形式：

$$\mathbf{u}_t(\mathbf{x}) = -\kappa(t)\!\left(\mathbf{x} + \rho(t)\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\right), \quad \kappa(t) := \frac{\dot{\rho}(t)}{1-\rho(t)}.$$

标量因子 $\kappa(t)$ 仅依赖于时间，这揭示了一种**时间重参数化自由度**：将速度场乘以 $t$ 的严格正函数，在经过适当的时间变量替换后，状态空间中的轨迹保持不变。

**引理 5.12**（轨迹的时变不变性）。设 $\kappa: [0,T] \to (0,\infty)$ 为 $C^1$ 函数，$\mathbf{w}(\mathbf{x},t)$ 关于 $\mathbf{x}$ 可测且局部 Lipschitz。考虑 $\dot{\mathbf{X}}_t = \kappa(t)\mathbf{w}(\mathbf{X}_t, t)$，$\mathbf{X}_0 = \mathbf{x}_0$。定义严格递增 $C^1$ 映射 $s(t) := \int_0^t \kappa(\tau)\,\mathrm{d}\tau$ 及其逆 $t(s)$。则 $\mathbf{Y}_s := \mathbf{X}_{t(s)}$ 满足 $\frac{\mathrm{d}}{\mathrm{d}s}\mathbf{Y}_s = \mathbf{w}(\mathbf{Y}_s, t(s))$，且状态空间像集相同。

**定义 5.13**（整流直线速度）。对直线插值，定义：

$$\tilde{\mathbf{u}}_t(\mathbf{x}) := -\!\left(\mathbf{x} + \rho(t)\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\right).$$

**推论 5.14**（到时间重参数化的等价性）。$\mathbf{X}_t$ 与 $\mathbf{Y}_s$ 的状态空间轨迹集合一致：$\{\mathbf{X}_t : t \in [0,T]\} = \{\mathbf{Y}_s : s \in [0, s(T)]\}$。

**两个实践意义**：
1. 可以任意选择单调的 $\rho$，而不改变状态空间路径集合；差异只是沿路径的遍历速度。
2. 整流场 $\tilde{\mathbf{u}}_t$ 消除了在 $\rho \uparrow 1$ 附近出现的奇异放大因子 $\kappa(t) = \dot{\rho}/(1-\rho)$（对线性调度），使 ODE 求解器的步长更加均匀。

将 DDPM 参数化代入 $\mathbf{X}_t = \sqrt{\bar{\alpha}_t}\mathbf{X}_0 + \sqrt{1-\bar{\alpha}_t}\mathbf{Z}$（对应 $\rho(t) = \sqrt{1-\bar{\alpha}_t}$），由命题 5.11 得：

$$\mathbf{u}_t(\mathbf{x}) = -\frac{\dot{\bar{\alpha}}_t}{2\bar{\alpha}_t}\!\left(\mathbf{x} + \sqrt{1-\bar{\alpha}_t}\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\right), \quad \tilde{\mathbf{u}}_t(\mathbf{x}) = -\!\left(\mathbf{x} + \sqrt{1-\bar{\alpha}_t}\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\right).$$

与 DDIM 的精确等价性在下一节建立。

---

### 5.4 DDIM 即流匹配（时间重参数化）

我们现在将直线（以及整流）流匹配构造与 DDIM 联系起来。关键事实：
- (i) 命题 5.11 的直线速度 $\mathbf{u}_t$ 与 VP 概率流 ODE（定理 5.4）输运相同边际；
- (ii) 乘以正的纯时间因子保持状态空间轨迹不变（引理 5.12）；
- (iii) 在 DDPM 参数化下，条件恒等式（缩放 Tweedie；引理 5.10）通过 $\nabla_\mathbf{x}\log p_t(\mathbf{x}) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{X}_t = \mathbf{x}]$ 给出速度的 $\boldsymbol{\epsilon}$-形式。

**命题 5.15**（到时间重参数化的轨迹等价性）。设 $p_t$ 是 VP 边际，$\mathcal{L}(\mathbf{X}_t) = \sqrt{\bar{\alpha}_t}\mathbf{X}_0 + \sqrt{1-\bar{\alpha}_t}\mathbf{Z}$，分别设直线速度及其整流版本为

$$\mathbf{u}_t(\mathbf{x}) = -\frac{\dot{\bar{\alpha}}_t}{2\bar{\alpha}_t}\!\left(\mathbf{x} + \sqrt{1-\bar{\alpha}_t}\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\right), \quad \tilde{\mathbf{u}}_t(\mathbf{x}) = -\!\left(\mathbf{x} + \sqrt{1-\bar{\alpha}_t}\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\right),$$

VP 概率流速度为 $\mathbf{v}_{\text{PF}}(\mathbf{x},t) = -\frac{\beta(t)}{2}\big(\mathbf{x} + \nabla_\mathbf{x}\log p_t(\mathbf{x})\big)$。则存在严格递增 $C^1$ 时间重参数化 $s_1(t)$ 和 $s_2(t)$，使这三个 ODE 具有相同的状态空间轨迹集合，且每个 ODE 都输运边际 $p_t$。

**整流场的 $\boldsymbol{\epsilon}$-形式**：将 $\nabla_\mathbf{x}\log p_t(\mathbf{x}) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{X}_t = \mathbf{x}]$ 代入 $\tilde{\mathbf{u}}_t$，得：

$$\tilde{\mathbf{u}}_t(\mathbf{x}) = -\!\left(\mathbf{x} - \mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{X}_t = \mathbf{x}]\right).$$

将条件期望替换为学得的预测器 $\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, t)$，得到实用速度 $\tilde{\mathbf{u}}_t^\theta(\mathbf{x}) = -(\mathbf{x} - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, t))$。

**定理 5.16**（DDIM 更新即整流流的单步积分）。固定递减时间网格 $T = \tau_0 > \tau_1 > \cdots > \tau_K = 0$，设 $\mathbf{x}_{\tau_k}$ 在直线特征线 $\mathbf{x}_{\tau_k} = \sqrt{\bar{\alpha}_{\tau_k}}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{\tau_k}}\boldsymbol{\epsilon}$ 上（对某些时不变端点 $(\mathbf{x}_0, \boldsymbol{\epsilon})$）。设 $\hat{\boldsymbol{\epsilon}}_\theta$ 是 $\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{X}_{\tau_k} = \mathbf{x}_{\tau_k}]$ 的估计量，定义

$$\hat{\mathbf{x}}_0(\mathbf{x}_{\tau_k}, \tau_k) := \frac{\mathbf{x}_{\tau_k} - \sqrt{1-\bar{\alpha}_{\tau_k}}\,\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_{\tau_k}, \tau_k)}{\sqrt{\bar{\alpha}_{\tau_k}}}.$$

则同一特征线上 $\tau_{k-1}$ 时刻的唯一点由 **DDIM 更新**给出：

$$\mathbf{x}_{\tau_{k-1}} = \sqrt{\bar{\alpha}_{\tau_{k-1}}}\,\hat{\mathbf{x}}_0(\mathbf{x}_{\tau_k}, \tau_k) + \sqrt{1-\bar{\alpha}_{\tau_{k-1}}}\,\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_{\tau_k}, \tau_k). \tag{36}$$

**推论 5.17**（与 VP 概率流 DDIM ($\eta=0$) 的一致性）。由 (36) 生成的 $\mathbf{x}_{\tau_{k-1}}$ 与通过在 $\bar{\alpha}$-时间上离散化 VP 概率流 ODE（定理 5.4）所得的确定性 DDIM 采样器（$\eta = 0$）完全一致，此时得分通过 $\nabla_\mathbf{x}\log p_t(\mathbf{x}) = -\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x},t)/\sqrt{1-\bar{\alpha}_t}$ 参数化。

**实现注意事项**：当使用祖先式（随机）更新（方差 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$，见第 4.1 节）时，均值使用修正因子 $\beta_t/\sqrt{1-\bar{\alpha}_t}$：

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)\right),$$

而 DDIM 对应于将逐步随机性设为零并在选定的若干时刻上使用端点公式 (36)。

---

### 5.5 离散化与实践训练注记

连续时间流匹配公式在实践中导致两种离散化：**训练离散化**（随机采样 $t$ 并回归速度目标）和**采样离散化**（沿有限网格积分 ODE）。本节记录了为常见启发式方法提供依据的不变量、速度目标与 $\boldsymbol{\epsilon}$-预测损失的联系，以及在温和正则性条件下数值积分的标准误差保证。

**引理 5.18**（回归最优值的时间加权不变性）。设 $w: [0,T] \to (0,\infty)$ 可积。对 CFM 或 MFM，考虑加权损失

$$\mathcal{L}_w(\theta) = \mathbb{E}\!\left[w(t)\,\left\|\mathbf{v}_\theta(\mathbf{X}_t, t) - \mathbf{U}_t^*\right\|_2^2\right],$$

其中 $\mathbf{U}_t^*$ 为目标（CFM 的 $\dot{\psi}_t(\mathbf{X}_0, \mathbf{X}_T)$ 或 MFM 的 $\mathbf{u}_t(\mathbf{X}_t)$）。则 $\mathcal{L}_w$ 在可测 $\mathbf{v}$ 上的唯一 $L^2$ 最小化子与无权损失相同，均为 $\mathbf{v}^*(\mathbf{x}, t) = \mathbb{E}[\mathbf{U}_t^* \mid \mathbf{X}_t = \mathbf{x}]$（几乎处处）。

> **意义**：非均匀时间采样和按时间重缩放作为预处理器，不改变最优预测器；它们影响优化速度和方差，而不影响一致性。

**命题 5.19**（$\boldsymbol{\epsilon}$-预测与整流速度的等价性）。在直线 (VP) 参数化 $\mathbf{X}_t = \sqrt{\bar{\alpha}_t}\mathbf{X}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ 下，设 $\boldsymbol{\epsilon}^*(\mathbf{x}, t) := \mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{X}_t = \mathbf{x}]$。贝叶斯最优整流速度（定义 5.13）满足：

$$\tilde{\mathbf{u}}_t(\mathbf{x}) = -\!\left(\mathbf{x} - \boldsymbol{\epsilon}^*(\mathbf{x}, t)\right).$$

因此，标准 $\boldsymbol{\epsilon}$-损失 $\mathbb{E}\|\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{X}_t, t) - \boldsymbol{\epsilon}\|_2^2$ 的最小化子通过可逆线性映射 $\tilde{\mathbf{u}}_t^\theta(\mathbf{x}) = -(\mathbf{x} - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, t))$ 诱导整流速度损失的最小化子。

> **意义**：训练 $\hat{\boldsymbol{\epsilon}}_\theta$ 等价于（到显式线性变换）学习驱动流的整流速度。

**引理 5.20**（一阶单步方法的全局误差）。设 $\mathbf{f}(\mathbf{x}, s)$ 在 $\mathbf{x}$ 上关于 $s \in [s_0, s_1]$ 一致局部 Lipschitz，Lipschitz 常数为 $L$，范数有界为 $M$。对初值问题 $\frac{\mathrm{d}}{\mathrm{d}s}\mathbf{x}(s) = \mathbf{f}(\mathbf{x}(s), s)$，$\mathbf{x}(s_k) = \mathbf{x}_k$，以及步长 $\Delta s_k := s_{k-1} - s_k$ 的一阶相容单步方法（如显式 Euler），若 $\max_k \Delta s_k \leq h$ 且 $hL < 1$，则全局误差满足：

$$\max_k \left\|\hat{\mathbf{x}}_k - \mathbf{x}(s_k)\right\|_2 \leq Ch,$$

其中 $C$ 仅依赖 $(L, M)$ 和时间域长度。

**推论 5.21**（整流直线流的采样精度）。对整流 ODE $\frac{\mathrm{d}}{\mathrm{d}s}\mathbf{x} = \tilde{\mathbf{u}}_{t(s)}(\mathbf{x})$，若 $\boldsymbol{\epsilon}^*(\cdot, t)$ 关于 $t$ 一致局部 Lipschitz，则任何最大步长为 $h$ 的一阶单步方法在状态上招致 $O(h)$ 全局误差。特别地，选取在 $\bar{\alpha}$-时间上均匀的网格（即 $t \mapsto \bar{\alpha}_t$ 单调，$s$ 正比于 $-\log\bar{\alpha}_t$），可保证近似均匀的收缩率和有利的误差常数。

> **实践意义**：DDIM 端点更新（定理 5.16）可视为整流流沿直线特征的精确单步积分（当预测器在两个网格时刻之间被固定时）。引理 5.20 量化了预测器随状态和时间变化时的偏差：在 $s$ 中近似均匀的步长（即 $\log(1-\bar{\alpha}_t)$ 或 $\bar{\alpha}_t$ 中均匀）减小误差，而整流避免了在终端噪声水平附近由因子 $\kappa(t) = \dot{\rho}(t)/(1-\rho(t))$ 引起的爆炸。

---

## 6 引导扩散（Guided Diffusion）

引导扩散用外部**控制信号**来增强去噪扩散模型，使采样朝向用户指定目标引导（如类别条件或文本条件生成）。从数学上看，它修改了逆时间转移，同时保持前向过程和变分基础不变。

使用与前面相同的符号：$\alpha_t := 1 - \beta_t$，$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$，DDPM/DDIM 逆向参数化为：

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{c}) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\; \mu_\theta(\mathbf{x}_t, t, \mathbf{c}),\; \sigma_t^2\mathbf{I}\right), \quad \mu_\theta(\mathbf{x}_t, t, \mathbf{c}) = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c})\right),$$

其中 $\sigma_t^2 = \tilde{\beta}_t := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$（DDPM）或 $\sigma_t^2 = 0$（DDIM）。

---

### 6.1 基于分类器的引导：后验得分修正

假设有一个辅助分类器 $p_\phi(y \mid \mathbf{x}_t, t)$，它在时间步 $t$ 时对**带噪**输入 $\mathbf{x}_t$ 预测标签 $y$ [2]。目标后验（到归一化常数）为：

$$p(\mathbf{x}_t \mid y) \propto p_\theta(\mathbf{x}_t)\,p_\phi(y \mid \mathbf{x}_t, t),$$

其梯度为两个得分之和：

$$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t \mid y) = \nabla_{\mathbf{x}_t}\log p_\theta(\mathbf{x}_t) + \nabla_{\mathbf{x}_t}\log p_\phi(y \mid \mathbf{x}_t, t). \tag{37}$$

在离散 DDPM 形式中，这给出逆向转移的**均值偏移**：

$$\bar{\mu}_\theta(\mathbf{x}_t, t \mid y) = \mu_\theta(\mathbf{x}_t, t) + \lambda\,\sigma_t^2\,\nabla_{\mathbf{x}_t}\log p_\phi(y \mid \mathbf{x}_t, t), \tag{38}$$

其中 $\lambda \geq 0$ 控制引导强度，$\sigma_t^2$ 将更新缩放到该步的噪声水平。采样过程为：

$$\mathbf{x}_{t-1} = \bar{\mu}_\theta(\mathbf{x}_t, t \mid y) + \sigma_t\,\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d).$$

**引理 6.1**（分类器引导均值偏移）。在小步（$\beta_t$ 小）近似下，从后验 $p(\mathbf{x}_t \mid y) \propto p_\theta(\mathbf{x}_t)\,p_\phi(y \mid \mathbf{x}_t, t)$ 采样，由式 (38) 的逆向更新一阶近似实现；即分类器添加了一个以 $\lambda\sigma_t^2$ 缩放的得分项到无引导均值 $\mu_\theta(\mathbf{x}_t, t)$。

> **直觉**：$\log p(\mathbf{x}_t \mid y) = \log p_\theta(\mathbf{x}_t) + \log p_\phi(y \mid \mathbf{x}_t, t)$。向其梯度方向的单步更新（小步近似）给出 $\bar{\mu}_\theta = \mu_\theta + \lambda\sigma_t^2\nabla_{\mathbf{x}_t}\log p_\phi(y \mid \mathbf{x}_t, t)$，$\sigma_t^2$ 因子确保量纲与逆向转移方差一致。

在**概率流 ODE** 中，同样的想法适用：将得分 $s_t(\mathbf{x}) = \nabla_\mathbf{x}\log p_t(\mathbf{x})$ 替换为 $s_t + \lambda\nabla_\mathbf{x}\log p_\phi(y \mid \mathbf{x}, t)$，将 VP 漂移 $\dot{\mathbf{x}} = -\frac{\beta(t)}{2}(\mathbf{x} + s_t(\mathbf{x}))$ 修正为：

$$\dot{\mathbf{x}} = -\frac{\beta(t)}{2}\!\left(\mathbf{x} + s_t(\mathbf{x}) + \lambda\,\nabla_\mathbf{x}\log p_\phi(y \mid \mathbf{x}, t)\right),$$

在小步极限下与离散均值偏移 (38) 一致。

---

### 6.2 无分类器引导：条件-无条件混合

无分类器引导通过训练**单个**扩散模型同时在条件和无条件模式下运行，消除了外部分类器的需求。去噪器在有条件信号 $\mathbf{c}$ 和无条件信号下都预测噪声；在采样时，将两个预测混合 [7]：

$$\hat{\boldsymbol{\epsilon}}_\lambda(\mathbf{x}_t, t, \mathbf{c}) = \hat{\boldsymbol{\epsilon}}_{\text{uncond}}(\mathbf{x}_t, t) + \lambda\!\left(\hat{\boldsymbol{\epsilon}}_{\text{cond}}(\mathbf{x}_t, t, \mathbf{c}) - \hat{\boldsymbol{\epsilon}}_{\text{uncond}}(\mathbf{x}_t, t)\right), \quad \lambda \geq 0, \tag{39}$$

将 $\hat{\boldsymbol{\epsilon}}_\lambda$ 代入逆向均值：

$$\mu_\theta^{(\lambda)}(\mathbf{x}_t, t, \mathbf{c}) = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\boldsymbol{\epsilon}}_\lambda(\mathbf{x}_t, t, \mathbf{c})\right), \quad \mathbf{x}_{t-1} = \mu_\theta^{(\lambda)}(\mathbf{x}_t, t, \mathbf{c}) + \sigma_t\,\mathbf{z}. \tag{40}$$

**引理 6.2**（无分类器引导即隐式梯度步）。若噪声预测器在条件嵌入中局部线性，则 $\Delta\hat{\boldsymbol{\epsilon}} := \hat{\boldsymbol{\epsilon}}_{\text{cond}} - \hat{\boldsymbol{\epsilon}}_{\text{uncond}}$ 作为增强与 $\mathbf{c}$ 一致性的学习方向。混合 (39) 等价于沿此方向走一步大小为 $\lambda$ 的步，通过 (40) 诱导类似于分类器引导的均值偏移。

> **证明**：设 $\mathbf{c} \mapsto \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c})$ 局部线性，对基准（丢弃）条件 $\hat{\boldsymbol{\epsilon}}_{\text{cond}} \approx \hat{\boldsymbol{\epsilon}}_{\text{uncond}} + \mathbf{J}_\mathbf{c}\mathbf{c}$，故差值 $\Delta\hat{\boldsymbol{\epsilon}}$ 捕获预测器对 $\mathbf{c}$ 的局部敏感性。代入后得 $\mu_\theta^{(\lambda)} = \mu_\theta\big|_{\text{uncond}} - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}\lambda\Delta\hat{\boldsymbol{\epsilon}}$，即与 $\lambda\Delta\hat{\boldsymbol{\epsilon}}$ 成比例的均值偏移。将 $\Delta\hat{\boldsymbol{\epsilon}}$ 解释为条件驱动得分方向（类比 $\nabla_{\mathbf{x}_t}\log p(\mathbf{c} \mid \mathbf{x}_t)$），表明无分类器引导向满足 $\mathbf{c}$ 迈出一阶步，精神上与分类器引导一致但无需显式分类器。

得分形式的等价性直接由 $\nabla_\mathbf{x}\log p_t(\mathbf{x}) = -\hat{\boldsymbol{\epsilon}}(\mathbf{x},t)/\sqrt{1-\bar{\alpha}_t}$ 给出：

$$s_t^{(\lambda)}(\mathbf{x}) := -\frac{\hat{\boldsymbol{\epsilon}}_\lambda(\mathbf{x}, t, \mathbf{c})}{\sqrt{1-\bar{\alpha}_t}} = (1-\lambda)\,s_t^{\text{uncond}}(\mathbf{x}) + \lambda\,s_t^{\text{cond}}(\mathbf{x}). \tag{41}$$

> **直觉**：无分类器引导将得分替换为无条件得分与条件得分的凸（或当 $\lambda > 1$ 时外推）组合；VP 概率流 ODE 以 $s_t^{(\lambda)}$ 代替 $s_t$。当 $\lambda > 1$ 时为**外推**：得分被推向远离无条件分布、更强烈满足条件 $\mathbf{c}$ 的方向，以牺牲多样性换取保真度。

---

### 6.3 时变引导调度

常数 $\lambda$ 在不同噪声水平下可能引导过强或不足。设 $\text{SNR}_t := \bar{\alpha}_t/(1-\bar{\alpha}_t)$，$\ell_t := \log\text{SNR}_t$。一个简单的调度族为：

$$\lambda_t = \lambda_{\max}\,\sigma\!\left(a\,\ell_t + b\right), \tag{42}$$

其中 $\sigma(u) = 1/(1+e^{-u})$，$a > 0$ 在晚期（高 SNR）时刻增强引导，$b$ 设定拐点。

**启发式理解**：在早期（低 SNR，高噪声）时用小 $\lambda_t$ 避免与噪声对抗；在晚期用大 $\lambda_t$ 锐化与 $\mathbf{c}$ 一致的属性。由于 CFM/MFM 下的最优回归对正时间加权不变（参见引理 5.18），$\lambda_t$ 作为采样侧预处理器而非改变学习的去噪器。

**DDIM vs. DDPM 中引导的作用方式**：
- 在 DDIM（$\sigma_t^2 = 0$）中，引导仅通过 $\hat{\boldsymbol{\epsilon}}_\lambda$ 进入；
- 在 DDPM 中，基于分类器的步骤 (38) 还额外以 $\sigma_t^2 = \tilde{\beta}_t$ 缩放，在低 SNR 时自然地调节引导。

---

### 6.4 稳定性：失效模式与补救措施

强引导会产生伪影。常见失效模式及实践对策（这些是采样侧调整，保持学习目标不变）：

- **过饱和 / 纹理模糊**：极大的 $\lambda$ 可能导致对比度坍缩或细节丢失。
  **修复**：对 $\hat{\mathbf{x}}_0$ 进行动态阈值处理（按百分位裁剪），或范数重缩放引导：
  $$\Delta\hat{\boldsymbol{\epsilon}} \leftarrow \Delta\hat{\boldsymbol{\epsilon}} \cdot \frac{\|\hat{\boldsymbol{\epsilon}}_{\text{uncond}}\|_2}{\|\Delta\hat{\boldsymbol{\epsilon}}\|_2 + \varepsilon}.$$

- **曝光漂移**：累积的正向或负向偏移改变全局亮度。
  **修复**：在像素空间对引导增量进行零均值化，或保持一个小噪声基底 $\sigma_t \leftarrow \max(\sigma_t, \sigma_{\min})$ 以维持随机性。

- **属性泄漏**：对一个属性的引导干扰其他属性。
  **修复**：使用时变调度 (42)（适中的晚期 $\lambda_{\max}$）；可选地，将 $\Delta\hat{\boldsymbol{\epsilon}}$ 投影到与属性 token 对齐的子空间（若可用）。

---

### 6.5 引导蒸馏

使用大 $\lambda$ 的引导采样在设备上速度较慢或不稳定。学生模型可以通过沿教师轨迹匹配预测的 $\hat{\mathbf{x}}_0$（或速度/得分）来**蒸馏**使用引导 $\lambda_T > 1$ 的教师模型 [9]。

**定义 6.3**（引导蒸馏目标）。给定教师预测器 $\hat{\boldsymbol{\epsilon}}_{\lambda_T}^T$ 和网格 $\{\tau_k\}$，定义学生损失：

$$\mathcal{L}_{\text{distill}} = \sum_k \mathbb{E}\!\left[\left\|\hat{\mathbf{x}}_0^S(\mathbf{x}_{\tau_k}) - \hat{\mathbf{x}}_0^T(\mathbf{x}_{\tau_k})\right\|_2^2\right], \quad \hat{\mathbf{x}}_0(\mathbf{x}_t) = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\boldsymbol{\epsilon}}(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}.$$

**命题 6.4**（精确匹配下的轨迹一致性）。若学生在网格 $\{\tau_k\}$ 上精确匹配教师的 $\hat{\mathbf{x}}_0$，则学生的 DDIM 采样复现教师的引导轨迹。

> **证明**：DDIM 端点关系（定理 5.16）将 $\mathbf{x}_{\tau_{k-1}}$ 表示为 $\hat{\mathbf{x}}_0(\mathbf{x}_{\tau_k})$ 和 $\hat{\boldsymbol{\epsilon}}(\mathbf{x}_{\tau_k})$ 的函数。在 $\tau_k$ 处 $\hat{\mathbf{x}}_0$ 的精确一致意味着在 $\tau_{k-1}$ 处计算端点的相等性；对 $k$ 的归纳完成证明。

---

## 7 结论

本教程提供了去噪扩散模型及其确定性对应物的自洽、数学精确的阐述。

**核心主线回顾**：

1. **前向构造**：以 $\alpha_t := 1-\beta_t$ 和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ 的方差保持链产生封闭形式边际 $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I}_d)$ 和重参数化 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$，支撑了分析和优化（通过重参数化梯度）。

2. **真实后验**：完整推导了真实单步后验 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$，含均值和方差。均值有等价的 $\mathbf{x}_0$-形式和 $\boldsymbol{\epsilon}$-形式，后验方差 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$，在噪声预测下训练化归为加权去噪。

3. **加速方法（离散时间）**：DDIM 定义了保持 DDPM 边际的逆条件族，允许自由的逐步方差 $\sigma_t^2$；$\sigma_t^2 = \tilde{\beta}_t$ 还原 DDPM，$\sigma_t^2 = 0$ 给出确定性采样器，均值清晰地分解为对齐 $\mathbf{x}_0$ 的分量和对齐 $\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0$ 的重缩放噪声分量，简化的步骤调度保持一致性。

4. **潜扩散（第 4.4 节）**：整个构造被平移到潜空间——编码器将数据映射到潜变量并在其上运行扩散，解码器在最后恢复样本。所有恒等式（前向边际、真实后验、含因子 $\beta_t/\sqrt{1-\bar{\alpha}_t}$ 的逆向均值以及 DDPM/DDIM 方差选择）在 $\mathbf{x} \mapsto \mathbf{z}$ 替换后完全转移。这解释了潜扩散为何在不改变底层数学的情况下实现大幅加速。

5. **可控生成（第 6 节）**：分类器引导通过以步骤方差缩放的得分项 $\nabla_{\mathbf{x}_t}\log p_\phi(y \mid \mathbf{x}_t, t)$ 修正逆向均值。无分类器引导通过混合条件和无条件噪声预测避免辅助分类器；分析表明这作为向条件对齐方向的学习一阶步，效果类似分类器引导。两种情形下，更新与相同的逆向参数化一致，保真度与多样性的权衡由引导尺度控制。

6. **流匹配视角**：将扩散族通过确定性概率流 ODE 重新表达，速度使用得分 $s_t = \nabla_\mathbf{x}\log p_t$。证明了该 ODE 与前向 SDE 具有相同时间边际（定理 5.4），将 DDIM 式确定性采样与连续时间视角联系起来（推论 5.17）。这一等价性阐明了为何高阶 ODE 求解器、步长控制和数值分析工具在扩散采样中有效，并激励了直接以速度场匹配规定边际的训练目标。

**一以贯之的代数纪律**：前向链提供了可处理的边际和用于优化的重参数化；真实后验确定了有原则的逆向步骤；DDIM 和概率流给出确定性采样器；潜扩散无需改变理论即可重置计算；引导以受控方式修改逆向均值。始终如一地使用 $\bar{\alpha}_t$ 和后验均值因子 $\beta_t/\sqrt{1-\bar{\alpha}_t}$，避免了常见的代数错误，使推导保持对齐。

**自然的后续方向**：
- **数值层面**：自适应求解器、学习时间参数化、蒸馏到极少步骤；
- **调节机制**：扩展结构化控制、适配器，在相同逆向均值框架内；
- **理论层面**：更精细的离散化误差保证、引导鲁棒性以及参数化可识别性，仍是活跃的研究方向。

希望本教程中的证明和精心符号使这些方向更易于入手，并作为实现和分析的可靠参考。

---

> **译注**：本译文力求忠实原文的数学内涵，技术术语保留英文对照，公式编号与原文一致。核心概念如 *score function*（得分函数）、*coupling*（耦合）、*interpolant*（插值）均按学界惯例翻译。

---

## References

[1] Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In *International Conference on Learning Representations (ICLR)*, 2023.

[2] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

[3] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[4] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[5] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In *International Conference on Learning Representations (ICLR)*, 2023.

[6] Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*. PMLR, 2021.

[7] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In *Proceedings of the 39th International Conference on Machine Learning (ICML)*. PMLR, 2022.

[8] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.

[9] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In *International Conference on Learning Representations (ICLR)*, 2022.

[10] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *Proceedings of the 32nd International Conference on Machine Learning (ICML)*. PMLR, 2015.

[11] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *International Conference on Learning Representations (ICLR)*, 2021.

[12] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations (ICLR)*, 2021.
