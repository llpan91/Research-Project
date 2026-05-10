# DDPM 与 DDIM：从随机扩散到确定性采样

## 摘要

本报告系统阐述了 DDPM（去噪扩散概率模型）和 DDIM（去噪扩散隐式模型）的理论基础与核心差异。首先从 DDPM 的后验分布出发，推导出其反向采样公式；随后通过引入方差控制参数，推导 DDIM 的确定性采样框架；最后进行详细的理论对比和实用性分析。核心发现是：DDIM 通过构造非马尔可夫前向过程，将反向采样从必须执行固定步数的随机过程转变为支持跳步采样的确定性映射，使采样速度提升 20~50 倍。

---

## 第一部分：理论基础

### 1. 问题背景

**DDPM 的核心困境**：
- 前向过程（加噪）：$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$
- 反向过程（去噪）：需要学习从 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 逐步恢复到 $\mathbf{x}_0$
- **问题**：采样必须按 $T \to T-1 \to \cdots \to 0$ 顺序执行，步数通常为 1000，采样速度极慢

**解决思路**：
- DDIM 的核心思想是：能否跳过中间步骤，直接从任意 $\mathbf{x}_t$ 跳到 $\mathbf{x}_s$（$s < t$）？
- 答案是肯定的，但需要重新理解反向过程的数学本质

### 2. 符号定义与回顾

| 符号 | 含义 |
|------|------|
| $\alpha_t = 1 - \beta_t$ | 第 $t$ 步的信号保留系数，$\beta_t \in (0,1)$ 是噪声调度超参数 |
| $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ | 前 $t$ 步信号保留系数的累乘（决定了 $\mathbf{x}_t$ 中原始信号的比例） |
| $\mathbf{x}_t$ | 第 $t$ 步的带噪样本 |
| $\mathbf{x}_0$ | 原始干净样本 |
| $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | 加的高斯噪声 |
| $q(\cdot)$ | 前向过程的分布（固定，无需训练） |
| $p_\theta(\cdot)$ | 反向过程的分布（由神经网络参数化） |
| $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ | 神经网络预测的噪声 |

---

## 第二部分：DDPM 的反向采样公式推导

### 3. 从后验分布出发

DDPM 的反向过程基于真实后验分布（也称 DDPM 后验），这是 Bayes 规则在扩散过程中的应用：

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_{t-1} ; \hat{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}_d \right)$$

其中后验方差为：
$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$

而**后验均值**是关键：
$$\hat{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$

其中 $\boldsymbol{\epsilon} = \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}}$ 是前向过程中被加入的噪声。

### 4. 重参数化与均值化简

**前向过程的重参数化**：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$$

将此代入后验均值，展开化简：

$$\begin{align}
\hat{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) &= \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{1-\bar{\alpha}_t} \mathbf{x}_t + \frac{\beta_t \sqrt{\bar{\alpha}_t}}{1-\bar{\alpha}_t} \mathbf{x}_0 \right) \\
&= \frac{\sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0
\end{align}$$

**关键观察**：后验均值可以表示为 $\mathbf{x}_t$ 和 $\mathbf{x}_0$ 的线性组合。

### 5. DDPM 采样公式的获取过程

#### 5.1 问题的关键

第 4 节我们得到的后验均值是：
$$\hat{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0$$

**但有个问题**：这个公式包含真实的原始样本 $\mathbf{x}_0$，但在**采样时我们不知道 $\mathbf{x}_0$**（那正是我们要生成的东西！）。因此需要用其他可获得的信息来表示这个均值。

#### 5.2 转化的关键：利用噪声表示

从前向过程的重参数化公式：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$$

可以反推得到：
$$\mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

将此代入后验均值，展开化简：

$$\begin{align}
\hat{\boldsymbol{\mu}}_t &= \frac{\sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \cdot \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}} \\
&= \frac{\sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} \mathbf{x}_t - \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon} \\
&= \frac{1}{1-\bar{\alpha}_t} \left[ \sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_{t-1}) + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{\sqrt{\bar{\alpha}_t}} \right] \mathbf{x}_t - \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{\sqrt{\bar{\alpha}_t}} \boldsymbol{\epsilon} \\
&= \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon} \right)
\end{align}$$

**关键发现**：后验均值可以用两种等价的形式表示：
- **含 $\mathbf{x}_0$ 的形式**（理论），用于推导变分下界
- **含 $\boldsymbol{\epsilon}$ 的形式**（实践），用于实际采样

$$\hat{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$

#### 5.3 用模型预测替代真实噪声

在训练时，扩散模型学习预测加入的噪声。对于任意 $\mathbf{x}_t$，模型预测：
$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \approx \boldsymbol{\epsilon}$$

采样时，用预测噪声 $\hat{\boldsymbol{\epsilon}}_\theta$ 替代真实噪声 $\boldsymbol{\epsilon}$：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \right)$$

#### 5.4 加入反向方差

DDPM 后验均值已经确定，但分布还包括方差。从高斯分布的采样可以写成：

$$\mathbf{x}_{t-1} \sim \mathcal{N}(\hat{\boldsymbol{\mu}}_t, \sigma_t^2 \mathbf{I})$$

因此采样时需要添加随机噪声项。采样公式变为：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \boldsymbol{\epsilon}_t$$

其中：
- $\sigma_t = \sqrt{\tilde{\beta}_t} = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t}$ 是 DDPM 的反向方差（由后验方差决定）
- $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是采样时新增的随机噪声

**这就是 DDPM 的完整采样公式！**

#### 5.5 特点

- 方差 $\sigma_t$ 是**固定的**，由后验方差决定，无法调节
- 必须添加随机噪声 $\boldsymbol{\epsilon}_t$，采样过程**完全随机**
- 必须按顺序执行所有 $T$ 步（无法跳步）
- 这是 DDPM 采样效率低的根本原因

---

## 第三部分：DDIM 的推导与优化

### 6. DDIM 的核心思想

DDIM 的关键改进是：**将反向方差 $\sigma_t$ 变成可调参数**，而不是固定值。这使得我们可以：

1. 设 $\sigma_t = 0$，使反向过程完全确定性
2. 通过跳步采样，减少所需的迭代次数

### 7. DDIM 采样公式的推导

基于 DDPM 的后验分布，如果我们要用模型预测的 $\hat{\mathbf{x}}_0$ 来表示 $\mathbf{x}_{t-1}$，有：

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

则采样公式可改写为：
$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) + \sigma_t \boldsymbol{\epsilon}_t$$

其中 $\sigma_t$ 是可调的方差参数。

### 8. 跳步采样（DDIM 的核心优势）

DDIM 支持从任意 $\mathbf{x}_t$ 跳步到 $\mathbf{x}_s$（$s < t$），采样公式为：

$$\mathbf{x}_s = \sqrt{\bar{\alpha}_s} \hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_s - \sigma_{t,s}^2} \cdot \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) + \sigma_{t,s} \boldsymbol{\epsilon}$$

其中方差参数为：
$$\sigma_{t,s} = \eta \sqrt{\frac{1-\bar{\alpha}_s}{1-\bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_s}{\bar{\alpha}_t}}$$

$\eta \in [0,1]$ 是全局随机性调节参数。

---

## 第四部分：DDPM vs DDIM 详细对比

### 9. 采样公式对比

| 方面 | DDPM | DDIM |
|------|------|------|
| **单步采样** | $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\boldsymbol{\epsilon}}_\theta \right) + \sqrt{\tilde{\beta}_t} \boldsymbol{\epsilon}_t$ | $\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \hat{\boldsymbol{\epsilon}}_\theta + \sigma_t \boldsymbol{\epsilon}_t$ |
| **跳步采样** | ❌ 不支持 | ✓ 支持任意跳步 |
| **方差参数** | 固定（$\sigma_t = \sqrt{\tilde{\beta}_t}$） | 可调（$\sigma_t$ 自由设置） |

### 10. 关键特性对比

| 对比维度 | DDPM | DDIM |
|----------|------|------|
| **采样随机性** | 每一步必须添加随机噪声，采样过程**完全随机** | 方差可调：$\sigma_t=0$ 时**完全确定性**，$\sigma_t>0$ 时引入随机性 |
| **采样步数** | 必须执行所有 $T$ 步（通常 1000 步） | 支持跳步，可压缩到 50/20 步，采样速度提升 **20~50 倍** |
| **生成质量** | 采样步数固定，生成质量**稳定且最优** | 跳步采样时：$\eta=0$ 生成质量**略优**；$\eta>0$ 可平衡速度与多样性 |
| **理论基础** | 基于**马尔可夫随机过程** | 基于**非马尔可夫过程**的确定性映射 |
| **隐空间特性** | 随机游走，无确定映射 | 确定性采样下，$\mathbf{x}_T$ 与 $\mathbf{x}_0$ **一一对应**（支持插值、编辑） |
| **模型复用** | 原生 DDPM 模型 | ✓ 可复用 DDPM 训练的模型，**无需重新训练** |
| **计算开销** | 采样步数多（1000 步） | 采样步数少（20~50 步），总开销大幅降低 |
| **适用场景** | 高质量生成，对速度要求不高 | 快速生成、隐空间编辑、大模型推理加速 |

### 11. 理论差异的深层原因

**DDPM 的本质**：
- 反向过程是**严格的马尔可夫随机链**
- 每一步 $\mathbf{x}_{t-1}$ 仅依赖于 $\mathbf{x}_t$（当前状态）
- 必须添加固定方差噪声，无法消除随机性
- **结果**：采样必须逐步进行，无法跳步

**DDIM 的本质**：
- 通过构造**非马尔可夫的前向过程**，反向过程可以是确定性映射
- 打破了马尔可夫链的步数限制
- 同一时间步可以从不同的中间时刻跳过来
- **结果**：支持任意跳步，采样效率大幅提升

---

## 第五部分：实践应用指南

### 12. 选择建议

**场景 1：追求最高生成质量**
- 推荐：**DDPM**（1000 步采样）
- 优势：生成质量最优，适合对质量要求极高的场景
- 劣势：采样速度最慢（几分钟到几十分钟）
- 典型应用：艺术创作、高质量图像生成

**场景 2：快速生成 + 隐空间编辑**
- 推荐：**DDIM**（50 步确定性采样，$\eta=0$）
- 优势：采样速度快（秒级），支持隐空间操作
- 劣势：生成质量略低于 DDPM
- 典型应用：实时推理、图像编辑、风格转换

**场景 3：平衡速度与多样性**
- 推荐：**DDIM**（20~50 步，$\eta \in (0,1)$，如 $\eta=0.5$）
- 优势：采样速度与 DDPM 相近质量的折中
- 劣势：生成结果存在随机波动
- 典型应用：批量生成、数据增强

### 13. 参数设置指南

```
# 确定性 DDIM（最快，质量相对较低）
DDIM(steps=50, eta=0.0)

# 平衡型 DDIM（推荐）
DDIM(steps=50, eta=0.5)

# 随机型 DDIM（接近 DDPM 质量）
DDIM(steps=100, eta=1.0)

# DDPM（最慢，质量最高）
DDPM(steps=1000)
```

---

## 第六部分：总结

### 14. 核心要点

| 要点 | 说明 |
|------|------|
| **关键差异** | DDPM 采用固定方差的随机采样；DDIM 采用可调方差的确定性或随机采样 |
| **核心突破** | 从马尔可夫链转变为非马尔可夫链，打破步数限制 |
| **实际收益** | 采样速度提升 20~50 倍，同时保持可接受的生成质量 |
| **理论统一** | DDIM 是 DDPM 的广义框架，当 $\eta=1$ 时完全退化为 DDPM |
| **模型复用** | DDIM 可复用任何 DDPM 预训练模型，无需重新训练 |

### 15. 研究方向

1. **进一步加速**：Consistency Models、Latent Consistency Models 在 DDIM 基础上进一步加速
2. **质量优化**：通过更优的噪声调度、指导机制提升跳步采样质量
3. **隐空间应用**：充分利用 DDIM 的确定性特性进行图像编辑、插值等
4. **多模态扩展**：将 DDIM 框架推广到文本、音频等模态

---

## 参考公式速查表

| 公式/概念 | 数学表达式 |
|----------|----------|
| 前向过程 | $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ |
| DDPM 采样 | $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\boldsymbol{\epsilon}}_\theta \right) + \sigma_t \boldsymbol{\epsilon}_t$ |
| DDIM 采样 | $\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \hat{\boldsymbol{\epsilon}}_\theta + \sigma_t \boldsymbol{\epsilon}_t$ |
| DDIM 跳步 | $\mathbf{x}_s = \sqrt{\bar{\alpha}_s} \hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_s - \sigma_{t,s}^2} \hat{\boldsymbol{\epsilon}}_\theta + \sigma_{t,s} \boldsymbol{\epsilon}$ |
| 累积系数 | $\bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$ |
| 预测干净样本 | $\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \hat{\boldsymbol{\epsilon}}_\theta}{\sqrt{\bar{\alpha}_t}}$ |
