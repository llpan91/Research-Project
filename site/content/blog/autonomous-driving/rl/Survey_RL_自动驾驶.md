---
title: "Survey_RL_自动驾驶"
date: 2026-05-11
tags: ["autonomous-driving", "强化学习"]
summary: "来自 自动驾驶 研究笔记"
draft: false
---

# Survey: 基于强化学习的自动驾驶研究综述

> 本报告系统梳理强化学习（RL）在自动驾驶中的研究进展，涵盖算法分类、任务场景、仿真平台、挑战与前沿方向。

---

## 一、引言

自动驾驶系统的核心难题是如何在开放、动态、不确定的交通环境中做出安全且高效的决策。传统方法依赖规则引擎和手工设计的状态机，难以覆盖长尾场景。**强化学习（RL）** 通过与环境的试错交互学习最优策略，天然适配序贯决策问题，已成为自动驾驶研究的重要范式。

### 1.1 为什么用 RL 做自动驾驶？

| 优势 | 说明 |
|------|------|
| 自主探索 | 不依赖专家标注数据，通过奖励信号自主学习 |
| 处理不确定性 | MDP 框架天然建模随机环境 |
| 长期优化 | 最大化累积回报，兼顾短期安全与长期效率 |
| 端到端潜力 | 从感知到控制的统一优化 |

### 1.2 RL 在自动驾驶中的定位

```
感知 → 预测 → 规划 → 控制
              ↑          ↑
           RL 可介入的位置
         (决策层 / 端到端)
```

---

## 二、问题建模：自动驾驶中的 MDP/POMDP

### 2.1 状态空间（State / Observation）

| 类型 | 表示 | 代表工作 |
|------|------|----------|
| 原始像素 | 前视摄像头图像 | DQN-based, End-to-End |
| BEV 语义图 | 鸟瞰视角语义栅格 | Roach, DreamerAD |
| 向量化表示 | 车辆状态 + 周围物体列表 | highway-env 系列 |
| 混合表示 | 图像 + 结构化向量 | InterFuser, TCP |
| 点云 / 体素 | LiDAR 3D 数据 | LiDAR-based RL |

### 2.2 动作空间（Action）

| 类型 | 说明 | 示例 |
|------|------|------|
| **离散动作** | 有限驾驶行为集合 | {加速, 减速, 左变道, 右变道, 保持} |
| **连续动作** | 连续控制量 | (steering ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]) |
| **路径点输出** | 预测未来轨迹点 | [(x₁,y₁), (x₂,y₂), ...] + PID 执行 |
| **分层动作** | 高层决策 + 底层控制 | 高层选行为 → 底层出控制 |

### 2.3 奖励设计（Reward）

奖励设计是 RL 自动驾驶中最关键也最具挑战的环节：

```
R_total = w₁·R_progress + w₂·R_speed + w₃·R_collision
        + w₄·R_lane + w₅·R_comfort + w₆·R_traffic_rule
```

| 奖励项 | 典型定义 |
|--------|----------|
| 路线进度 | 沿路线前进距离的正奖励 |
| 速度奖励 | 接近目标速度时给正奖励 |
| 碰撞惩罚 | 碰撞事件给予 -1 ~ -100 的大额惩罚 |
| 车道偏离 | 偏离车道中心的负奖励，正比于横向偏移 |
| 舒适度 | 对横向加速度、急转向、急刹车的惩罚 |
| 交通规则 | 闯红灯、逆行、超速等违规的惩罚 |

**挑战**：奖励塑形（reward shaping）需大量领域经验，权重调节困难，稀疏奖励导致学习低效。

---

## 三、RL 算法分类与代表工作

### 3.1 分类框架

```
RL for Autonomous Driving
├── Model-Free RL
│   ├── Value-Based
│   │   ├── DQN, Double DQN, Dueling DQN
│   │   └── Rainbow, C51 (分布式 RL)
│   ├── Policy-Gradient
│   │   ├── REINFORCE
│   │   ├── PPO, TRPO
│   │   └── A3C, A2C
│   └── Actor-Critic
│       ├── DDPG, TD3
│       ├── SAC (Soft Actor-Critic)
│       └── IMPALA, V-trace
├── Model-Based RL
│   ├── Dyna-style (学模型+用模型生成数据)
│   ├── World Model (Dreamer 系列)
│   ├── MPC + learned dynamics
│   └── TD-MPC / TD-MPC2
├── Offline RL
│   ├── CQL, IQL, BCQ, BEAR
│   └── Decision Transformer
├── Multi-Agent RL (MARL)
│   ├── Independent Learners
│   ├── CTDE (中心化训练分布式执行)
│   └── Communication-based MARL
├── Safe / Constrained RL
│   ├── CMDP (Constrained MDP)
│   ├── CPO, PCPO
│   └── Lagrangian methods
└── Hierarchical RL
    ├── Option framework
    └── Goal-conditioned RL
```

### 3.2 Model-Free 方法

#### 3.2.1 Value-Based 方法

| 方法 | 核心思想 | 自动驾驶应用 |
|------|----------|-------------|
| **DQN** | 深度 Q 网络，离散动作 | 简单车道保持、变道决策 |
| **Double DQN** | 减少 Q 值过估计 | 交叉口通行决策 |
| **Dueling DQN** | 分离状态价值与优势函数 | 高速公路决策 |
| **Rainbow** | 集成多种 DQN 改进 | 复杂城市场景 |

**局限**：仅适用于离散动作空间，难以处理连续控制。

#### 3.2.2 Policy-Gradient / Actor-Critic 方法

| 方法 | 特点 | 自动驾驶应用 |
|------|------|-------------|
| **PPO** | 稳定、易调参、最广泛使用 | 端到端驾驶、CARLA benchmark |
| **SAC** | 最大熵框架、自动温度调节 | 连续控制、复杂城市驾驶 |
| **TD3** | Twin Critic、延迟更新 | 高速公路车道变换 |
| **DDPG** | 确定性策略梯度 | 连续控制（早期工作） |
| **A3C/A2C** | 异步/同步多 worker 并行 | 加速训练收敛 |

**PPO 和 SAC 是当前自动驾驶 RL 研究中最主流的算法选择。**

### 3.3 Model-Based 方法

| 方法 | 核心思想 | 代表工作 |
|------|----------|----------|
| **Dreamer 系列** | 潜在空间世界模型 + 想象训练 | DreamerV3, DreamerAD |
| **TD-MPC / TD-MPC2** | 时序差分 + 模型预测控制 | Hansen et al. 2022/2024 |
| **MBPO** | Model-Based Policy Optimization | Janner et al. 2019 |
| **SimPLe** | 用学到的模型替代真实环境 | Kaiser et al. 2020 |
| **ISO-Dream** | 隔离可控/不可控因素 | Pan et al. 2022 |

**核心优势**：样本效率高 10-100 倍；可在学到的模型中进行安全探索。

### 3.4 Offline RL 方法

从预收集的数据集中学习策略，无需在线交互：

| 方法 | 核心思想 | 自动驾驶优势 |
|------|----------|-------------|
| **CQL** | 保守 Q 学习，惩罚 OOD 动作 | 从人类驾驶数据学习 |
| **IQL** | 隐式 Q 学习，避免策略外查询 | 离线数据利用 |
| **Decision Transformer** | 将 RL 转化为序列建模 | 利用 Transformer 架构 |
| **BCQ / BEAR** | 批约束 / 引导策略 | 安全的离线学习 |

**意义**：直接利用大量已有的人类驾驶数据，避免在线探索的安全风险。

### 3.5 Multi-Agent RL (MARL)

| 范式 | 说明 | 应用场景 |
|------|------|----------|
| **Independent Learners** | 每辆车独立学习 | 简单交通流 |
| **CTDE** | 集中训练、分散执行 | 协作换道、交叉口 |
| **Communication** | 车辆间信息共享 | V2V 协作驾驶 |
| **Mean Field** | 大规模交通流近似 | 宏观交通优化 |

### 3.6 Safe / Constrained RL

| 方法 | 核心思想 | 说明 |
|------|----------|------|
| **CMDP** | 约束马尔可夫决策过程 | 在优化回报的同时满足安全约束 |
| **CPO / PCPO** | 约束策略优化 | 保证每次更新不违反安全约束 |
| **Lagrangian** | 拉格朗日乘子法 | 将约束转化为惩罚项 |
| **SafeDreamer** | 世界模型 + 安全约束 | 在想象空间中进行安全规划 |
| **Shield** | 安全屏蔽层 | 在执行前过滤不安全动作 |

---

## 四、典型任务场景

### 4.1 场景分类

| 场景 | 难度 | RL 方法 |
|------|------|---------|
| 车道保持 (Lane Keeping) | ★☆☆ | DQN, PPO |
| 自适应巡航 (ACC) | ★☆☆ | DDPG, SAC |
| 高速变道 (Lane Change) | ★★☆ | DQN, PPO, SAC |
| 匝道合流 (Ramp Merge) | ★★☆ | MARL, PPO |
| 信号灯交叉口 (Signalized Intersection) | ★★★ | PPO, SAC, Hierarchical RL |
| 无保护左转 (Unprotected Left Turn) | ★★★ | Model-Based, Safe RL |
| 环岛 (Roundabout) | ★★★ | MARL, Hierarchical |
| 城市复杂场景 (Urban Complex) | ★★★★ | World Model, End-to-End |
| 混合交通 (Pedestrians + Cyclists) | ★★★★ | Safe RL, POMDP |

### 4.2 端到端 vs 模块化

```
模块化方案:  感知 → 预测 → 规划 → 控制  (RL 仅在某一模块)
端到端方案:  传感器输入 ────────────→ 控制输出  (RL 统一优化)
混合方案:    感知模块 → RL 决策/规划 → 控制模块
```

---

## 五、仿真平台与 Benchmark

| 平台 | 特点 | 适用场景 |
|------|------|----------|
| **CARLA** | 高保真城市仿真、传感器模拟、多天气 | 全栈端到端研究、benchmark 标准 |
| **highway-env** | 轻量级、纯向量化、快速迭代 | 算法原型验证、高速场景 |
| **MetaDrive** | 可组合场景、安全 RL 友好 | 安全 RL、泛化性研究 |
| **SMARTS** | 多智能体、社会性驾驶 | MARL 研究 |
| **nuPlan** | 基于真实数据的规划 benchmark | 规划算法评估 |
| **Waymax** | Waymo 发布、大规模仿真 | 真实交通场景回放 |
| **SUMO** | 宏观交通流仿真 | 交通信号控制、宏观优化 |
| **LGSVL / AirSim** | 高保真渲染 | 感知 + RL 联合研究 |

### CARLA Leaderboard 主要指标

| 指标 | 定义 |
|------|------|
| Route Completion (RC) | 路线完成百分比 |
| Infraction Score (IS) | 违规乘法惩罚因子 |
| Driving Score (DS) | DS = RC × IS，综合评分 |

---

## 六、代表性工作梳理（按时间线）

| 年份 | 工作 | 方法 | 亮点 |
|------|------|------|------|
| 2016 | Mnih et al. | A3C + 模拟器 | 早期端到端 RL 驾驶 |
| 2018 | Dosovitskiy et al. | RL + CARLA | CARLA benchmark 建立 |
| 2019 | Chen et al. | Model-Free RL | 城市驾驶 |
| 2020 | Toromanoff et al. | Rainbow + IL 预训练 | CARLA 挑战赛优胜 |
| 2021 | Chen et al. (Roach) | PPO + BEV + IL | RL 训练特权 agent → IL 蒸馏 |
| 2021 | Zhang et al. | SAC + 分层 | 高速公路决策 |
| 2022 | Shao et al. (TCP) | RL + IL 互补 | 轨迹+控制双路预测 |
| 2022 | ISO-Dream | 隔离世界模型 | 可控 vs 不可控因素分离 |
| 2023 | Think2Drive | World Model + RL | 思考后驾驶 |
| 2023 | DreamerV3 | 通用世界模型 RL | 跨领域 SOTA |
| 2023 | DreamerAD | DreamerV3 → AD | 世界模型自动驾驶适配 |
| 2024 | SafeDreamer | Dreamer + CMDP | 安全世界模型 RL |
| 2024 | DriveWM | 驾驶世界模型 | 大规模视频生成式世界模型 |

---

## 七、关键挑战与开放问题

### 7.1 样本效率

```
问题: Model-Free RL 需要数百万步交互
缓解:
  → Model-Based RL (Dreamer 系列)
  → Offline RL (利用已有数据)
  → IL 预训练 + RL 微调 (Roach 范式)
  → 迁移学习 / 课程学习
```

### 7.2 安全性

```
问题: 探索阶段不可避免碰撞等危险行为
缓解:
  → Constrained RL (CMDP, CPO)
  → Safety Shield (过滤不安全动作)
  → 在世界模型中安全探索 (SafeDreamer)
  → 仿真训练 + Sim2Real
```

### 7.3 Sim-to-Real 迁移

```
问题: 仿真器训练的策略在真实世界性能退化
缓解:
  → Domain Randomization (随机化仿真参数)
  → Domain Adaptation (对抗训练)
  → 高保真仿真 (缩小 domain gap)
  → Real-world fine-tuning
```

### 7.4 奖励设计

```
问题: 手工设计奖励困难、权重敏感
缓解:
  → Inverse RL (从专家演示学习奖励)
  → Reward Learning (学习奖励函数)
  → 多目标优化 / Pareto RL
  → LLM-guided reward design
```

### 7.5 泛化性

```
问题: 策略在训练场景外表现差
缓解:
  → 大规模多样化训练场景
  → 课程学习 (由简入难)
  → 元学习 (Meta-RL)
  → Foundation Model + RL
```

### 7.6 可解释性

```
问题: 深度 RL 策略是黑盒，难以获得监管认证
缓解:
  → 注意力可视化
  → 世界模型预测可视化
  → Hierarchical RL (可解释的高层决策)
  → 语言解释 (LLM + RL)
```

---

## 八、前沿趋势（2024-2025+）

### 8.1 Foundation Model + RL

- **LLM 作为驾驶大脑**：LLM 提供高层推理与常识，RL 提供精细控制
- **VLM (视觉-语言模型)**：多模态理解交通场景
- **代表工作**：DriveGPT4, LanguageMPC, GPT-Driver

### 8.2 生成式世界模型

- **视频生成模型**作为世界模拟器：GAIA-1, DriveDreamer, Vista
- **扩散模型 (Diffusion)**：生成多样化未来场景
- **代表工作**：GenAD, Copilot4D, OccWorld

### 8.3 Offline RL + 大规模数据

- 利用 nuScenes, Waymo Open Dataset 等大规模真实数据
- Decision Transformer / Trajectory Transformer 范式
- 从人类驾驶日志中离线学习策略

### 8.4 安全对齐 (Safety Alignment)

- 借鉴 RLHF (人类反馈强化学习) 思路
- 人类偏好对齐的驾驶策略
- 安全约束的形式化验证

### 8.5 多智能体协作

- V2X (车路协同) + MARL
- 大规模交通流优化
- 社会性驾驶行为建模

---

## 九、总结

| 维度 | 现状 | 趋势 |
|------|------|------|
| 算法 | PPO/SAC 主流，World Model 崛起 | Foundation Model + RL 融合 |
| 数据 | 依赖仿真，离线数据利用不足 | Offline RL + 大规模真实数据 |
| 安全 | 约束 RL 初步探索 | 形式化安全保证 + RLHF |
| 泛化 | 场景特定策略 | 通用驾驶 agent |
| 部署 | 仿真验证为主 | Sim2Real + 闭环测试 |

> **核心观点**：RL 在自动驾驶中正从 "能跑通仿真" 走向 "能安全泛化"，Model-Based RL（特别是世界模型方法）和 Offline RL 是提升样本效率与安全性的两大关键路径；Foundation Model 与 RL 的深度融合将是下一阶段的核心方向。

---

## 参考文献（推荐阅读）

1. Kiran, B.R. et al. "Deep Reinforcement Learning for Autonomous Driving: A Survey." IEEE TITS, 2022.
2. Hafner, D. et al. "Mastering Diverse Domains through World Models." (DreamerV3) arXiv:2301.04104, 2023.
3. Zhang, Z. et al. "Roach: An End-to-End RL Coach for Autonomous Driving." ICCV 2021.
4. Shao, H. et al. "TCP: Trajectory and Control Prediction." NeurIPS 2022.
5. Hu, A. et al. "MILE: Model-Based Imitation Learning for Urban Driving." NeurIPS 2022.
6. Huang, Z. et al. "SafeDreamer: Safe Reinforcement Learning with World Models." ICLR 2024.
7. Wang, Y. et al. "DriveDreamer: Real-World-Driven World Models for AD." arXiv 2023.
8. Hu, A. et al. "GAIA-1: A Generative World Model for Autonomous Driving." arXiv 2023.

---

## 专题：为什么自动驾驶 RL 需要 World Model 而非传统向量预测作 Reward？

### 核心矛盾

**闭环训练的因果性需求** 与 **传统预测模块的开环假设** 之间存在根本矛盾。

---

### 1. 因果性缺失：预测不依赖 ego 的动作

传统预测模块（如轨迹预测）本质上在做：

```
P(future_agents | current_obs)
```

而 RL 的 reward 需要的是：

```
R(s, a) → 需要知道 ego 执行动作 a 后，环境如何演变
```

**问题**：如果 ego 刹车/变道/加速，周围车辆的行为会**反应性地改变**。传统预测模块不以 ego action 为条件，无法回答"我做这个动作后，后车会怎样"。

> 举例：ego 急插队，传统预测模块可能仍然预测后车匀速行驶，导致 RL agent 学到"可以随意插队"的错误策略。

---

### 2. 开环 vs 闭环的分布偏移

传统预测模块在**真实数据分布**下训练，其隐含假设是 ego 的行为符合正常驾驶员。

RL 训练期间，agent 会进行大量**探索性的非典型动作**，导致：

```
训练预测模块的数据分布  ≠  RL探索时产生的状态分布
```

预测模块遇到 OOD 状态时，输出的向量结果不可靠，基于它的 reward 信号就会产生错误的梯度方向。

---

### 3. 多步展开的时序一致性崩溃

RL 的价值估计（n-step return / Monte Carlo）需要对未来多步进行连续模拟：

```
s_t → s_{t+1} → s_{t+2} → ... → s_{t+n}
```

用独立预测模块做多步展开时：
- **各 agent 预测独立**，没有交互约束（A车预测不知道B车的预测）
- **误差逐步累积**，到第5步时已经严重不合理
- **物理可行性无法保证**（两辆车可能"穿越"）

World Model 的联合建模可以保证多步的**空间一致性和社会合理性**。

---

### 4. Reward Hacking 风险

向量化预测只给出轨迹点/速度，缺乏中间状态的细粒度信息：

| 基于向量的 Reward | 问题 |
|---|---|
| TTC (Time to Collision) | 仅用预测轨迹算 TTC，忽略交互导致的轨迹变化 |
| 舒适度指标 | 向量无法反映路面状况、障碍物形状等细节 |
| 通行效率 | 静态向量地图无法模拟信号灯变化、动态障碍物闯入等 |

Agent 容易学到"欺骗"预测模块的策略，而非真正安全的驾驶行为。

---

### 5. OCC / 地图预测的额外问题

对于 OCC 等网格化环境表达做时序预测，还有：

- **传播一致性**：OCC 预测是逐帧独立的，不保证物理上的连续演化
- **Ego 视角感知**：World Model 可以模拟"如果 ego 移动到新位置，能看到什么"；OCC 预测做不到这一点
- **遮挡处理**：World Model 可以维护遮挡区域的概率状态；独立 OCC 预测对遮挡的处理是被动的

---

### World Model 解决了什么？

```
World Model 本质上是一个可微分的、以 ego action 为条件的环境模拟器：

s_{t+1} = f(s_t, a_ego, θ)
```

| 能力 | 传统向量预测 | World Model |
|---|---|---|
| ego-conditioned 预测 | ✗ | ✓ |
| 多智能体交互建模 | ✗ | ✓ |
| 多步一致性展开 | 差 | ✓ |
| 支持 Dyna/想象训练 | ✗ | ✓ |
| 反事实推理 | ✗ | ✓ |
| OOD 鲁棒性 | 低 | 相对更好 |

---

### 实践中的折中方案

不是说完全不能用向量预测，但需要注意适用条件：

1. **仅用于离线 reward shaping**（非闭环）：用真实标注轨迹作 supervision 信号，不做在线模拟
2. **与 World Model 混合**：用向量预测初始化场景，用 World Model 做交互展开
3. **受限场景**：高速路等交互简单的场景，agent 行为对他车影响有限，可以做一阶近似

对于城市复杂场景的端到端 RL，需要 World Model 的**反应性 + 一致性 + 可展开性**，这是传统独立预测模块结构上无法提供的。
