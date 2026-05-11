---
title: "RAD_RAD-2_深度系统解读"
date: 2026-05-11
tags: ["autonomous-driving", "扩散模型在自动驾驶中的应用"]
summary: "来自 自动驾驶 研究笔记"
draft: false
---

# RAD / RAD-2 深度系统解读

> 本文档针对本目录下两篇论文进行系统梳理：
>
> - `RAD.pdf`: **RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning**
> - `RAD-V2.pdf`: **RAD-2: Scaling Reinforcement Learning in a Generator-Discriminator Framework**
>
> 写作时间：2026-05-08。  
> 资料基础：本地 PDF 全文为主，结合互联网公开的一手资料，包括 arXiv、项目页、OpenReview、GitHub、3DGS/Street Gaussians/DiffusionDrive 等相关论文页面。

## 0. 一句话总览

RAD 和 RAD-2 讨论的是同一条主线：**如何把端到端自动驾驶规划从“开环模仿专家轨迹”推进到“闭环交互中自我修正”**。

RAD 的核心贡献是：用 3D Gaussian Splatting 重建真实道路片段，构造接近真实视觉输入的闭环仿真环境，再用 RL + IL 联合训练端到端驾驶策略，使策略在碰撞、偏航、偏离专家轨迹等风险上获得真实闭环反馈。

RAD-2 的核心贡献是：承认直接用稀疏 RL 奖励优化高维扩散轨迹不稳定，于是把系统拆成 **扩散生成器 Generator** 和 **RL 训练的轨迹判别器 Discriminator**。生成器负责给出多模态候选轨迹，判别器负责基于闭环长期结果重排序；同时用 BEV-Warp 替代昂贵的图像级/3DGS 渲染训练环境，大幅提升闭环 RL 吞吐。

如果说 RAD 证明了“3DGS 闭环 RL 可以让端到端驾驶更安全”，RAD-2 则进一步回答了“如何让这种闭环 RL 扩展到扩散式多模态规划，并且训练得更稳定、更高效”。

## 1. 研究背景：为什么端到端自动驾驶需要闭环 RL

### 1.1 端到端 IL 的根本问题

现有端到端自动驾驶规划多数依赖 imitation learning，即给定传感器输入，监督网络模仿人类专家轨迹。这条路线有明显工程吸引力：数据规模大、训练稳定、离线评估容易、部署链路简单。但 RAD 系列认为它存在两个根本问题。

第一是 **causal confusion**。IL 学到的是观测和动作之间的统计相关，而不一定是真正导致专家动作的因果因素。例如模型可能从历史轨迹外推未来轨迹，而不是理解“前方行人会穿行，所以需要减速”。

第二是 **open-loop gap**。IL 训练和常规离线评测通常是开环的：模型预测错了也不会改变下一帧输入。但真实驾驶是闭环系统：一次小偏差会改变车体位置，下一帧观测分布随之变化，误差逐步积累，最终进入训练数据很少覆盖的状态。

这两个问题都指向同一个缺口：模型需要在交互环境中经历“动作改变世界状态，世界状态再反过来影响动作”的过程，并从长期后果中学习。

### 1.2 为什么不是直接用真实道路 RL

真实道路闭环 RL 不现实：安全风险、成本、低效率都不可接受。因此需要仿真环境。

传统游戏引擎仿真器如 CARLA 可以交互，但视觉和真实世界有 sim-to-real gap。纯日志回放真实视觉，但 ego 车辆一旦偏离原轨迹，就没有真实传感器输入。世界模型和视频生成模型更灵活，但长时序多视角生成容易漂移，吞吐也重。

RAD/RAD-2 分别给出两种折中：

- RAD：用 3DGS 把真实片段重建成可重新渲染的数字孪生环境，强调真实视觉与几何。
- RAD-2：用 BEV 特征空间的几何 warp 做闭环交互，强调训练吞吐与可扩展性，同时在 3DGS 环境中验证泛化。

### 1.3 扩散模型在自动驾驶规划中的位置

扩散模型适合表达多模态连续轨迹分布：同一个场景下，合理行为可能包括跟车、变道、减速、绕行等多个模式。DiffusionDrive 等工作已经说明，扩散式规划器能比单轨迹回归更好建模复杂驾驶分布。

但扩散规划也带来一个 RL 难题：输出是高维、连续、时序相关的轨迹，而闭环奖励通常是低维标量，例如是否碰撞、TTC、是否完成路线。把低维稀疏奖励直接反传到高维轨迹生成分布，容易出现 credit assignment 困难和训练不稳定。

RAD-2 的生成器-判别器框架正是为了解这个问题：**不直接用 RL 改扩散生成器的完整高维输出，而先让 RL 训练一个低维评分/重排序模块，再用结构化信号逐步移动生成器分布。**

## 2. RAD：3DGS 闭环强化学习训练端到端驾驶策略

### 2.1 RAD 的问题定义

RAD 面向的是端到端驾驶策略训练。输入是多视角图像序列，输出是短时控制动作分布。论文的核心假设是：

> 如果能在真实重建的 3DGS 环境里让策略闭环试错，并通过安全相关奖励约束策略，那么端到端策略可以比纯 IL 更好地学习碰撞规避和异常状态恢复。

这不是简单“加一个 RL fine-tuning”。RAD 对仿真环境、动作空间、奖励拆分、PPO 目标、辅助目标、IL 正则都做了专门设计。

### 2.2 策略网络结构

RAD 的感知-规划主干可以概括为：

```text
多视角图像
  -> BEV Encoder
  -> Map Head / Agent Head 得到地图 token 和交通参与者 token
  -> Image Encoder 得到 dense image token
  -> Planning Head 融合 scene token、导航信息、ego state
  -> 输出横向动作分布 pi(ax|s)、纵向动作分布 pi(ay|s)
  -> 采样动作控制车辆
```

几个关键点：

- **BEV Encoder** 把多视角图像映射到 BEV 特征。
- **Map Head** 用 map tokens 学习车道线、道路边界、箭头、交通信号等向量化地图元素。
- **Agent Head** 用 agent tokens 预测周围参与者的位置、朝向、尺寸、速度和多模态未来轨迹。
- **Image Encoder** 额外提取原图 dense token，补充 map/agent token 不容易表达的细节。
- **Planning Head** 用 Transformer decoder 将 planning embedding 与 scene representation 交互，再结合导航和自车状态输出动作分布。

RAD 的动作空间不是连续轨迹，而是短时 0.5 秒 horizon 内的 **解耦离散动作**：

- 横向位移 `ax`: 61 个离散选项，范围约为 `[-0.75m, 0.75m]`。
- 纵向位移 `ay`: 61 个离散选项，范围约为 `[0m, 15m]`。

这种设计降低了 RL 搜索空间。横向和纵向分别建模，也方便后续奖励和优势函数按控制维度拆分。

### 2.3 三阶段训练范式

RAD 采用三阶段训练：

| 阶段 | 训练内容 | 更新参数 | 目的 |
|---|---|---|---|
| Perception Pre-Training | 地图、参与者感知监督 | BEV encoder、map head、agent head | 让 token 具备结构化场景语义 |
| Planning Pre-Training | 用专家轨迹做 IL | image encoder、planning head | 避免 RL 冷启动，初始化人类驾驶先验 |
| Reinforced Post-Training | 3DGS 闭环中 RL + IL 交替 | image encoder、planning head | 用闭环反馈提升安全性，同时保持人类对齐 |

强化后训练阶段中，RAD 使用多个 worker 并行 rollout。每个 worker 随机采样 3DGS 环境，让当前策略控制 ego 车闭环交互，并把 `(s_t, a_t, r_{t+1}, s_{t+1}, ...)` 写入 rollout buffer。优化时交替执行：

- RL step：使用 PPO 从闭环 rollout buffer 更新策略。
- IL step：用专家示范做监督更新，作为 human alignment 正则。

论文中给出的 reinforced post-training 关键配置包括：

- RL worker 数：32
- RL batch size：32
- IL batch size：128
- GAE：`gamma = 0.9, lambda = 0.95`
- PPO clip：横向 `epsilon_x = 0.1`，纵向 `epsilon_y = 0.2`
- deviation threshold：`dmax = 2.0m`，`psi_max = 40°`
- RL:IL 训练比例约为 4:1

### 2.4 3DGS 闭环环境与交互机制

RAD 的环境来自真实驾驶片段重建。论文使用 StreetGaussian 风格的 3DGS 重建，并做了若干优化：

- 道路表面使用 mesh 约束，使高斯球贴合路面几何。
- 天空单独建模，避免前景和天空混淆。
- 对车辆、行人等前景物体优化 pose。
- 加入深度和法向一致性监督，改善几何和表面细节。

在闭环交互中：

- ego 车由 AD policy 控制。
- 其他交通参与者采用 log-replay，以尽量恢复真实交通流。
- 车辆动力学使用 kinematic bicycle model。
- 每个时间步策略输出 0.5 秒内横纵向动作，换算成速度和转角，更新 ego pose。
- 3DGS 环境根据新 pose 渲染/生成下一状态和奖励。

这里的关键价值是：ego 可以偏离原始 log 轨迹，系统仍能从新视角产生传感器输入。这使“闭环”不再停留在轨迹层，而是进入了视觉感知输入层。

### 2.5 奖励建模：四类终止性负反馈

RAD 的 reward 由四类事件组成：

| 奖励源 | 触发条件 | 主要约束 |
|---|---|---|
| Dynamic Collision `r_dc` | ego bbox 与动态障碍 bbox 重叠 | 避免与车辆/行人等动态物体碰撞 |
| Static Collision `r_sc` | ego bbox 与静态障碍高斯重叠 | 避免撞路缘、隔离带、静态障碍 |
| Positional Deviation `r_pd` | 与专家轨迹最近点距离超过阈值 | 不要偏离可行驾驶走廊 |
| Heading Deviation `r_hd` | 与专家轨迹匹配 heading 差超过阈值 | 不要朝向异常、跨道漂移 |

任一事件触发都会立即终止 episode。论文给出的理由是：碰撞或大幅偏离之后，3DGS 环境生成的后续传感器数据可能变噪，继续训练会污染 RL 信号。

这个奖励设计很务实，但也意味着 RAD 的 RL 目标仍然强依赖专家轨迹。它不是让模型自由探索所有可能驾驶策略，而是在专家轨迹附近通过闭环负反馈学会更安全的局部调整。

### 2.6 PPO 优化：横纵向奖励拆分

RAD 根据动作维度拆分 reward：

- 横向 reward：`r_x = r_sc + r_pd + r_hd`
- 纵向 reward：`r_y = r_dc`

对应地，value function 也拆成：

- `V_x(s)`：估计横向累计奖励
- `V_y(s)`：估计纵向累计奖励

优势函数通过 GAE 分别计算，再用改造后的 PPO 目标优化横向和纵向动作分布。

这个拆分的直觉是：

- 动态碰撞尤其是前车/行人碰撞，更直接对应纵向减速或加速。
- 静态障碍、轨迹偏离、航向偏离，更直接对应横向转向修正。

这种维度拆分降低了 credit assignment 难度，也让后续辅助目标更容易设计。

### 2.7 辅助目标：把稀疏事件变成方向性梯度

RAD 认为 PPO 的终止性奖励仍然太稀疏，因此增加四个 directional auxiliary objectives：

| 辅助目标 | 行为含义 |
|---|---|
| Dynamic Collision Aux | 如果前方碰撞，增加减速动作概率；如果后方碰撞，增加加速动作概率 |
| Static Collision Aux | 如果左侧静态障碍风险高，增加右转概率；右侧风险高则反之 |
| Positional Deviation Aux | 如果 ego 偏左，增加向右纠偏概率；偏右则反之 |
| Heading Deviation Aux | 根据 heading 偏差方向增加相反修正动作概率 |

这部分是 RAD 工程味最强的设计之一：它没有指望稀疏 RL 自动学出所有控制方向，而是把“风险发生时应该往哪个动作概率方向推”显式编码进训练目标。代价是引入先验和规则，收益是收敛更快、方向更稳定。

### 2.8 RAD 实验结果解读

#### 2.8.1 RL + IL 训练策略对比

| Stage 3 策略 | CR↓ | DCR↓ | SCR↓ | DR↓ | ADD↓ | Long. Jerk↓ | Lat. Jerk↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| IL | 0.229 | 0.211 | 0.018 | 0.066 | 0.238 | 3.928 | 0.103 |
| RL | 0.143 | 0.128 | 0.015 | 0.080 | 0.345 | 4.204 | 0.085 |
| RL + IL | 0.089 | 0.080 | 0.009 | 0.063 | 0.257 | 4.495 | 0.082 |

解读：

- 纯 IL 最像专家轨迹，ADD 最低，但碰撞率高。
- 纯 RL 安全性提升，但偏离专家行为更多，ADD 变差。
- RL + IL 在安全性和人类行为相似性之间取得最好折中。

这印证了论文主张：RL 解决闭环安全反馈，IL 负责约束行为分布不要跑偏。

#### 2.8.2 奖励源消融

完整四类 reward 时 CR = 0.089，是所有配置中最低。去掉 dynamic collision reward 的配置 CR = 0.238，说明动态障碍碰撞惩罚对安全性最关键。

这也符合自动驾驶场景常识：复杂城市道路里，最难的并不是静态道路边界，而是动态参与者带来的速度和交互不确定性。

#### 2.8.3 辅助目标消融

完整 PPO + 四类辅助目标时 CR = 0.089。只有 PPO 时 CR = 0.249；只有辅助目标无 PPO 时 CR = 0.187。

这说明：

- 辅助目标单独就能提供有用方向性信号。
- 但它不能替代真正的闭环策略梯度。
- PPO 和辅助目标组合时效果最好。

#### 2.8.4 与 IL 方法的闭环对比

RAD 在 3DGS evaluation benchmark 上与 TransFuser、VAD、GenAD、VADv2 对比：

| Method | CR↓ | DCR↓ | SCR↓ | DR↓ | ADD↓ | Long. Jerk↓ | Lat. Jerk↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 0.320 | 0.273 | 0.047 | 0.235 | 0.263 | 4.538 | 0.142 |
| VAD | 0.335 | 0.273 | 0.062 | 0.314 | 0.304 | 5.284 | 0.550 |
| GenAD | 0.341 | 0.299 | 0.042 | 0.291 | 0.265 | 11.37 | 0.320 |
| VADv2 | 0.270 | 0.240 | 0.030 | 0.243 | 0.273 | 7.782 | 0.171 |
| RAD | 0.089 | 0.080 | 0.009 | 0.063 | 0.257 | 4.495 | 0.082 |

最显著的是 CR 从 VADv2 的 0.270 降到 0.089，约 3 倍改善。RAD 的 ADD 并不是绝对最低，但很接近 TransFuser/GenAD，同时碰撞明显更低，这说明它用一点行为偏移换来了更强安全性。

### 2.9 RAD 的核心贡献与局限

RAD 的真正贡献不是“用了 PPO”，而是完成了一个较完整的闭环训练闭环：

```text
真实驾驶数据
  -> 3DGS 重建
  -> 可交互视觉闭环环境
  -> 解耦短时动作空间
  -> 安全/偏离奖励
  -> PPO + IL 联合训练
  -> 闭环 benchmark 评测
```

主要局限：

- 3DGS 重建对非刚体行人、未观测视角、弱光等仍有限。
- 其他交通参与者 log-replay，不会真正响应 ego 行为，因此交互博弈不完整。
- reward 依赖专家轨迹偏离阈值，可能限制策略学到超越专家的合理行为。
- 每个场景独立重建 3DGS，训练环境构建成本高。
- 评测 benchmark 规模和分布仍是论文内部构建，外部复现难度较高。

## 3. RAD-2：面向扩散规划的生成器-判别器闭环 RL

### 3.1 RAD-2 相对 RAD 解决了什么

RAD-2 的标题是 Scaling Reinforcement Learning in a Generator-Discriminator Framework。它关注的不是简单提升 RAD，而是把 RAD 的闭环 RL 思路迁移到更强的扩散式多模态规划器上，并解决两个新问题：

1. **高维轨迹直接 RL 不稳定**：扩散生成器输出完整未来轨迹，维度高、时序强相关，稀疏标量奖励很难分配到具体轨迹变化。
2. **3DGS 训练吞吐不够**：大规模 RL 需要大量闭环交互，逐场景重建和图像级渲染成本高。

因此 RAD-2 提出两项核心改造：

- 用 **Generator-Discriminator** 解耦轨迹生成和轨迹评分。
- 用 **BEV-Warp** 做高吞吐特征级闭环仿真。

### 3.2 生成器-判别器框架

RAD-2 将规划 policy 拆成两个模块：

```text
Diffusion Generator Gθ:
  输入场景观测 o
  生成 M 条多模态候选轨迹 C = {τ1, ..., τM}

RL-based Discriminator Dφ:
  输入候选轨迹和场景上下文
  输出每条轨迹的质量分数
  选择/重排序最优轨迹用于闭环执行
```

形式上：

- `Gθ(τ | o)` 表示扩散生成器建模的候选轨迹分布。
- `Dφ(τ | o, C)` 表示判别器在候选集合中的轨迹选择分布。
- 两者组合成 joint policy。

这背后的思想是：**让扩散模型保持生成多样性，让 RL 主要优化一个低维评分函数**。轨迹质量的长期后果通过闭环 reward 训练判别器，而不是直接把稀疏 reward 硬塞进扩散去噪链。

### 3.3 Diffusion Generator

生成器负责建模多模态未来轨迹分布。

输入包括：

- BEV feature `Tb`
- 静态地图元素 `Xmap`
- 动态参与者 `Xagent`
- 导航输入 `Xnav`

经过轻量编码器得到 map token、agent token、navigation token，再与 BEV feature 融合成 scene embedding。随后 DiT-style conditional denoising network 从高斯噪声轨迹开始，经过 K 步去噪生成 M 条候选未来轨迹。

与 RAD 的离散短时动作不同，RAD-2 的生成器直接输出连续 `(x, y)` 轨迹，表达力更强，也更接近当前扩散规划路线。

### 3.4 RL-based Discriminator

判别器的作用不是判断真假，而是判断“这条候选轨迹在当前场景下闭环执行后是否安全、高效、长期质量好”。

结构上：

- 对轨迹点序列用 MLP 嵌入。
- 加 `[CLS]` token 后送入 Transformer encoder，得到轨迹级 query。
- 用独立参数的 map/agent 编码器构建 scene condition。
- 轨迹 query 通过 cross-attention 聚合 map、BEV、agent、agent-map interaction 信息。
- MLP + sigmoid 输出 `[0,1]` 分数。

判别器天然输出低维标量/分数，和 RL reward 的标量属性更匹配，因此优化稳定性更好。

### 3.5 BEV-Warp：高吞吐闭环环境

RAD-2 的 BEV-Warp 是非常关键的工程设计。它不重新渲染图像，也不每个训练 step 调 3DGS，而是在 BEV feature 上做几何 warp。

基本过程：

1. 从真实 log 初始化参考 BEV feature 和参考 ego pose。
2. 当前策略选择轨迹，iLQR controller 跟踪该轨迹，得到新的模拟 ego pose。
3. 计算模拟 pose 与 log reference pose 的相对变换矩阵。
4. 用 bilinear interpolation 对参考 BEV feature 做空间 warp，得到下一步模拟 BEV feature。
5. 将 warped BEV feature 输入感知/规划模块继续闭环。

BEV-Warp 依赖一个假设：BEV feature 具备较强的空间等变性。也就是说，真实世界里的位姿变化可以近似对应为 BEV 特征图上的几何变换。

它的优势：

- 不需要昂贵图像级渲染。
- 不需要每个场景训练 3DGS。
- 可以大规模跑闭环 RL。
- 对 BEV-centric 架构特别适配。

它的限制：

- 只适合显式 BEV 空间表示的系统。
- 不能自然支持纯 raw image policy 或无空间网格结构的统一 latent policy。
- warp 不能生成真正新出现/被遮挡变化后的内容，只是局部几何近似。
- 动态 agent 和复杂遮挡的真实性仍受 log/feature 表示限制。

RAD-2 因此仍在 3DGS 环境做额外验证，用来说明 BEV-Warp 训练得到的能力不是只在特征仿真里有效。

### 3.6 Trajectory-Based Controller

RAD-2 中 generator 输出的是轨迹，不是底层控制。论文使用 iLQR-based controller 跟踪选中轨迹。

这带来两个好处：

- planner 只需要输出未来参考轨迹，符合多数高阶规划系统接口。
- 控制执行与轨迹生成解耦，闭环环境中的 ego pose 更新更稳定。

但这也意味着实验中的闭环性能不仅由生成器/判别器决定，还受到 controller tracking 质量影响。

### 3.7 Temporally Consistent Rollout

多模态轨迹规划有一个隐含问题：如果每一帧都重新采样并切换轨迹模式，车辆意图会高频抖动，reward 也很难归因到某个具体轨迹选择。

RAD-2 使用 trajectory reuse / latched execution：

- 在时间 `t` 选中一条最优轨迹。
- 将它转成控制序列。
- 接下来固定执行 `Hreuse` 个 step，而不是每帧立刻重规划换模式。

这样做的目的是让一段闭环结果更明确地对应某次轨迹选择，降低 credit assignment 噪声。

论文消融显示 `Hreuse = 8` 最平衡：

| Hreuse | CR↓ | Safety@1↑ | EP@1.0↑ |
|---:|---:|---:|---:|
| 2 | 0.355 | 0.580 | 0.701 |
| 4 | 0.324 | 0.627 | 0.604 |
| 8 | 0.337 | 0.615 | 0.728 |
| 16 | 0.332 | 0.596 | 0.744 |

论文文字认为 8 在稳定 credit assignment 和响应灵活性之间最合适；虽然 CR 不是该表最低，但综合 safety/progress 更优。

### 3.8 RAD-2 奖励：安全 TTC + 进度 EP

RAD-2 的 reward 比 RAD 更偏向轨迹级闭环表现，主要包括：

#### 3.8.1 Safety-Criticality Reward

安全奖励基于 Time-To-Collision。对每个 simulation step，计算 ego 未来投影占用与环境真实占用的最早交集时间 `Tt`。序列级奖励取 rollout 内最差时间裕度：

```text
r_coll = min_t (Tt - Tmax)
```

这是 bottleneck-style reward：只要 rollout 中任一时刻有严重风险，整段序列 reward 就会被最危险时刻主导。

#### 3.8.2 Navigational Efficiency Reward

效率奖励基于 Ego Progress，记为 `rho`。论文设定一个目标区间 `[rho_low, rho_high] = [1.05, 1.10]`。低于下界表示太慢，高于上界表示过激或不人类对齐。

这说明 RAD-2 并不是简单鼓励越快越好，而是鼓励“比参考路线略积极、但仍在人类可接受窗口内”的进度。

### 3.9 TC-GRPO：轨迹级组相对策略优化

RAD-2 的 TC-GRPO 可以理解为把 GRPO 思想改造成适合轨迹闭环 planning 的形式。

对同一个初始状态生成一组 rollout `{O_i}`，每个 rollout 有一个序列级 reward `r_i`。对组内 reward 标准化得到 advantage：

```text
A_i = (r_i - mean(group rewards)) / std(group rewards)
```

然后只在新轨迹采样/latched interval 开始的稀疏决策点上计算 PPO-style clipped objective。这样 advantage 强化的是一段持续轨迹意图，而不是每一帧零散动作。

RAD-2 还加入自适应 entropy regularization，避免判别器分数过早塌缩到 0 或 1。消融显示带 entropy 的目标更安全：

| RL Objective | CR↓ | Safety@1↑ | EP@1.0↑ |
|---|---:|---:|---:|
| without entropy | 0.254 | 0.697 | 0.727 |
| with entropy | 0.234 | 0.730 | 0.736 |

### 3.10 OGO：On-policy Generator Optimization

如果只训练判别器，生成器候选分布不变，系统上限会被候选轨迹质量限制。因此 RAD-2 进一步提出 OGO，让 generator 也随闭环反馈逐步变好。

关键是它不直接对完整高维轨迹做 RL，而是做 **reward-guided longitudinal optimization**：

- 如果 TTC 太低，有碰撞风险，对轨迹进行纵向减速/时间压缩。
- 如果进度不足且安全裕度足够，对轨迹进行纵向加速/时间扩展。
- 保持空间路径形状，主要优化纵向时序。
- 将优化后的轨迹作为 on-policy supervision，用 MSE fine-tune generator。

这是一种很工程化的折中：只沿着 reward 最相关、最稳定的纵向维度移动生成器分布，避免破坏扩散生成器已经学到的空间多模态结构。

### 3.11 RAD-2 训练数据与流程

论文给出的关键数据规模：

- 生成器预训练：约 50,000 小时真实驾驶数据。
- BEV-Warp 闭环数据：50k clips，每段 10-20 秒。
- 训练集：安全导向 10k clips，效率导向 10k clips。
- 闭环评测：安全导向 512 clips，效率导向 512 clips。
- 3DGS benchmark：1044 clips 训练，256 clips 评测。

RL 实现要点：

- 每个 clip 下采集多个 rollout。
- 使用 trajectory reuse 维持短期行为一致性。
- 对每个 clip 计算 reward 均值和方差，低方差 clip 被过滤掉，因为缺少区分性训练信号。
- rollout 存入 FIFO replay buffer，长度为 8。
- group size 为 4，做组内标准化 advantage。
- 判别器每次新 batch 进入都更新；生成器等 buffer 完整刷新后再优化，频率约 8:1。
- 判别器从预训练 planning head 可共享组件初始化，显著好于随机初始化。

### 3.12 RAD-2 实验结果解读

#### 3.12.1 BEV-Warp 闭环评测

| Method | Safety CR↓ | AF-CR↓ | Safety@1↑ | Safety@2↑ | EP-Mean↑ | EP@1.0↑ | EP@0.9↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 0.563 | 0.275 | 0.400 | 0.346 | 0.897 | 0.244 | 0.531 |
| VAD | 0.594 | 0.299 | 0.371 | 0.312 | 0.904 | 0.252 | 0.623 |
| GenAD | 0.592 | 0.305 | 0.363 | 0.309 | 0.930 | 0.467 | 0.736 |
| ResAD | 0.533 | 0.264 | 0.418 | 0.281 | 0.970 | 0.516 | 0.894 |
| RAD-2 | 0.234 | 0.092 | 0.730 | 0.596 | 0.988 | 0.736 | 0.984 |

相对强 baseline ResAD：

- CR 从 0.533 降到 0.234。
- AF-CR 从 0.264 降到 0.092。
- Safety@1 从 0.418 升到 0.730。
- EP@1.0 从 0.516 升到 0.736。

这说明 RAD-2 不只是更保守减少碰撞，也提升了路线进度。

#### 3.12.2 3DGS photorealistic benchmark

| Method | CR↓ | AF-CR↓ | Safety@1↑ | Safety@2↑ |
|---|---:|---:|---:|---:|
| Senna-2 | 0.269 | 0.077 | 0.667 | 0.565 |
| RAD | 0.281 | 0.113 | 0.613 | 0.543 |
| RAD-2 | 0.250 | 0.078 | 0.723 | 0.644 |

RAD-2 在 CR 和 Safety@1/2 上最好，但 AF-CR 与 Senna-2 基本接近。这里要谨慎解读：RAD-2 的闭环 safety margin 很强，但 at-fault collision 不一定全面压倒 Senna-2。

#### 3.12.3 Senna-2 open-loop benchmark

| Method | FDE↓ | ADE↓ | CR(%)↓ | DCR(%)↓ | SCR(%)↓ |
|---|---:|---:|---:|---:|---:|
| ResAD | 0.634 | 0.234 | 0.378 | 0.367 | 0.011 |
| Senna | 0.633 | 0.236 | 0.294 | 0.286 | 0.008 |
| Senna-2 | 0.597 | 0.225 | 0.288 | 0.283 | 0.005 |
| RAD-2 | 0.553 | 0.208 | 0.142 | 0.138 | 0.004 |

RAD-2 的开环轨迹精度也更好。这个结果很重要，因为它说明闭环 RL 并没有牺牲离线轨迹拟合，反而通过生成器-判别器协同改善了轨迹质量。

#### 3.12.4 训练管线消融

| ID | Gen IL pretrain | Gen RL | Gen IL fine-tune | Disc RL | CR↓ | Safety@1↑ | EP@1.0↑ |
|---:|---|---|---|---|---:|---:|---:|
| 1 | ✓ |  |  |  | 0.533 | 0.418 | 0.516 |
| 2 | ✓ | ✓ |  |  | 0.287 | 0.682 | 0.391 |
| 3 | ✓ | ✓ | ✓ |  | 0.403 | 0.555 | 0.527 |
| 4 | ✓ |  |  | ✓ | 0.337 | 0.615 | 0.728 |
| 5 | ✓ | ✓ | ✓ | ✓ | 0.234 | 0.730 | 0.736 |

解读：

- 只做 generator RL 能显著降 CR，但效率 EP@1.0 下滑，说明生成器偏保守。
- 加 IL fine-tune 可恢复部分效率，但安全性回退。
- 只训练 discriminator 也能提升安全和效率，说明重排序很有价值。
- generator + discriminator 联合优化最好。

#### 3.12.5 设计选择消融

关键结论：

- reward-variance clip filtering 提升 EP@1.0，从 0.662 到 0.728，同时安全基本不变。
- 判别器从 planning head 初始化显著优于随机初始化：CR 0.337 vs 0.426。
- TC-GRPO group size = 4 最均衡：CR 0.234，Safety@1 0.730。
- 推理时增加候选轨迹数 M 可以提升效率，说明判别器具备 inference-time scaling 能力。

推理时 M 的消融：

| M | CR↓ | Safety@1↑ | EP@1.0↑ |
|---:|---:|---:|---:|
| 8 | 0.275 | 0.693 | 0.667 |
| 16 | 0.266 | 0.689 | 0.711 |
| 32 | 0.234 | 0.730 | 0.736 |
| 64 | 0.252 | 0.699 | 0.816 |
| 128 | 0.234 | 0.719 | 0.814 |

更大的候选集提升效率，但安全指标并非单调。这反映了搜索空间扩大后，判别器可以找到更积极轨迹，但也可能引入更难排序的候选。

## 4. RAD 与 RAD-2 的系统对比

| 维度 | RAD | RAD-2 |
|---|---|---|
| 论文目标 | 证明 3DGS 闭环 RL 可提升端到端驾驶安全性 | 扩展到 diffusion planning，并解决高维 RL 不稳定和训练吞吐 |
| 策略输出 | 横向/纵向离散短时动作分布 | 多条连续未来轨迹候选 |
| 核心 planner | BEV + map/agent/image tokens + planning head | Diffusion generator + RL discriminator |
| RL 优化对象 | 直接优化动作 policy | 主要优化 discriminator，生成器用 OGO 结构化微调 |
| 仿真环境 | 3DGS photorealistic digital twin | BEV-Warp 高吞吐训练，3DGS 验证 |
| reward | 动态碰撞、静态碰撞、位置偏离、航向偏离 | TTC 安全 margin + Ego Progress |
| credit assignment | 横纵向 reward/value/advantage 拆分 | Temporally consistent rollout + group relative advantage |
| IL 的角色 | RL 交替训练中的人类对齐正则 | 生成器预训练和必要的 fine-tuning prior |
| 工程优势 | 视觉真实性强，闭环输入接近真实相机 | 训练吞吐高，适合大规模 RL 和 diffusion planner |
| 主要限制 | 3DGS 重建成本高，log-replay 交互有限 | 依赖 BEV 空间等变性，对非 BEV 架构适配有限 |

可以把两者理解成阶段性演进：

```text
RAD:
  从 IL 到 3DGS 闭环 RL
  重点是把真实视觉环境纳入 RL 训练闭环

RAD-2:
  从离散动作 RL 到扩散轨迹生成 + RL 重排序
  重点是让闭环 RL 能稳定、高吞吐地 scale
```

## 5. 与相关工作的关系

### 5.1 与 3D Gaussian Splatting / Street Gaussians

3D Gaussian Splatting 提供了实时、高质量新视角渲染的表示基础。Street Gaussians 等工作把它扩展到动态城市街景重建。RAD 借用这类能力，把真实驾驶片段变成可交互闭环环境。

区别是：很多 3DGS/NeRF 仿真工作偏重 closed-loop evaluation，RAD 进一步把它放入 RL training loop。

### 5.2 与 DiffusionDrive

DiffusionDrive 代表了扩散式端到端规划路线：用截断扩散从 anchor Gaussian 分布去噪到多模态驾驶动作分布。RAD-2 与这一路线高度相关，但关注点不同：

- DiffusionDrive 重点是 diffusion planner 本身如何高效建模多模态动作。
- RAD-2 重点是 diffusion planner 如何接受闭环负反馈，以及如何用 RL discriminator 稳定改进。

因此 RAD-2 可以看作对扩散式规划器的一种闭环 RL 外壳：生成器产生候选，判别器根据长期闭环效果排序。

### 5.3 与传统 RL + IL

早期 RL+IL 自动驾驶多在 CARLA 或结构化 BEV 输入上做，环境真实性、传感器输入和真实部署链路有差距。RAD 系列的区别是：

- RAD 强调真实场景重建和多视角视觉输入。
- RAD-2 强调扩散规划与高吞吐特征级闭环训练。

## 6. 工程复现与落地视角

### 6.1 如果复现 RAD，需要哪些模块

最小系统需要：

1. 多视角图像到 BEV 的感知主干。
2. map/agent token 监督训练数据。
3. 专家驾驶轨迹和 ego odometry。
4. 3DGS 场景重建管线，最好支持动态物体和路面几何约束。
5. 可根据 ego 新 pose 渲染状态的闭环环境。
6. 动态/静态碰撞检测，专家轨迹偏离检测。
7. PPO + GAE + IL 交替训练框架。
8. 多 worker rollout buffer。

真正难点不在 PPO 代码，而在数据和仿真：

- 如何批量重建 3DGS 场景。
- 如何保证 novel view 质量。
- 如何获得可靠 map/agent 标注。
- 如何定义碰撞、静态障碍和轨迹偏离。
- 如何避免环境噪声污染 RL。

### 6.2 如果复现 RAD-2，需要哪些模块

最小系统需要：

1. diffusion-based trajectory generator。
2. 可输出空间等变 BEV feature 的感知主干。
3. trajectory discriminator，能融合轨迹、BEV、map、agent 信息。
4. BEV-Warp 环境，支持基于 pose delta 的 feature warp。
5. iLQR 或等价 controller 跟踪候选轨迹。
6. TTC、ego progress 等闭环 reward 计算。
7. trajectory reuse / latched execution。
8. TC-GRPO 训练判别器。
9. OGO 生成器纵向优化和 supervised fine-tuning。
10. replay buffer、clip filtering、scenario balancing。

RAD-2 的工程难点在于系统耦合：

- BEV feature 是否真的等变。
- warp 后感知头是否还能稳定输出。
- generator/discriminator 更新频率是否平衡。
- group sampling 是否产生足够 reward 方差。
- 判别器是否过拟合仿真 reward，而不是学到通用轨迹质量。

### 6.3 对自动驾驶工程的启示

1. **闭环训练比开环指标更重要**。开环 ADE/FDE 好，不代表闭环安全；RAD 系列用 collision、TTC、progress 等闭环指标直接优化行为后果。
2. **IL 不应被简单抛弃**。RAD 和 RAD-2 都保留 IL 作为人类先验和稳定器。
3. **RL 的对象要选对**。RAD-2 的关键洞察是：高维扩散轨迹不适合被稀疏 reward 直接优化，先优化低维 discriminator 更稳定。
4. **仿真吞吐决定 RL 上限**。RAD 的 3DGS 环境真实但重；RAD-2 的 BEV-Warp 更适合大规模迭代。
5. **reward 不只是打分，也是系统设计语言**。RAD 的四类事件和 RAD-2 的 TTC/EP，实际上定义了论文所追求的驾驶风格。

## 7. 批判性评价

### 7.1 最值得肯定的点

RAD 的价值在于打通了真实重建视觉环境和端到端 RL。它把“自动驾驶端到端模型只会开环模仿”的问题拉回到闭环交互本质上。

RAD-2 的价值在于把扩散模型和 RL 的矛盾拆开处理：生成器负责表达复杂行为分布，判别器负责用闭环结果学习偏好。这比直接 RL fine-tune diffusion trajectory 更稳健，也更符合实际系统工程。

### 7.2 需要谨慎的点

第一，环境真实性仍不等于真实世界交互。RAD 中其他交通参与者 log-replay，不会响应 ego 的异常行为；RAD-2 的 BEV-Warp 更是 feature-level 近似。这些环境能提供有效训练信号，但不能完全替代真实交通博弈。

第二，reward 定义可能塑造保守策略。RAD 用碰撞/偏离终止，RAD-2 用 TTC 和 progress 窗口，都会让模型偏向某种预设风格。若 reward 与真实安全/舒适/法规目标不一致，模型会优化错误代理。

第三，benchmark 复现门槛高。论文给出大量内部数据和闭环场景，但公开代码、数据、重建环境与训练配置是否足够复现同等结果，需要实际验证。

第四，BEV-Warp 的适用面有限。它非常适合 BEV-centric pipeline，但对于直接从 raw camera 到 planning 的视觉语言/端到端大模型架构，可能需要 latent world model 或可微渲染替代。

第五，真实车部署证据仍需要更多细节。RAD-2 摘要提到 real-world deployment 改善 perceived safety and smoothness，但论文主体公开细节相对有限。工程采信时需要更多道路测试协议、接管率、ODD 覆盖和失效案例。

## 8. 这两篇论文对“自动驾驶中的扩散模型”的意义

如果从扩散模型专题看，RAD 本身不是扩散 planner 论文，它更像是为后续扩散规划闭环训练提供 RL 环境范式。RAD-2 才是直接把扩散生成器纳入闭环 RL 框架的工作。

RAD-2 给扩散规划提出了一个重要方向：

```text
不要只问：如何生成更多样、更像人的轨迹？
还要问：如何在闭环后果中筛掉危险轨迹，并让生成分布逐步远离低质量区域？
```

因此，RAD-2 相比普通 diffusion planning 的关键增量是 **closed-loop negative feedback**。扩散模型负责“想象候选未来”，RL 判别器负责“根据真实交互后果评估这些未来”。

这对后续研究有几个启发：

- 扩散 planner 可以不直接端到端 RL fine-tune，而是先引入可学习 scorer/ranker。
- inference-time scaling 在自动驾驶规划中是可行的：候选数越多，判别器有机会挑出更好方案。
- 生成器优化可以从低维稳定子空间开始，例如纵向速度/进度，而不是一次性改完整空间轨迹。
- 闭环仿真环境的形式会直接决定可训练规模。3DGS、BEV-Warp、世界模型可能会长期共存。

## 9. 推荐阅读顺序

如果要高效理解这组工作，建议顺序如下：

1. 先读 RAD 的 Introduction 和 Method，理解 IL 的 open-loop gap、3DGS 环境、RL+IL 联合训练。
2. 再读 RAD 的 Reward Modeling、Policy Optimization、Auxiliary Objective，理解它怎样把自动驾驶风险转成可训练信号。
3. 看 RAD 实验表 1-4，重点理解 IL/RL/RL+IL 的 trade-off。
4. 再读 RAD-2 的 Introduction，理解为什么 diffusion planner 不能直接套 RAD 式 RL。
5. 深读 RAD-2 的 Generator-Discriminator、BEV-Warp、TC-GRPO、OGO。
6. 看 RAD-2 表 1-4 和表 5-10，理解各个工程设计为什么必要。
7. 最后对照两篇 limitations，判断未来真正可扩展方向。

## 10. 关键词表

| 术语 | 含义 |
|---|---|
| IL | Imitation Learning，用专家示范监督模型 |
| RL | Reinforcement Learning，用闭环 reward 优化策略 |
| 3DGS | 3D Gaussian Splatting，高质量新视角渲染表示 |
| BEV | Bird's-Eye View，鸟瞰空间表示 |
| Open-loop gap | 开环训练/评测与闭环部署之间的分布差距 |
| Causal confusion | 模型学到相关性捷径，而不是真正因果因素 |
| PPO | Proximal Policy Optimization，常用策略梯度算法 |
| GAE | Generalized Advantage Estimation，优势估计方法 |
| TC-GRPO | RAD-2 的 temporally consistent group relative policy optimization |
| OGO | On-policy Generator Optimization，RAD-2 中基于闭环反馈优化生成器 |
| TTC | Time-To-Collision，碰撞时间裕度 |
| EP | Ego Progress，自车路线进度 |
| AF-CR | At-Fault Collision Rate，自车责任碰撞率 |

## 11. 参考资料与链接

本节列出本文档使用的主要公开资料。PDF 内容以本地文件为准；外部链接用于核对论文状态、项目实现与相关背景。

### 核心论文与项目

- RAD arXiv: <https://arxiv.org/abs/2502.13144>
- RAD OpenReview NeurIPS 2025 页面: <https://openreview.net/forum?id=9V3crVSPH7>
- RAD 项目页: <https://hgao-cv.github.io/RAD/>
- RAD / RAD-2 GitHub 仓库: <https://github.com/hustvl/RAD>
- RAD-2 arXiv: <https://arxiv.org/abs/2604.15308>
- RAD-2 项目页: <https://hgao-cv.github.io/RAD-2/>

### 背景论文

- 3D Gaussian Splatting for Real-Time Radiance Field Rendering: <https://arxiv.org/abs/2308.04079>
- Street Gaussians for Modeling Dynamic Urban Scenes: <https://arxiv.org/abs/2401.01339>
- DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving: <https://arxiv.org/abs/2411.15139>

## 12. 最终判断

RAD 系列的主线非常清晰：端到端自动驾驶不能只停留在开环模仿，必须进入闭环交互；扩散模型不能只生成“像专家”的轨迹，还要能利用闭环失败信号学会“避开坏轨迹”。

RAD 给出的是闭环 RL 的可行性证明：用 3DGS 把真实场景转成可交互环境，再用 RL+IL 让策略安全性显著提升。

RAD-2 给出的是可扩展路径：用 BEV-Warp 提高仿真吞吐，用生成器-判别器解耦高维轨迹生成和低维 reward 优化，用 TC-GRPO 和 OGO 稳定训练扩散规划器。

从研究价值看，RAD-2 更接近未来方向，因为它把 diffusion planning、closed-loop RL、inference-time scaling 和高吞吐仿真连接起来。但从工程可信度看，RAD 的 3DGS 视觉闭环仍然是重要基线，因为它更接近真实传感器输入。两者合起来，构成了一条有说服力的路线：**先用高保真重建证明闭环 RL 有效，再用特征级仿真和生成器-判别器框架把它规模化。**
