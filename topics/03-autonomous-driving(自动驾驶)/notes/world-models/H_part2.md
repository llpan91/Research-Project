# H_经典基础 论文分析报告（Part 2）

---

## DreamerV2 (Hafner 2021) — Mastering Atari with Discrete World Models

**机构/会议**：Google Research / DeepMind / University of Toronto；ICLR 2021

**核心贡献**：
- 首个在 Atari 55款游戏基准上、纯粹在世界模型内部学习行为便达到人类水平的强化学习智能体
- 将世界模型的潜在表示从高斯连续变量改为**离散类别变量**（32×32 categoricals），并引入 KL Balancing 技术稳定训练
- 以单 GPU、单环境实例超越 Rainbow 与 IQN 等顶级无模型算法

**方法要点**：
世界模型基于 RSSM（循环状态空间模型），含 CNN 图像编码器、GRU 循环模型、离散后验/先验状态预测器及图像/奖励/折扣预测头。行为学习阶段在潜在空间中展开 H=15 步想象轨迹，使用 Actor-Critic + λ-return 优化策略，世界模型与策略独立训练。

**主要结果**：
Atari 55任务上，200M 帧内 Gamer 归一化中位分超过 Rainbow 和 IQN；在人形机器人连续控制任务中仅凭像素输入也能学会站立和行走。

**优缺点**：
优点：离散潜在表示提升训练稳定性；无需前瞻搜索即可规划；样本效率优于无模型方法。缺点：需要 200M 帧仍属数据密集；模型精度对长期预测存在累积误差；仅在有限领域验证泛化能力。

---

## DreamerV3 (Hafner 2023) — Mastering Diverse Domains through World Models

**机构/会议**：Google DeepMind / University of Toronto；arXiv 2023

**核心贡献**：
- 提出首个**单一固定超参数配置**在 150+ 个跨领域任务（Atari、Minecraft、DMLab、Control Suite 等）上均超越专门调优算法的通用世界模型智能体
- 引入 **symlog 变换**、**free bits**、**分类 Critic** 等鲁棒性技术，彻底解决跨域奖励量级不一致问题
- 首个无需人类数据即可从零在 Minecraft 中收集钻石的算法

**方法要点**：
沿用 RSSM 架构，但损失函数拆分为预测损失、动态损失与表示损失三项（权重 1/1/0.1）；Critic 建模为指数间隔分箱的分类分布；Actor 使用归一化的百分位回报缩放梯度；symlog 对输入与目标统一压缩，配合 free bits 防止 KL 崩溃。

**主要结果**：
57 个 Atari 任务中位分超 PPO/Rainbow/MuZero；Minecraft 钻石收集成功率约 10%；DMLab、ProcGen、BSuite 等多域均优于调优专家算法。

**优缺点**：
优点：真正开箱即用的跨域通用性；鲁棒性技术体系完整。缺点：计算开销仍较大；Minecraft 成功率绝对值偏低；对极稀疏奖励的长时序探索能力有限。

---

## IRIS (Micheli 2023) — Transformers are Sample-Efficient World Models

**机构/会议**：University of Geneva；ICLR 2023

**核心贡献**：
- 将 **离散自编码器 + GPT-like 自回归 Transformer** 组合用作世界模型，首次证明 Transformer 在样本高效 RL 中的有效性
- 仅 2 小时真实游戏时长（Atari 100k benchmark）即达到人类归一化均分 1.046，10/26 款游戏超越人类
- 将环境动态建模转化为**序列建模问题**，彻底摆脱 RNN 结构

**方法要点**：
离散 VQ 自编码器将图像压缩为 K 个 token；GPT 式 Transformer G 对帧 token 与动作 token 的交错序列建模，自回归预测下一帧 token、奖励及终止信号；策略 π 完全在想象 MDP 内训练，RL 目标借用 DreamerV2 的 Actor-Critic 框架。

**主要结果**：
Atari 100k 基准上均分 1.046（HNS），优于 SimPLe、EfficientZero（无前瞻搜索）等方法，并在 26 款游戏中 10 款超越人类水平。

**优缺点**：
优点：Transformer 长程上下文建模能力强；样本效率极高；无需前瞻搜索。缺点：自回归生成逐 token 展开，推理速度较慢；序列长度随时间增长带来计算瓶颈；仅在离散动作 Atari 上验证，连续控制扩展性未知。
