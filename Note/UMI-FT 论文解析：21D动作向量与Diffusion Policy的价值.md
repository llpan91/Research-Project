# UMI-FT 论文解析：21D 动作向量与 Diffusion Policy 的实际价值

> 论文：*In-the-Wild Compliant Manipulation with UMI-FT*
> arXiv: 2601.09988v1, 15 Jan 2026

---

## 一、为什么输出 21 维动作向量（含 9D 位姿），而不是 6DoF？

### 1. 21 维向量的具体构成

论文 Section IV-B (Output Decoding) 明确给出了分解：

| 分量 | 维度 | 说明 |
|------|------|------|
| Reference Pose | **9D** | 3D 平移 + 6D 旋转（旋转矩阵前两行） |
| Virtual Target Pose | **9D** | 同上格式，柔顺控制器的实际设定目标 |
| Stiffness Value | **1D** | 标量，编码机械臂刚度矩阵 |
| Gripper Width | **1D** | 期望夹爪宽度 |
| Grasp Force | **1D** | 期望夹持力 |
| **总计** | **21D** | |

---

### 2. 为什么旋转用 6D 而不是 3D（欧拉角）或 4D（四元数）？

这是遵循 Diffusion Policy [30] 的惯例，其数学根源来自 Zhou et al. 的连续旋转表示理论：

- **欧拉角 (3D)**：存在**万向节锁 (Gimbal Lock)**，且角度空间存在 ±π 跳变等不连续性，对神经网络的回归学习非常不友好
- **四元数 (4D)**：存在**对径等价问题** —— `q` 和 `-q` 表示同一旋转，导致损失函数的拓扑空间不连续，网络学习时会在两个等价点间震荡
- **旋转矩阵前两行 (6D)**：这是一种**拓扑连续的表示**。取 3×3 旋转矩阵的前两行（6 个数），推理时通过 Gram-Schmidt 正交化恢复第三行，重建完整旋转矩阵。这种表示在 SO(3) 到 ℝ⁶ 的映射上是连续的，对神经网络回归最友好

> **结论**：9D = 3D position + 6D continuous rotation，本质上就是 6DoF 位姿的一种**对学习更友好的参数化形式**，不是增加了新的信息维度。

---

### 3. 为什么要输出「两个」位姿（Reference Pose + Virtual Target Pose）？

这是 UMI-FT 最核心的设计之一，来源于 ACP (Adaptive Compliance Policy) [8] 框架。两个位姿配合 stiffness 标量，共同编码了一个**时变虚拟弹簧模型**：

```
                   Stiffness (k)
Reference Pose ←────/\/\/\/────→ Virtual Target Pose
  (弹簧一端)           弹簧          (弹簧另一端 = 控制器设定目标)
```

- **Reference Pose（参考位姿）**：策略认为末端执行器「应该在的位置」，即名义轨迹点
- **Virtual Target Pose（虚拟目标位姿）**：发送给底层柔顺控制器的**实际设定目标**，它与参考位姿之间的偏移量编码了期望施加的力
- **Stiffness（刚度）**：弹簧常数 `k`，决定偏移量到力的映射关系

**核心逻辑**：

```
F = k × (Virtual Target - Reference Pose)
```

**具体例子（白板擦除任务）**：

- 策略希望擦板时施加 10N 向下力
- 不直接输出"10N"，而是输出 Reference Pose 在白板表面、Virtual Target 穿透白板表面几毫米、Stiffness 设为合适的值
- 底层柔顺控制器追踪 Virtual Target，遇到白板阻挡后自然产生弹簧力，实现柔顺接触

**为什么不直接输出力？**

因为直接预测力并执行需要精确的力控制器，且纯力控制在自由空间（未接触时）没有意义。用虚拟弹簧模型，**自由空间走位置、接触后走力**，天然统一了两种行为模式。

---

## 二、Diffusion Policy 在这个工作中的具体、实际价值

### 1. 系统架构中的定位

论文 Fig. 3 展示了三层控制回路：

```
┌────────────────────────────────────────────────────────────┐
│  Learned Visuomotor Policy (Diffusion Policy)    @ 1 Hz    │  ← 最慢，做"决策"
│  输入: RGB, Depth, F/T (32帧), Proprioception              │
│  输出: 21D 动作向量                                          │
├────────────────────────────────────────────────────────────┤
│  Arm Compliance Controller (Admittance)          @ 500 Hz  │  ← 中层，做"柔顺执行"
│  Grasp Force Controller                          @ 30 Hz   │  ← 底层，做"夹持力调节"
└────────────────────────────────────────────────────────────┘
```

Diffusion Policy 是**最上层的「大脑」**，以 1Hz 的频率生成包含位姿、刚度、夹持力的完整动作指令；底层两个基于模型的控制器以高频率做实际的物理交互。

---

### 2. 为什么必须用 Diffusion Policy 而非普通回归网络？

#### (a) 动作的多模态性（Multi-modality）

在灯泡插入任务中，当针脚需要对齐插槽时，机器人可以顺时针或逆时针旋转搜索——两种方向都是有效的。普通回归网络（MSE loss）会把两个模态取平均，输出一个**介于两者之间的无效动作**（比如不旋转）。Diffusion Policy 通过去噪过程对完整的动作分布建模，可以**采样到任意一个有效模态**而不会模态坍缩。

#### (b) 时序动作块预测（Action Chunk Prediction）

Diffusion Policy 不是逐帧预测单个动作，而是一次生成**一段未来动作序列**。这对柔顺操作至关重要：

- 擦白板时需要持续稳定的接触力轨迹
- 穿刺西葫芦时需要平滑的力递增过程
- 时序一致的动作块天然避免了帧间抖动

#### (c) 灵活的多模态条件输入

Diffusion Policy 本质是条件生成模型，可以自然地接受**任意多模态输入**作为条件。论文中的观测编码流程：

```
RGB (2帧)          → ViT-B/32      ──┐
Depth              → ViT-B/32      ──┤
Left F/T (32帧)    → CausalConv    ──┤→ Self-Attention Fusion → Diffusion → 21D Action
Right F/T (32帧)   → CausalConv    ──┤
Proprioception     ─────────────────┘
```

各个模态的 token 经过 Self-Attention 融合后，作为条件送入扩散过程。这种架构天然支持 UMI-FT 新增的力/力矩模态，不需要重新设计网络结构。

---

### 3. 实验中的直接证据

论文所有 baseline 对比中，Diffusion Policy 都是**固定的策略骨干**，变量是输入模态和输出执行方式：

| 方法 | 策略骨干 | 力观测 | 柔顺控制 | 白板擦除 | 灯泡插入 |
|------|---------|--------|---------|---------|---------|
| **Ours (ACP)** | Diffusion Policy | CoinFT | 有 | **92%** | **95%** |
| DP w/ F | Diffusion Policy | CoinFT | 无 | 28% | 60% |
| DP | Diffusion Policy | 无 | 无 | 16% | 0% |
| DP w/ CM | Diffusion Policy | 接触麦克风 | 无 | 0% | 20% |

- 即使在没有柔顺控制的 baseline (DP w/ F) 中，Diffusion Policy + 力观测已经能完成部分任务（灯泡 60%、西葫芦 70%），说明 Diffusion Policy 本身确实学到了**力感知的操作策略**
- 完整系统 (ACP) 的大幅提升来自**将 Diffusion Policy 的高层决策与底层柔顺控制解耦**

---

### 4. 总结：Diffusion Policy 的不可替代价值

| 价值 | 具体体现 |
|------|---------|
| **多模态动作分布建模** | 避免接触丰富任务中的模态平均问题 |
| **时序一致的动作块** | 保证力轨迹平滑，避免抖动 |
| **灵活的多模态条件输入** | 天然融合 RGB + Depth + F/T + Proprioception |
| **高层-低层解耦** | 1Hz 做决策 + 500Hz 做柔顺执行，既有智能又有反应速度 |

> Diffusion Policy 在这个工作中不是「可以替换的模块」，而是整个系统的**决策核心**——它以低频率输出「做什么」（21D 向量），底层控制器以高频率解决「怎么做」（柔顺接触）。
