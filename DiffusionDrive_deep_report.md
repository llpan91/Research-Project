# DiffusionDrive 深度报告

> 论文：**DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving**  
> 会议：CVPR 2025  
> 代码：<https://github.com/hustvl/DiffusionDrive>  
> 论文页：<https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html>  
> PDF：<https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.pdf>  
> arXiv：<https://arxiv.org/abs/2411.15139>  
> 补充材料：<https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liao_DiffusionDrive_Truncated_Diffusion_CVPR_2025_supplemental.pdf>  
> 本报告生成日期：2026-05-07

---

## 0. 核心结论

DiffusionDrive 的核心贡献不是简单地“把 diffusion policy 用到自动驾驶规划”，而是把扩散模型改造成一个更适合实时驾驶决策的生成式规划器：

1. **从纯噪声起步改为从驾驶先验起步**  
   普通 diffusion 从标准高斯噪声开始逐步去噪。DiffusionDrive 从少量轨迹 anchor 附近的高斯分布开始，只在扩散链后段做采样和去噪，因此称为 **truncated diffusion**。

2. **用少量 anchor 保留多模态先验，但不把规划离散化**  
   方法只需要约 20 条 anchor trajectory，就能覆盖常见驾驶模式。anchor 不是最终答案，而是扩散生成的初始化中心，模型仍然输出连续轨迹。

3. **用 2 步去噪达到接近实时推理**  
   论文报告在 NAVSIM navtest 上达到 **88.1 PDMS、45 FPS**，相比普通 diffusion policy 的 20 步去噪显著加速。

4. **显著缓解 mode collapse**  
   从不同 anchor 附近开始采样，使候选轨迹天然分散在不同驾驶模式附近。论文的 roadmap 消融中，多样性指标从 vanilla diffusion 的 **11%** 提升到 truncated diffusion / DiffusionDrive 的约 **70%+**。

5. **规划 decoder 的场景交互设计同样关键**  
   论文不是只改 noise schedule，而是设计了 diffusion decoder，使候选轨迹能与 BEV/PV 特征、agent query、map query 交互。消融显示 spatial cross-attention 对最终规划质量尤其重要。

一句话总结：**DiffusionDrive 把 diffusion 的连续多模态生成能力、anchor 的驾驶行为先验、transformer decoder 的场景交互能力组合起来，形成了一个实时、多样、精度较高的端到端规划模块。**

---

## 1. 论文要解决的问题

### 1.1 端到端自动驾驶规划的多解性

自动驾驶规划不是一个单解回归问题。同一个场景下，合理动作可能有多种：

- 在路口可以直行、左转、右转；
- 遇到慢车可以跟车、变道、减速；
- 前方障碍物可绕行，也可停车等待；
- 交互场景中，ego 车辆行为取决于其他交通参与者的未来响应。

因此，一个只输出单条轨迹的规划网络天然会有表达瓶颈。训练时如果用 L1/L2 loss 拟合人类轨迹，模型容易学到“平均行为”。在多模态场景中，平均轨迹可能既不对应左转，也不对应直行，而是落在不可驾驶区域。

### 1.2 现有路线的典型取舍

论文主要讨论了三类相关路线。

#### 路线 A：单轨迹回归

代表方法包括 Transfuser、UniAD、VAD 等端到端规划框架中的常见 planning head。其基本形式是：

```text
sensor features -> planning head -> one future trajectory
```

优点：

- 结构简单；
- 推理快；
- 易于训练；
- 与 perception/prediction/map 模块集成方便。

缺点：

- 无法自然表达多种可行驾驶模式；
- 对复杂交互或岔路场景不够稳健；
- 容易受 imitation learning 中 one-to-many 标签问题影响。

#### 路线 B：大规模 trajectory vocabulary / anchor set

代表思路是预先构建大量候选轨迹，然后网络做分类、打分或 refine。某些方法会使用 4096、8192 甚至更多候选轨迹。

优点：

- 多模态表达能力比单轨迹回归更强；
- 候选轨迹可带有驾驶先验；
- 打分式选择较容易与规则、代价函数结合。

缺点：

- action space 被离散化；
- anchor 不覆盖的行为难以生成；
- 大 vocabulary 带来存储和计算开销；
- 过度依赖候选库质量。

#### 路线 C：vanilla diffusion policy

扩散模型理论上适合建模多模态连续分布。用于规划时，可以把 future trajectory 当作待生成样本，从高斯噪声开始逐步去噪。

优点：

- 连续动作空间；
- 多模态生成能力强；
- 不必把动作完全离散成 vocabulary。

缺点：

- 普通 diffusion 推理步数多，实时性差；
- 多次采样可能收敛到相似模式，出现 mode collapse；
- 从纯噪声生成驾驶轨迹没有充分利用驾驶行为先验。

DiffusionDrive 的设计正是对路线 B 和路线 C 的折中：**用少量 anchor 提供先验，用 truncated diffusion 保持连续生成。**

---

## 2. 方法总览

DiffusionDrive 的整体结构可以抽象为：

```text
多传感器输入 / 相机输入
        |
感知 backbone 与场景编码器
        |
BEV/PV feature + agent query + map query
        |
少量 anchor trajectory
        |
anchored Gaussian 采样
        |
2-step truncated diffusion denoising
        |
多条候选规划轨迹 + confidence score
```

关键模块有三个：

1. **Truncated diffusion policy**  
   把扩散过程截断，只从轨迹先验附近开始去噪。

2. **Anchored Gaussian distribution**  
   使用 K-Means 得到的轨迹 anchor 作为初始分布中心，而不是从零均值高斯噪声开始。

3. **Cascade diffusion decoder**  
   让 noisy trajectory samples 与空间特征、动态目标、地图信息进行交互，并逐层 refine。

---

## 3. 普通 diffusion policy 为什么不够好

为了理解 DiffusionDrive，先看普通 diffusion 在规划任务中的问题。

### 3.1 标准扩散过程

设真实未来轨迹为：

$$
\tau = \{(x_t, y_t)\}_{t=1}^{T_f}
$$

标准 diffusion 的正向加噪过程通常可写为：

$$
q(\tau_i|\tau_0)=\mathcal{N}(\sqrt{\bar{\alpha}_i}\tau_0, (1-\bar{\alpha}_i)I)
$$

当扩散步数足够大时，样本逐渐接近标准高斯噪声：

$$
\tau_T \sim \mathcal{N}(0, I)
$$

反向过程则学习：

$$
p_\theta(\tau_{i-1}|\tau_i, z)
$$

其中 \(z\) 是场景条件特征。

### 3.2 在规划任务中的两个主要矛盾

#### 矛盾一：速度

图像生成可以接受几十步甚至上百步采样，但自动驾驶规划需要低延迟。论文在 roadmap 中比较了普通 diffusion policy：20 步 DDIM 去噪时只有约 **7 FPS**，很难用于高频规划闭环。

#### 矛盾二：多样性退化

普通 diffusion 虽然从不同噪声采样，但如果场景条件强、训练目标偏向唯一人类轨迹，多个采样可能最终被 denoising 网络拉到相似区域。论文称这种现象为 mode collapse。

这对规划特别不利：我们希望候选轨迹覆盖不同意图，而不是输出多条几乎重合的轨迹。

---

## 4. Truncated Diffusion 的核心思想

### 4.1 直觉解释

驾驶轨迹和图像不同。图像生成要从随机噪声中创造复杂纹理、结构和语义；驾驶轨迹通常是低维、有强约束、有强先验的曲线。

常见轨迹模式并不多：

- 保持车道；
- 轻微左偏或右偏；
- 左转；
- 右转；
- 减速停车；
- 加速通过；
- 换道；
- 绕障。

因此没有必要从完全随机的高斯噪声开始。更合理的做法是：

```text
先找到一些常见驾驶模式的中心轨迹
再在这些中心附近加少量噪声
最后让网络根据场景把它们修正成最终轨迹
```

这就是 DiffusionDrive 的 truncated diffusion。

### 4.2 从 anchor 附近开始

论文用训练集中的 ego future trajectories 做 K-Means 聚类，得到 \(K\) 条 anchor：

$$
\mathcal{A} = \{a_1, a_2, \ldots, a_K\}
$$

每个 anchor 是一条完整的未来轨迹，而不是单个点。

训练或推理时，不再从：

$$
\tau_T \sim \mathcal{N}(0,I)
$$

开始，而是从某个 anchor 附近采样：

$$
\tau_i^k = \sqrt{\bar{\alpha}_i}a_k + \sqrt{1-\bar{\alpha}_i}\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
$$

这里的 \(i\) 不是完整扩散链末端的大步数，而是截断后的较小噪声等级。直观上，样本仍然保留 anchor 的大体形状，只带有适度扰动。

### 4.3 为什么叫 truncated

普通 diffusion 的反向过程是：

```text
pure noise -> ... -> trajectory
```

DiffusionDrive 的反向过程是：

```text
noisy anchor -> trajectory
```

也就是跳过从纯噪声到轨迹粗结构的大段生成过程，只学习后半段 refinement。它不是简单减少 DDIM steps，而是改变初始分布，使少步采样仍有意义。

### 4.4 anchor 不是 vocabulary

这是理解论文的关键。

在 vocabulary 方法中，anchor 往往是候选答案。模型的主要任务是选择或轻量修正。若真实轨迹不在 vocabulary 覆盖范围内，模型会受限。

在 DiffusionDrive 中，anchor 是生成过程的起点。模型输出的是连续 offset / denoised trajectory，因此可以离开 anchor 本身。换言之：

```text
vocabulary 方法：anchor ≈ action candidate
DiffusionDrive：anchor ≈ generative prior center
```

这使它同时获得：

- anchor 的多模态先验；
- diffusion 的连续生成能力；
- 较小候选数量下的高覆盖能力。

---

## 5. 训练机制

### 5.1 输入与输出

对每个场景，模型接收：

- 场景特征 \(z\)，来自传感器 backbone、BEV/PV encoder、object/map queries；
- 一组 noisy trajectory samples；
- diffusion timestep embedding。

模型输出：

- 每条候选轨迹的 denoised trajectory；
- 每条候选轨迹的 confidence score。

可抽象为：

$$
\{\hat{\tau}_k, \hat{s}_k\}_{k=1}^{K}
= f_\theta(\{\tau_i^k\}_{k=1}^{K}, z, i)
$$

其中 \(\hat{\tau}_k\) 是第 \(k\) 个候选的预测轨迹，\(\hat{s}_k\) 是对应置信度。

### 5.2 positive / negative assignment

训练时，需要告诉模型哪一个 anchor 对应当前 ground truth。论文采用最近距离匹配：

$$
k^* = \arg\min_k d(a_k, \tau_{gt})
$$

其中 \(d(\cdot)\) 可以理解为轨迹级距离，例如 waypoint 平均 L2。

匹配到的 anchor 作为 positive，其余 anchor 作为 negative。

这种做法的作用是：

- positive anchor 学习精确轨迹重建；
- classifier / score head 学习当前场景下哪个模式更合理；
- negative anchors 不被强行回归到 ground truth，避免所有模式塌缩到同一轨迹。

### 5.3 损失函数

论文训练目标由两部分组成：

1. **轨迹重建损失**  
   对 matched positive trajectory 进行 L1 监督：

   $$
   \mathcal{L}_{reg}=||\hat{\tau}_{k^*}-\tau_{gt}||_1
   $$

2. **分类 / 置信度损失**  
   使用 BCE 监督每个候选轨迹是否为 positive：

   $$
   \mathcal{L}_{cls}=\text{BCE}(\hat{s}_k, y_k)
   $$

整体可写为：

$$
\mathcal{L}=\mathcal{L}_{reg}+\lambda\mathcal{L}_{cls}
$$

这里 \(\lambda\) 控制分类项权重。

### 5.4 训练阶段的重要含义

这个训练方式有两个值得注意的后果。

第一，模型不是被要求“所有候选都拟合 GT”。这能保护多模态候选的分化。

第二，score head 学到的是模式选择能力。推理时 top-1 轨迹由 confidence 选出，top-k 则可用于下游规划、安全评估或规则重排序。

---

## 6. 推理流程

推理时没有 ground truth，流程如下：

1. 从固定 anchor set 中取 \(K\) 条 anchor；
2. 对每条 anchor 采样噪声，得到 noisy anchor trajectories；
3. 将 noisy trajectories、场景特征、timestep embedding 输入 diffusion decoder；
4. 进行少量 DDIM denoising，论文主设置为 2 步；
5. 得到多条候选轨迹和对应 confidence；
6. 选择最高分轨迹作为最终规划输出，也可以保留 top-k。

伪代码如下：

```python
anchors = load_kmeans_trajectory_anchors()
features = scene_encoder(sensor_inputs)

samples = []
for anchor in anchors:
    noise = randn_like(anchor)
    noisy_traj = sqrt(alpha_bar[t]) * anchor + sqrt(1 - alpha_bar[t]) * noise
    samples.append(noisy_traj)

traj_candidates = samples
for t in ddim_steps:  # e.g. 2 steps
    traj_candidates, scores = diffusion_decoder(
        traj_candidates,
        features,
        timestep=t,
    )

final_index = argmax(scores)
final_traj = traj_candidates[final_index]
```

工程上需要注意：这里的 anchors 可以批量化，decoder 一次处理 \(K\) 条候选，所以并不是对每条轨迹单独跑完整模型。

---

## 7. Diffusion Decoder 架构拆解

### 7.1 为什么不能只用简单 MLP 或 UNet

轨迹规划不是单纯的轨迹形状生成。最终轨迹必须与场景交互：

- 避开车辆、行人、骑行者；
- 遵守道路拓扑；
- 跟随 lane centerline；
- 避免驶出可行驶区域；
- 考虑交通灯、停止线、前车速度等信息。

如果只用一个 MLP 从场景 embedding 回归轨迹，场景信息可能过度压缩。如果只用普通 UNet denoise 低维轨迹，空间交互又不充分。

因此论文设计了 cascade diffusion decoder，让每个 trajectory query 在去噪过程中主动从场景特征中取信息。

### 7.2 输入表示

decoder 的主要输入包括：

- noisy trajectory samples；
- diffusion timestep embedding；
- BEV 或 PV spatial feature；
- agent/object queries；
- map queries；
- trajectory query embeddings。

在 NAVSIM 实现中，论文主要基于 Transfuser，使用 BEV feature 和 object queries。  
在 nuScenes 实现中，论文基于 SparseDrive，使用 object queries、map queries 和 perspective-view image features。

### 7.3 Spatial cross-attention

spatial cross-attention 的作用是让候选轨迹根据自身 waypoint 位置去读取空间特征。

可以理解为：

```text
每条 noisy trajectory 有一组未来坐标点
decoder 根据这些坐标点在 BEV/PV feature 上采样或注意力聚合
得到与该候选轨迹路径相关的局部场景证据
```

这对规划非常重要。例如一条候选轨迹向左变道，则它需要读取左侧车道、左侧车辆、边界、可行驶区域等信息；一条直行轨迹则需要关注前方车道和前车。

消融结果显示，spatial cross-attention 是 DiffusionDrive 中最关键的 decoder 组件之一。

### 7.4 Agent / map cross-attention

agent cross-attention 让轨迹候选与动态目标 query 交互；map cross-attention 让轨迹候选与地图元素 query 交互。

这类交互可以补充 spatial feature：

- 动态目标 query 更适合表达目标实例级属性，例如位置、速度、类别；
- map query 更适合表达拓扑和矢量化道路结构；
- spatial feature 更适合表达 dense environment context。

论文结果显示，agent/map cross-attention 有增益，但单独依赖它们不如空间特征交互充分。

### 7.5 Cascade refinement

decoder 采用多层级 cascade refinement。每一层接收上一层轨迹候选，再输出 refined offset 和 score。

这种设计适合规划：

```text
粗轨迹模式 -> 场景交互 -> 修正 -> 再交互 -> 再修正
```

与一次性回归相比，cascade 允许模型逐步消除 noisy anchor 与真实轨迹之间的差异。

---

## 8. 与 anchor vocabulary 方法的本质区别

DiffusionDrive 和大规模 vocabulary 方法都使用 anchor，因此容易被误解为同一类方法。实际差异很大。

| 维度 | 大 vocabulary / anchor 方法 | DiffusionDrive |
|---|---|---|
| anchor 数量 | 通常很多，例如 4096 / 8192 | 主实验约 20 |
| anchor 角色 | 候选答案或离散动作 | 生成分布中心 |
| 输出空间 | 偏离散，依赖候选库覆盖 | 连续，可大幅修正 anchor |
| 多模态来源 | 候选库枚举 | anchor prior + noisy sampling |
| 推理负担 | 大候选打分或后处理 | 少量候选 + 少步去噪 |
| OOD 风险 | vocabulary 不覆盖则受限 | 仍依赖 anchor，但可连续外推 |

关键区别可以概括为：

```text
vocabulary 方法关心“选哪条”
DiffusionDrive 关心“从哪类模式附近生成”
```

---

## 9. 实验设置

论文主要在两个 benchmark 上验证：

1. **NAVSIM**
   - 更强调 closed-loop-style evaluation；
   - 使用 PDMS 等综合指标；
   - 主实验包含 camera + LiDAR；
   - 报告 FPS，能反映实时性。

2. **nuScenes**
   - 经典 open-loop planning benchmark；
   - 使用 L2 error 和 collision rate；
   - 与 UniAD、VAD、SparseDrive 等方法比较。

### 9.1 NAVSIM 指标理解

论文报告的核心指标包括：

- **PDMS**：PDM Score，综合规划质量指标；
- **NC**：No Collision；
- **DAC**：Drivable Area Compliance；
- **TTC**：Time To Collision；
- **C**：Comfort；
- **EP**：Ego Progress；
- **FPS**：推理速度。

PDMS 越高越好，FPS 越高越好。NC、DAC、TTC 等指标用于衡量安全性和合理性。

### 9.2 nuScenes 指标理解

nuScenes planning 常用：

- **L2 error**：预测轨迹与专家轨迹的距离误差；
- **Collision rate**：规划轨迹发生碰撞的比例；
- 通常按 1s、2s、3s 或平均值报告。

open-loop L2 并不等价于真实驾驶能力，但能衡量 imitation learning 的轨迹拟合质量。

---

## 10. 主要结果

### 10.1 NAVSIM navtest

论文在 NAVSIM navtest 上给出的关键对比如下：

| 方法 | Anchor 数 | PDMS | FPS |
|---|---:|---:|---:|
| Transfuser | 0 | 84.0 | 60 |
| DRAMA | 0 | 85.5 | 未报告 |
| Hydra-MDP-V8192-W-EP | 8192 | 86.5 | 未报告 |
| DiffusionDrive | 20 | 88.1 | 45 |

解读：

- DiffusionDrive 相比 Transfuser 提升 **4.1 PDMS**；
- 相比使用 8192 vocabulary 的 Hydra-MDP 仍更高；
- 只使用 20 个 anchor，说明 anchor 在这里不是暴力枚举；
- FPS 45，说明 truncated diffusion 没有牺牲实时性。

### 10.2 roadmap 消融

论文给出了从 Transfuser 到 DiffusionDrive 的分步演进：

| 版本 | 规划模块 | 去噪步数 | PDMS | 多样性 | FPS |
|---|---|---:|---:|---:|---:|
| Transfuser | MLP | 1 | 84.0 | 0% | 60 |
| TransfuserDP | vanilla diffusion / UNet | 20 | 84.6 | 11% | 7 |
| TransfuserTD | truncated diffusion | 2 | 85.7 | 70% | 27 |
| DiffusionDrive | diffusion decoder | 2 | 88.1 | 74% | 45 |

这个表非常关键，因为它把贡献拆开了：

1. **vanilla diffusion 只带来很小 PDMS 提升，但速度大幅下降。**
2. **truncated diffusion 同时提升速度和多样性。**
3. **diffusion decoder 进一步显著提升规划质量，并把 FPS 拉到更实用的水平。**

这说明论文不是“用 diffusion 所以有效”，而是“把 diffusion 截断并与场景交互 decoder 结合才有效”。

### 10.3 nuScenes open-loop

论文在 nuScenes 上与多种方法比较，关键结果如下：

| 方法 | Avg L2 | Avg Collision | FPS |
|---|---:|---:|---:|
| VAD | 0.72 | 0.22 | 4.5 |
| SparseDrive | 0.61 | 0.08 | 9.0 |
| DiffusionDrive | 0.57 | 0.08 | 8.2 |

解读：

- DiffusionDrive 的平均 L2 低于 SparseDrive；
- collision 与 SparseDrive 持平；
- FPS 略低于 SparseDrive，但仍在相近量级；
- 说明方法不仅适用于 NAVSIM，也能迁移到 nuScenes 框架。

---

## 11. 消融实验解读

### 11.1 去噪步数

论文结论显示，DiffusionDrive 对 denoising steps 不敏感：

- 1 step 已经接近最优；
- 2 steps 是主设置；
- 更多 steps 收益很小。

这和图像 diffusion 很不同。原因在于 trajectory 是低维结构化数据，而且初始化来自 anchor prior，不需要长链生成复杂语义。

工程含义：

- 实时系统中优先使用 1-2 steps；
- 如果 latency 预算非常紧，可以尝试 1 step；
- 如果追求极致 open-loop 精度，可以测试 3-4 steps，但收益可能不成比例。

### 11.2 候选数量 \(N_{infer}\)

论文报告 \(N_{infer}\) 从 10 增加到 20 有明显收益，从 20 到 40 后收益趋于饱和。

这说明：

- 少量候选不足以覆盖场景多模态；
- 约 20 个候选已经能覆盖多数常见驾驶模式；
- 继续增加候选会增加算力，但边际收益下降。

工程含义：

- 20 是较合理的默认值；
- 对高算力离线评估可用 40；
- 对车端实时部署可测试 10-20 的 latency/quality tradeoff。

### 11.3 decoder 组件

论文的 decoder 消融表明：

- spatial cross-attention 是最关键组件；
- agent/map query cross-attention 能进一步增强；
- cascade refinement 有助于逐步提升质量；
- 只改 diffusion 初始化而不设计强 decoder，效果不够充分。

工程含义：

DiffusionDrive 的效果依赖“轨迹与场景空间特征的细粒度交互”。如果把它移植到其他栈中，不能只替换 planning head，还需要保证 trajectory candidates 能访问到足够高质量的空间和实例特征。

### 11.4 anchor prior 的重要性

补充材料中比较了基于当前状态外推的 prior 与 anchored Gaussian prior。论文结论是 K-Means anchor prior 明显更好。

原因：

- 当前速度/方向外推只能表达短期惯性；
- K-Means anchor 来自真实驾驶轨迹分布，覆盖更丰富驾驶意图；
- 路口转向、减速停车、换道等行为不能只靠当前状态线性外推。

---

## 12. 为什么 DiffusionDrive 能缓解 mode collapse

mode collapse 的根源在于多个采样最终被同一个条件分布峰值吸引。

DiffusionDrive 从不同 anchor 附近开始，天然把候选轨迹分散到不同模式区域：

```text
anchor 1: lane keeping
anchor 2: slight left
anchor 3: slight right
anchor 4: left turn
anchor 5: right turn
...
```

训练时又只把最近 GT 的 anchor 标为 positive，避免所有 anchor 都被同一条 GT 强行拉拢。因此，不同 anchor 可以保留模式差异。

从概率角度看，vanilla diffusion 近似从一个无结构先验：

$$
p(\tau_T)=\mathcal{N}(0,I)
$$

开始；DiffusionDrive 则使用混合先验：

$$
p(\tau_i)=\sum_{k=1}^{K}\pi_k \mathcal{N}(\sqrt{\bar{\alpha}_i}a_k, (1-\bar{\alpha}_i)I)
$$

这个混合分布的每个分量对应一类驾驶模式。由于各分量中心不同，采样初始状态已经分离，后续 denoising 更容易维持多样性。

---

## 13. 工程实现视角

### 13.1 数据准备

落地 DiffusionDrive 需要先准备轨迹 anchor：

1. 收集训练集 ego future trajectories；
2. 统一坐标系，例如 ego-centric frame；
3. 对每条轨迹做固定长度采样；
4. 使用 K-Means 聚类；
5. 保存 \(K\) 条 cluster centers 作为 anchors。

需要注意：

- anchor 必须与训练/推理轨迹坐标定义一致；
- horizon、采样频率、归一化尺度必须一致；
- 如果训练数据分布改变，anchor 也应重新聚类；
- 不同城市、道路规则、车辆动力学可能需要不同 anchor set。

### 13.2 模型接入方式

如果现有系统已有 BEV encoder 或 occupancy/prediction encoder，可以把 DiffusionDrive 视为 planning decoder 替换：

```text
原始：
BEV feature -> MLP / transformer planning head -> one trajectory

替换：
BEV feature + object/map queries
       -> diffusion decoder
       -> multi-modal trajectory candidates + scores
```

关键不是完全复刻论文 backbone，而是满足 decoder 所需的信息：

- dense spatial feature；
- dynamic agent feature；
- map / lane feature；
- ego state / route command；
- diffusion timestep embedding；
- noisy trajectory coordinates。

### 13.3 训练细节建议

实际训练时应重点检查：

- positive anchor assignment 是否合理；
- candidate score 是否校准；
- noisy trajectory 是否与 anchor 尺度匹配；
- regression loss 是否只作用于 positive；
- negative candidates 是否被错误拉向 GT；
- score head 是否出现全部低分或全部高分；
- anchor coverage 是否覆盖训练集主要 maneuver。

### 13.4 推理调参旋钮

主要可调参数包括：

| 参数 | 影响 | 建议 |
|---|---|---|
| anchor 数 \(K\) | 多样性、算力 | 默认 20，按 latency 调整 |
| denoising steps | 精度、延迟 | 默认 2，低延迟可试 1 |
| noise level | 候选扰动范围 | 过小会缺少探索，过大会偏离先验 |
| score threshold | 输出稳定性 | 可结合安全模块过滤 |
| top-k 输出 | 下游选择空间 | 闭环系统建议保留多候选 |

### 13.5 与安全规划模块结合

论文主要输出 neural planner 的轨迹和 confidence。工程落地时，不建议直接把 top-1 作为唯一控制输入，而应结合：

- collision checking；
- drivable area checking；
- comfort / jerk 约束；
- traffic rule checking；
- dynamic feasibility；
- fallback planner；
- emergency braking policy。

DiffusionDrive 最适合作为高质量 candidate generator，再由安全层做约束过滤或重排序。

---

## 14. 与其他代表方法的关系

### 14.1 与 Transfuser

Transfuser 是强基线，融合相机和 LiDAR 特征后回归规划轨迹。DiffusionDrive 在 NAVSIM 实验中以 Transfuser 为感知基础之一，替换或增强其 planning head。

差异：

- Transfuser 偏单轨迹回归；
- DiffusionDrive 生成多条候选轨迹；
- DiffusionDrive 利用 anchor prior 和 denoising refinement。

### 14.2 与 VAD / SparseDrive

VAD 和 SparseDrive 属于更完整的端到端视觉自动驾驶框架，强调 object/map query 和 vectorized scene representation。

DiffusionDrive 在 nuScenes 上基于 SparseDrive 进行扩展，说明它可以作为一种 planning decoder 插入 query-based E2E driving framework。

差异：

- SparseDrive 更偏稀疏场景表示和规划；
- DiffusionDrive 强调生成式多模态规划；
- DiffusionDrive 在 SparseDrive 基础上降低 L2，collision 持平。

### 14.3 与 Hydra-MDP / VADv2 类 vocabulary 方法

Hydra-MDP 使用大规模 motion vocabulary，能覆盖多模态轨迹，但候选数量很大。

DiffusionDrive 的优势在于：

- anchor 数量少；
- 不需要大规模离散 action space；
- 不依赖额外 rule-based post-processing；
- 输出仍可连续修正。

### 14.4 与 Diffusion Policy

Diffusion Policy 在机器人控制中效果很好，但自动驾驶规划更强调实时性和场景结构约束。

DiffusionDrive 的改造点：

- 从 anchored Gaussian 而非 pure Gaussian 开始；
- 截断扩散链；
- 使用场景交互 decoder；
- 用 score head 选择候选。

---

## 15. 论文亮点

### 15.1 问题定义准确

论文没有泛泛宣传 diffusion，而是明确指出普通 diffusion 在自动驾驶中的两个问题：慢和 mode collapse。

### 15.2 方法简单但针对性强

truncated diffusion 的直觉非常直接：轨迹有强先验，因此不要从纯噪声开始。这是一个符合驾驶任务结构的改造。

### 15.3 实验链条完整

论文不仅报告最终 SOTA，还给出 roadmap：

```text
MLP planning
-> vanilla diffusion
-> truncated diffusion
-> diffusion decoder
```

这个链条能证明各模块的独立贡献。

### 15.4 工程可用性较强

2-step denoising、20 anchors、45 FPS 说明方法有实际部署潜力。相比大 vocabulary 和 vanilla diffusion，它的算力需求更合理。

---

## 16. 局限性与风险

### 16.1 仍然主要是 imitation learning

DiffusionDrive 学习人类驾驶轨迹分布，但不提供形式化安全保证。它能生成看起来合理的轨迹，不等价于在所有交互场景中安全。

### 16.2 anchor 仍然依赖数据分布

虽然 anchor 不是最终答案，但 anchor set 仍决定了初始模式覆盖。极端 OOD 场景下，如果 anchor 没有覆盖某类行为，模型可能仍然难以生成足够好的轨迹。

### 16.3 benchmark 与真实闭环驾驶有差距

NAVSIM 和 nuScenes 都是重要 benchmark，但不能完全代表真实道路闭环表现。尤其是：

- open-loop L2 不等价于驾驶质量；
- non-reactive simulation 不包含完整交互反馈；
- 真实系统还需要控制、预测、安全冗余和规则约束。

### 16.4 score calibration 很关键

模型输出多条候选轨迹后，需要依赖 confidence 选择最终轨迹。如果 score head 校准不好，可能出现：

- 安全轨迹被低估；
- 激进轨迹被高估；
- top-1 不稳定；
- 多候选质量好但排序错误。

这在工程部署中需要单独评估。

### 16.5 与预测模块的关系仍可深化

论文 decoder 会与 agent query 交互，但没有把强交互博弈式规划作为核心。真实驾驶中，ego 轨迹会影响其他 agent 反应。这类 interactive planning 仍是后续方向。

---

## 17. 对后续研究的启发

### 17.1 更好的 prior

K-Means anchor 简单有效，但还可以探索：

- route-conditioned anchors；
- map-conditioned anchors；
- speed-conditioned anchors；
- learned latent anchors；
- scenario-specific anchor retrieval；
- hierarchical maneuver prior。

### 17.2 与规则和代价函数结合

DiffusionDrive 可以作为 candidate generator，下游使用规则或优化器重排序：

```text
DiffusionDrive top-k
    -> collision cost
    -> traffic rule cost
    -> comfort cost
    -> progress cost
    -> final selection
```

这可能比只依赖 neural score 更稳健。

### 17.3 闭环训练

当前 imitation learning 仍有 covariate shift 问题。后续可以尝试：

- DAgger；
- reinforcement learning fine-tuning；
- differentiable simulation；
- closed-loop metric optimization；
- human preference / safety preference tuning。

### 17.4 联合预测与规划

多模态 ego planning 与多 agent prediction 天然相关。未来可考虑：

- ego candidate 与 agent future joint diffusion；
- interaction-aware denoising；
- conditional response prediction；
- game-theoretic scoring。

---

## 18. 复现和落地检查清单

### 18.1 复现前应确认

- 数据集版本是否与论文一致；
- trajectory horizon 和 waypoint 频率是否一致；
- anchor 聚类输入是否使用相同坐标系；
- backbone 是否使用相同 pretrained 权重；
- evaluation server / metric 版本是否一致；
- FPS 测试硬件是否可比。

### 18.2 训练时应监控

- positive anchor 分布是否均衡；
- 每个 anchor 被选为 positive 的频率；
- top-1 score accuracy；
- top-k oracle trajectory error；
- candidate diversity；
- collision / drivable area violation；
- denoising step 中轨迹是否逐渐变合理。

### 18.3 部署时应监控

- top-1 与 top-k 差异；
- high confidence failure cases；
- OOD 场景中的 anchor coverage；
- low-speed / stop-and-go 场景；
- intersection turning cases；
- lane change and merge cases；
- emergency braking fallback 触发率。

---

## 19. 关键 takeaways

1. **自动驾驶轨迹生成不需要完整 diffusion 链**  
   轨迹低维且有强驾驶先验，从 anchor 附近开始比从纯噪声开始更合理。

2. **少量 anchor 可以作为生成先验，而不是离散动作库**  
   这是 DiffusionDrive 区别于 vocabulary 方法的核心。

3. **truncated diffusion 同时解决速度和多样性问题**  
   2-step denoising 让 diffusion policy 接近实时，多 anchor 初始化缓解 mode collapse。

4. **场景交互 decoder 是性能提升的另一半**  
   没有 spatial cross-attention、agent/map interaction 和 cascade refinement，仅靠 truncated noise schedule 不足以达到最终效果。

5. **工程上应把它看作 candidate generator**  
   最稳妥的落地方式是输出 top-k 轨迹，再结合安全检查、规则约束和 fallback planner。

---

## 20. 参考资料

1. Liao et al., **DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving**, CVPR 2025.  
   <https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html>

2. Official CVPR 2025 PDF.  
   <https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.pdf>

3. arXiv version.  
   <https://arxiv.org/abs/2411.15139>

4. Official supplemental material.  
   <https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liao_DiffusionDrive_Truncated_Diffusion_CVPR_2025_supplemental.pdf>

5. Official GitHub repository.  
   <https://github.com/hustvl/DiffusionDrive>
