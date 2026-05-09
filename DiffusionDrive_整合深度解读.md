# DiffusionDrive 整合深度解读

> 整合来源：
>
> - `DiffusionDrive_deep_report.md`
> - `DiffusionDrive Reading Note.md`
>
> 目标：合并两篇笔记的主要内容，保留方法、公式、实验、工程启示和批判性分析，删去重复的背景解释、重复实验表述和多次出现的 anchor/vanilla diffusion 定义。
>
> 论文：**DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving**  
> 会议：CVPR 2025 Highlight  
> 代码：<https://github.com/hustvl/DiffusionDrive>  
> arXiv：<https://arxiv.org/abs/2411.15139>  
> CVPR 页面：<https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html>

---

## 0. 核心结论

DiffusionDrive 的价值不在于简单地把 diffusion policy 搬到自动驾驶规划，而在于它把扩散模型改造成一个 **实时、可多模态生成、能接入现有端到端驾驶框架的 planning head**。

核心设计可以压缩为一句话：

> 用少量驾驶轨迹 anchor 构造 anchored Gaussian，把扩散过程截断到靠近真实轨迹的后半段，再用场景交互式 diffusion decoder 进行少步去噪，最终生成多条连续候选轨迹并用 confidence 选择 top-1。

它同时解决了 vanilla diffusion policy 在自动驾驶规划里的两个主要问题：

1. **推理太慢**  
   普通 diffusion policy 常需要约 20 步 DDIM 去噪，论文 roadmap 中约 7 FPS；DiffusionDrive 用 2 步去噪，在 NAVSIM navtest 上达到 **45 FPS**。

2. **候选轨迹多样性不足**  
   从纯高斯噪声开始的多次采样容易被场景条件拉到同一行为模式，出现 mode collapse；DiffusionDrive 从不同 anchor 附近采样，使候选天然分布在不同驾驶模式周围，多样性从 vanilla diffusion 的 **11%** 提升到约 **74%**。

最终结果上，论文报告 DiffusionDrive 在 NAVSIM navtest 上达到 **88.1 PDMS**，使用约 **20 条 anchor**，相比 8192-anchor vocabulary 方法仍有优势，同时保持实时性。

---

## 1. 论文定位

### 1.1 任务定位

DiffusionDrive 面向端到端自动驾驶中的 planning/control。输入是多相机、LiDAR、地图或中间 BEV/query 表示，输出未来几秒 ego 车辆轨迹。

它不是一个完整自动驾驶系统，而是一个 **规划头替换方案**：

```text
已有感知/场景编码模块
    -> BEV/PV feature + agent query + map query
    -> DiffusionDrive planning head
    -> 多模态候选轨迹 + confidence score
```

论文在两个体系中验证：

- **NAVSIM**：基于 Transfuser 风格的多相机 + LiDAR 感知框架。
- **nuScenes**：基于 SparseDrive，把原规划模块替换成 diffusion decoder。

这使实验结论更聚焦：性能提升主要来自 planning head 的范式变化，而不是完全不同的 backbone。

### 1.2 它解决的核心矛盾

自动驾驶规划天然是 one-to-many 问题。同一个场景里，合理未来可能包括保持车道、轻微绕行、变道、减速等待、加速通过等。如果用单条轨迹回归，模型容易学到“平均行为”；而平均轨迹在多模态场景里往往不可执行。

已有路线各有问题：

| 路线 | 代表方法 | 优点 | 主要问题 |
|---|---|---|---|
| 单轨迹回归 | Transfuser, UniAD, ST-P3, TCP | 快、简单、易训练 | 难表达多种可行驾驶意图 |
| 大规模 trajectory vocabulary | Hydra-MDP, VADv2, GUMP, GameFormer | 有多模态候选和驾驶先验 | action space 离散化，候选库巨大，依赖覆盖度 |
| vanilla diffusion policy | Diffuser (Janner et al.), Diffusion Policy (Chi et al.), CTG++ | 连续多模态生成能力强 | 去噪步数多，实时性差，容易 mode collapse |

DiffusionDrive 的折中是：

```text
少量 anchor 提供驾驶先验
    +
truncated diffusion 保留连续生成能力
    +
强场景交互 decoder 保证轨迹可行性
```

---

## 2. Vanilla Diffusion 为什么不适合直接做驾驶规划

### 2.1 标准扩散过程

设未来轨迹为：

$$
\tau = \{(x_t, y_t)\}_{t=1}^{T_f}
$$

标准 diffusion 的前向加噪过程可以写为：

$$
q(\tau_i|\tau_0)=\mathcal{N}(\sqrt{\bar{\alpha}_i}\tau_0, (1-\bar{\alpha}_i)I)
$$

当加噪步数足够大时，样本趋近标准高斯：

$$
\tau_T \sim \mathcal{N}(0,I)
$$

反向过程学习从噪声还原轨迹：

$$
p_\theta(\tau_{i-1}|\tau_i, z)
$$

其中 \(z\) 是场景条件特征。

### 2.2 自动驾驶里的三个不匹配

第一，驾驶轨迹是低维、强约束、强先验的数据，不像图像那样需要从纯随机噪声中生成复杂纹理和语义。常见驾驶模式数量有限，完全从 \(\mathcal{N}(0,I)\) 起步并不高效。

第二，驾驶规划有严格延迟预算。多相机感知 backbone 本身已经重，再叠加 20 步以上 denoising，会让规划频率难以满足实时闭环。

第三，场景条件过强时，不同初始噪声的差异可能被压制，最终多条轨迹收敛到相似模式。这就是论文强调的 mode collapse：看似采样多次，实际没有覆盖不同意图。

---

## 3. Truncated Diffusion：从纯噪声生成改为从驾驶先验修正

### 3.1 直觉

DiffusionDrive 的核心直觉是：驾驶轨迹不需要从零创造，应该从“像驾驶”的模式附近开始。

普通 diffusion：

```text
pure Gaussian noise -> ... -> trajectory
```

DiffusionDrive：

```text
noisy anchor trajectory -> refined trajectory
```

也就是跳过从纯噪声到粗轨迹结构的大段生成过程，只学习靠近真实数据的后段 refinement，因此叫 **truncated diffusion**。

### 3.2 Anchor 的定义

Anchor 是训练集中 ego future trajectories 的 K-Means 聚类中心。每个 anchor 是一条完整未来轨迹，而不是单个 waypoint：

$$
a_k = \{(x_t, y_t)\}_{t=1}^{T_f},\quad k=1,\dots,K
$$

它可以理解为典型驾驶模式的压缩表示，例如：

- 保持车道；
- 轻微左偏或右偏；
- 左转、右转；
- 变道；
- 减速停车；
- 绕障或通过。

NAVSIM 主设置中约使用 **20 条 anchor**。这与 4096/8192 级别的 trajectory vocabulary 有本质区别：anchor 不是最终候选答案，而是生成分布的中心。

### 3.3 Anchored Gaussian

训练或推理时，不从标准高斯采样初始轨迹，而从每个 anchor 附近采样 noisy trajectory：

$$
\tau_k^i = \sqrt{\bar{\alpha}_i}a_k + \sqrt{1-\bar{\alpha}_i}\epsilon,\quad
\epsilon\sim\mathcal{N}(0,I)
$$

这里 \(i\) 是截断后的噪声等级，远小于完整扩散链末端。样本仍保留 anchor 的大体形状，只带有适度扰动。

从概率角度看，DiffusionDrive 的初始分布近似为混合先验：

$$
p(\tau_i)=\sum_{k=1}^{K}\pi_k
\mathcal{N}(\sqrt{\bar{\alpha}_i}a_k,(1-\bar{\alpha}_i)I)
$$

每个 Gaussian component 对应一类驾驶模式。由于初始中心不同，候选轨迹在 denoising 前就已分散，后续更容易保持多样性。

### 3.4 Anchor 不是 Vocabulary

这点是理解论文的关键。

| 维度 | 大 vocabulary / anchor 方法 | DiffusionDrive |
|---|---|---|
| anchor 数量 | 通常 4096 / 8192 | 主设置约 20 |
| anchor 角色 | 候选答案或离散动作 | 生成分布中心 |
| 输出空间 | 偏离散，依赖候选库覆盖 | 连续轨迹，可大幅修正 anchor |
| 多模态来源 | 候选库枚举 | anchor prior + noisy sampling |
| 推理负担 | 大量候选打分/筛选 | 少量候选 + 少步去噪 |

可以概括为：

```text
vocabulary 方法关心“选哪条”
DiffusionDrive 关心“从哪类模式附近生成”
```

---

## 4. 训练机制

### 4.1 输入与输出

每个训练样本输入：

- 场景特征 \(z\)：来自 BEV/PV feature、agent query、map query 等；
- 一组 noisy anchor trajectories；
- diffusion timestep embedding。

模型输出：

- 每个候选的 denoised trajectory \(\hat{\tau}_k\)；
- 每个候选的 confidence score \(\hat{s}_k\)。

形式上：

$$
\{\hat{\tau}_k,\hat{s}_k\}_{k=1}^{K}
= f_\theta(\{\tau_k^i\}_{k=1}^{K}, z, i)
$$

### 4.2 Positive / Negative Anchor Assignment

训练时需要确定哪条 anchor 对应当前 ground-truth 轨迹。论文采用最近距离匹配：

$$
k^*=\arg\min_k d(a_k,\tau_{gt})
$$

其中 \(d(\cdot)\) 可理解为轨迹级距离，例如 waypoint 平均 L2。

匹配到的 anchor 为 positive，其余为 negative：

$$
y_k =
\begin{cases}
1, & k=k^* \\
0, & k\ne k^*
\end{cases}
$$

这个设计避免所有候选轨迹都被拉向同一条 GT，从训练目标上保护多模态分化。

### 4.3 损失函数

训练目标由重构和分类两部分组成。为了避免求和形式在部分渲染器中显示不清，这里采用分步展开写法。

总损失为：

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda \cdot \mathcal{L}_{\text{cls}}
$$

重构损失只作用于 positive anchor：

$$
\mathcal{L}_{\text{rec}} = L_{\text{reg}}(\hat{\tau}_{k^*}, \;\tau_{\text{gt}})
$$

其中：

$$
k^* = \arg\min_k d(a_k, \tau_{\text{gt}})
$$

$k^*$ 是与 GT 最近的 anchor 索引，只有这一条候选被要求回归到 GT 轨迹。$L_{\text{reg}}$ 通常是 smooth-L1 或 L2 loss，逐 waypoint 计算后取平均。

分类损失作用于所有 anchor：

$$
\mathcal{L}_{\text{cls}} = \sum_{k=1}^{K} \text{BCE}(\hat{s}_k, \;y_k)
$$

标签定义为：

$$
y_k =
\begin{cases}
1, & k = k^* \quad\text{(positive)} \\
0, & k \ne k^* \quad\text{(negative)}
\end{cases}
$$

合在一起即：

$$
\mathcal{L}
= \underbrace{L_{\text{reg}}(\hat{\tau}_{k^*}, \tau_{\text{gt}})}_{\text{positive anchor 回归}}
+ \lambda
\underbrace{\sum_{k=1}^{K} \text{BCE}(\hat{s}_k, y_k)}_{\text{所有 anchor 分类}}
$$

也可以写成等价的求和形式：

$$
\mathcal{L}
= \sum_{k=1}^{K} y_k \cdot L_{\text{reg}}(\hat{\tau}_k,\tau_{\text{gt}})
+ \lambda \sum_{k=1}^{K} \text{BCE}(\hat{s}_k,y_k)
$$

因为 $y_k$ 只在 $k=k^*$ 时为 1，其余为 0，所以重构项实际只保留单条 positive anchor。

核心含义：

- positive anchor 负责回归当前 GT 轨迹；
- score head 学习当前场景下哪个模式最合理；
- negative anchors 主要参与分类，不被强行回归到 GT。

推理时，模型可以输出 top-1，也可以保留 top-k 给后续安全过滤或规则重排序。

---

## 5. 推理流程

DiffusionDrive 推理可以概括为：

1. 加载 K-Means 得到的固定 anchor set；
2. 对每个 anchor 采样噪声，得到 noisy anchor trajectories；
3. 输入 noisy trajectories、场景特征和 timestep embedding；
4. 用 diffusion decoder 做少量 DDIM denoising，主设置为 2 步；
5. 输出多条候选轨迹和 confidence；
6. 选择最高分轨迹作为最终规划输出。

伪代码：

```python
anchors = load_kmeans_trajectory_anchors()
features = scene_encoder(sensor_inputs)

samples = []
for anchor in anchors:
    noise = randn_like(anchor)
    noisy_traj = sqrt(alpha_bar[t]) * anchor + sqrt(1 - alpha_bar[t]) * noise
    samples.append(noisy_traj)

traj_candidates = stack(samples)

for t in ddim_steps:  # usually 2 steps
    traj_candidates, scores = diffusion_decoder(
        traj_candidates,
        features,
        timestep=t,
    )

final_traj = traj_candidates[argmax(scores)]
```

工程上，anchors 是批量化输入，decoder 一次处理多条候选，不是为每条轨迹单独跑完整模型。

---

## 6. Diffusion Decoder 架构

DiffusionDrive 的性能不只来自 truncated schedule。真正让轨迹变得可行的是 **场景交互式 diffusion decoder**。

### 6.1 为什么不能只用 MLP 或普通 UNet

轨迹规划必须与场景细节交互：

- 避开车辆、行人、骑行者；
- 遵守道路拓扑和可行驶区域；
- 跟随 lane centerline；
- 考虑停止线、交通灯、前车速度；
- 满足 comfort 和动态可行性。

如果只用 MLP，从全局场景 embedding 回归轨迹，空间信息过度压缩。若只用普通 UNet 处理低维轨迹，又缺少和道路/目标的细粒度对齐。

### 6.2 Decoder 输入

核心输入包括：

- noisy trajectory samples；
- diffusion timestep embedding；
- dense BEV/PV spatial feature；
- agent/object queries；
- map queries；
- trajectory query embeddings。

NAVSIM 实现中主要基于 Transfuser，使用 BEV feature 和 object queries。nuScenes 实现中基于 SparseDrive，使用 object queries、map queries 和 perspective-view image features。

### 6.3 Spatial Cross-Attention

Spatial cross-attention 让每条候选轨迹根据自身 waypoint 坐标去读取空间特征。

直观理解：

```text
候选轨迹向左变道
    -> 读取左侧车道、左侧车辆、边界、可行驶区域

候选轨迹直行
    -> 读取前方车道、前车、停止线、路口结构
```

这部分是 decoder 的关键组件。消融显示，缺少有效的 spatial cross-attention 时，planning quality 明显下降。

### 6.4 Agent / Map Cross-Attention

Agent query 提供实例级动态信息，例如其他车辆的位置、速度、类别和运动状态。Map query 提供道路拓扑和矢量化结构。它们与 dense spatial feature 互补：

- spatial feature 适合局部环境证据；
- agent query 适合动态对象关系；
- map query 适合车道、边界、拓扑。

### 6.5 Cascade Refinement

Decoder 采用级联 refinement：上一层输出 refined trajectory，下一层继续与场景交互并修正。

```text
noisy anchor
  -> coarse scene-aware trajectory
  -> refined trajectory
  -> score
```

这适合规划任务，因为轨迹从 anchor 出发，本身已经合理，只需要逐步根据场景修正细节。

---

## 7. 实验结果整合

### 7.1 NAVSIM navtest

核心结果：

| 方法 | Anchor 数 | PDMS | FPS |
|---|---:|---:|---:|
| Transfuser | 0 | 84.0 | 60 |
| DRAMA | 0 | 85.5 | 未报告 |
| Hydra-MDP-V8192-W-EP | 8192 | 86.5 | 未报告 |
| DiffusionDrive | 20 | 88.1 | 45 |

解读：

- DiffusionDrive 比 Transfuser 高 **4.1 PDMS**；
- 比使用 8192 vocabulary 的 Hydra-MDP 仍高；
- 只用 20 anchors，说明它不是靠暴力候选枚举；
- 45 FPS 说明 2-step truncated diffusion 有实际实时潜力。

### 7.2 Roadmap 消融

论文最有价值的表之一是从 Transfuser 到 DiffusionDrive 的逐步演进：

| 版本 | 规划模块 | 去噪步数 | PDMS | 多样性 | FPS |
|---|---|---:|---:|---:|---:|
| Transfuser | MLP | 1 | 84.0 | 0% | 60 |
| TransfuserDP | vanilla diffusion / UNet | 20 | 84.6 | 11% | 7 |
| TransfuserTD | truncated diffusion | 2 | 85.7 | 70% | 27 |
| DiffusionDrive | diffusion decoder | 2 | 88.1 | 74% | 45 |

这张表说明：

1. 直接换成 vanilla diffusion 收益很小，但速度急剧下降。
2. Truncated diffusion 是速度和多样性改善的关键。
3. Diffusion decoder 是最终 PDMS 提升的关键。

因此，论文结论不是“diffusion 一定更好”，而是：

> 只有把 diffusion 改成 anchored + truncated + scene-interactive decoder，才适合自动驾驶规划。

### 7.3 nuScenes Open-Loop

在 nuScenes 上，DiffusionDrive 替换 SparseDrive planning head：

| 方法 | Avg L2 | Avg Collision | FPS |
|---|---:|---:|---:|
| VAD | 0.72 | 0.22 | 4.5 |
| SparseDrive | 0.61 | 0.08 | 9.0 |
| DiffusionDrive | 0.57 | 0.08 | 8.2 |

解读：

- 相比 SparseDrive，DiffusionDrive L2 更低，collision 持平；
- 相比 VAD，L2 降低约 20.8%，collision 降低约 63.6%；
- FPS 略低于 SparseDrive，但仍在相近实时量级。

### 7.4 Decoder 结构消融

Decoder 消融可以总结为：

- Transfuser_TD 使用 UNet + ego query、无完整 cascade 时，PDMS 约 85.7；
- 最终 DiffusionDrive 使用 deformable spatial cross-attention + cascade decoder，PDMS 达 88.1；
- 相比中间版本，完整 decoder 不只是涨分，还减少了部分参数；
- 没有 cross-attention、只靠简单 MLP/FFN 时，性能明显不足。

这说明 decoder 是方法的另一半，而不是可有可无的实现细节。

### 7.5 去噪步数、候选数量、Anchor Prior

关键消融结论：

- **Denoising steps**：1 step 已较强，2 steps 为主配置，更多 step 收益有限。
- **\(N_{infer}\)**：从 10 到 20 有明显提升，从 20 到 40 边际收益下降。
- **Cascade depth**：层数增加有收益但递减，默认选择 2 层兼顾效果与计算。
- **Anchor prior**：K-Means anchor 明显优于当前速度/航向线性外推 prior。
- **跨场景泛化**：将 NAVSIM 聚类 anchors 迁移到 CARLA Longest6 仍有可用效果，说明 anchor 不是完全过拟合单一 benchmark。

---

## 8. 为什么 DiffusionDrive 能缓解 Mode Collapse

Mode collapse 的根源是：多次采样虽然从不同噪声开始，但在强场景条件和单一 GT 监督下，去噪网络容易把所有样本拉向同一个高概率模式。

DiffusionDrive 从两个层面缓解：

1. **初始化层面**  
   不同 anchor 的中心不同，初始 noisy trajectory 已分布在不同驾驶模式附近。

2. **训练目标层面**  
   只有最近 GT 的 anchor 是 positive，其余 anchor 不被强制回归 GT，避免所有候选被同一标签吸到一起。

因此，候选轨迹既有全局模式差异，也有 anchor 内部的局部扰动。

---

## 9. 工程实现视角

### 9.1 数据准备

落地 DiffusionDrive 需要先生成 anchors：

1. 收集训练集 ego future trajectories；
2. 统一到 ego-centric 坐标系；
3. 固定 horizon 和 waypoint 频率；
4. 用 K-Means 聚类；
5. 保存 cluster centers 作为 anchor set。

注意事项：

- 坐标系、尺度、采样频率必须与训练/推理一致；
- 数据分布改变时，anchor 应重新聚类或做条件化；
- 城市、道路规则、车辆动力学差异可能影响 anchor 覆盖。

### 9.2 接入现有系统

若已有 BEV encoder、prediction encoder 或 query-based E2E driving framework，可以把 DiffusionDrive 看作 planning decoder 替换：

```text
原始：
BEV feature -> MLP / transformer planning head -> one trajectory

替换：
BEV feature + object/map queries + anchors
    -> diffusion decoder
    -> multi-modal trajectory candidates + scores
```

关键要求：

- 有足够高质量的 dense spatial feature；
- 有动态 agent 或 object-level 表征；
- 有地图/道路结构表征；
- noisy trajectory 坐标能与 feature 空间对齐；
- score head 需要单独评估校准质量。

### 9.3 训练时应监控

- positive anchor 分布是否严重不均衡；
- 每个 anchor 被选为 positive 的频率；
- top-1 score accuracy；
- top-k oracle error；
- candidate diversity；
- collision / drivable area violation；
- denoising 过程中轨迹是否逐步变合理；
- negative candidates 是否被错误回归到 GT。

### 9.4 推理调参

| 参数 | 影响 | 建议 |
|---|---|---|
| anchor 数 \(K\) | 多样性、算力 | 默认约 20，按场景和 latency 调整 |
| denoising steps | 精度、延迟 | 默认 2，极低延迟可尝试 1 |
| noise level | 探索范围 | 太小缺少多样性，太大偏离先验 |
| \(N_{infer}\) | 候选覆盖度 | 10-20 常用，离线可更大 |
| score threshold | 输出稳定性 | 应结合安全模块调校 |
| top-k 输出 | 下游选择空间 | 闭环系统建议保留 top-k |

### 9.5 与安全模块结合

工程落地不应直接把 top-1 neural trajectory 当作唯一控制决策。更稳妥的用法是：

```text
DiffusionDrive top-k candidates
    -> collision checking
    -> drivable area checking
    -> traffic rule checking
    -> comfort / jerk cost
    -> dynamic feasibility checking
    -> fallback / emergency policy
    -> final trajectory
```

DiffusionDrive 更适合做 **高质量 candidate generator**，再由规则、代价函数或安全层进行过滤和重排序。

---

## 10. 与代表方法的关系

### 10.1 Transfuser

Transfuser 是强感知-规划基线，偏单轨迹回归。DiffusionDrive 在 NAVSIM 上以 Transfuser 感知模块为基础，替换 planning head，从单轨迹变成多候选生成。

### 10.2 VAD / SparseDrive

VAD 和 SparseDrive 强调 vectorized scene representation、object/map query 和端到端规划。DiffusionDrive 可插入这类 query-based 框架，作为生成式 planning decoder。

### 10.3 Hydra-MDP / VADv2 类 Vocabulary 方法

这类方法依赖大规模 motion vocabulary。DiffusionDrive 的优势是：

- anchor 少得多；
- 输出连续，不被 vocabulary 完全离散化；
- 不需要大规模候选库；
- 推理负担较轻。

### 10.4 Diffusion Policy

机器人领域的 Diffusion Policy 通常可以接受较多 denoising steps，但自动驾驶规划要求更高实时性和更强场景几何约束。DiffusionDrive 的任务特化改造是：

- anchored Gaussian；
- truncated diffusion chain；
- spatial/agent/map cross-attention；
- candidate score head。

### 10.5 与 DiffusionDriveV2 / RL 扩展

后续 DiffusionDriveV2 在 v1 基础上引入 RL 约束。可以把它理解为对 v1 的自然延伸：

- v1 解决“扩散规划如何实时、多模态”；
- v2 进一步关注“生成的多模态轨迹如何通过 RL 保证高质量和安全偏好”。

这条路线与 RAD/RAD-2 的闭环 RL 思路相呼应：单纯 imitation learning 的生成式规划器仍缺少闭环负反馈，后续需要把 collision、progress、comfort、safety margin 等指标纳入训练或重排序。

---

## 11. 局限性与风险

### 11.1 仍然主要是 Imitation Learning

DiffusionDrive 学习的是人类轨迹分布。它能生成像专家、贴近专家的轨迹，但不提供形式化安全保证，也不直接优化闭环失败后果。

### 11.2 Anchor 依赖数据分布

Anchor 不是最终答案，但仍决定初始模式覆盖。极端 OOD 场景下，如果 anchor set 缺少对应行为模式，模型生成能力会受限。

### 11.3 Score Calibration 很关键

多候选规划最终要选 top-1。若 score head 校准不好，会出现：

- 安全轨迹被低估；
- 激进轨迹被高估；
- top-k 质量不错但 top-1 错误；
- 场景切换时输出不稳定。

因此，score 不能只看分类 loss，还应在闭环或规则检查中验证。

### 11.4 Benchmark 不等于真实闭环驾驶

NAVSIM 和 nuScenes 都不能完全代表真实交通交互：

- open-loop L2 不等价于驾驶质量；
- non-reactive simulation 不包含完整 agent response；
- 实车还需要控制、预测、安全冗余、法规和 fallback。

### 11.5 交互预测仍可深化

Decoder 与 agent query 有交互，但并未真正建模 ego 轨迹改变后其他 agent 的响应。未来更理想的方向是 ego planning 与 multi-agent prediction 联合生成。

---

## 12. 后续研究方向

### 12.1 更好的 Prior

K-Means anchor 简单有效，但还可以扩展为：

- route-conditioned anchors；
- map-conditioned anchors；
- speed-conditioned anchors；
- scenario retrieval anchors；
- learned latent anchors；
- hierarchical maneuver prior。

### 12.2 规则/代价函数重排序

DiffusionDrive 输出 top-k 后，可以用显式代价函数重排序：

```text
score = neural_confidence
      - collision_cost
      - drivable_area_cost
      - rule_violation_cost
      - comfort_cost
      + progress_reward
```

这比只依赖 neural confidence 更适合工程部署。

### 12.3 闭环训练

后续可结合：

- DAgger；
- reinforcement learning fine-tuning；
- differentiable simulation；
- 3DGS / BEV-Warp / world model；
- human preference 或 safety preference optimization。

### 12.4 联合预测与规划

更进一步可以做：

- ego candidate 与 surrounding agents future joint diffusion；
- interaction-aware denoising；
- conditional response prediction；
- game-theoretic trajectory scoring。

---

## 13. 复现与落地检查清单

### 13.1 复现前确认

- 数据集版本和 evaluation server 是否一致；
- trajectory horizon、waypoint 数量和频率是否一致；
- anchor 聚类坐标系是否正确；
- backbone 和 pretrained weights 是否一致；
- FPS 测试硬件和 batch 设置是否可比；
- 是否使用官方 checkpoint 先跑通 NAVSIM navtest。

### 13.2 训练阶段确认

- anchored Gaussian 公式和 noise schedule 是否实现正确；
- regression loss 是否只作用于 positive anchor；
- BCE 标签是否与 nearest-anchor assignment 对齐；
- decoder cross-attention 的坐标归一化是否正确；
- cascade refinement 是否共享权重或按论文配置实现；
- top-k diversity 是否随着训练保留。

### 13.3 部署阶段确认

- top-1 与 top-k 轨迹差异；
- high-confidence failure case；
- OOD 场景 anchor coverage；
- 低速 stop-and-go；
- 路口转弯；
- lane change / merge；
- emergency braking fallback 触发率。

---

## 14. 关键 Takeaways

1. **驾驶轨迹生成不需要完整 diffusion 链**  
   轨迹低维且有强先验，从 anchor 附近开始比从纯噪声开始更合理。

2. **Anchor 是生成先验，不是离散动作库**  
   这是 DiffusionDrive 与大 vocabulary 方法的根本区别。

3. **Truncated diffusion 同时改善速度和多样性**  
   2-step denoising 解决实时性，多 anchor 初始化缓解 mode collapse。

4. **Decoder 的场景交互能力决定上限**  
   没有 spatial cross-attention、agent/map interaction 和 cascade refinement，truncated schedule 本身不足以达到最终效果。

5. **工程上应把 DiffusionDrive 当成 candidate generator**  
   最稳妥的落地方式是输出 top-k，再结合安全检查、规则约束和 fallback planner。

6. **后续重点是闭环质量控制**  
   v1 解决实时多模态生成，下一步是通过 RL、偏好优化或仿真闭环，把碰撞、进度、舒适和规则纳入训练目标。

---

## 15. 补充 Q&A

### Q1：用少量驾驶轨迹 anchor 构造 anchored Gaussian，具体是怎么构造的？

构造分为**离线准备**和**在线采样**两个阶段：

**离线准备（训练前一次性完成）：**

```text
训练集 N 条 ego future trajectory
    ↓ 统一到 ego-centric 坐标系，固定 horizon（如 4s）和 waypoint 频率（如 2Hz → 8 个点）
    ↓ 每条轨迹表示为 (x₁,y₁, x₂,y₂, ..., x₈,y₈) ∈ ℝ¹⁶
    ↓ 对这 N 个 16 维向量做 K-Means 聚类（K≈20）
    ↓ 得到 K 个聚类中心 a₁, a₂, ..., a_K
```

每个 $a_k \in \mathbb{R}^{T_f \times 2}$ 就是一条"典型驾驶模式轨迹"。

**在线采样（训练和推理时每次执行）：**

对每个 anchor $a_k$，在截断时刻 $i$ 处用前向扩散公式生成 noisy sample：

$$
\tau_k^i = \underbrace{\sqrt{\bar\alpha_i} \cdot a_k}_{\text{保留 anchor 结构}} + \underbrace{\sqrt{1-\bar\alpha_i} \cdot \epsilon}_{\text{添加随机扰动}}, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这等价于从以下高斯分布采样：

$$
\tau_k^i \sim \mathcal{N}\left(\sqrt{\bar\alpha_i} \cdot a_k, \;(1-\bar\alpha_i) I\right)
$$

把 K 个 anchor 的分布合在一起，整体初始分布就是一个 **Gaussian Mixture Model（GMM）**：

$$
p(\tau_i) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}\left(\sqrt{\bar\alpha_i} \cdot a_k, \;(1-\bar\alpha_i) I\right)
$$

其中 $\pi_k$ 可以是均匀权重 $1/K$。每个 Gaussian component 的均值是 anchor 经噪声衰减后的位置，方差由截断时刻 $i$ 对应的 noise schedule 决定。

**关键参数的作用：**

| 参数 | 含义 | 典型值 |
|---|---|---|
| $K$ | anchor 数量，即 GMM 组件数 | ≈20 |
| $i$ | 截断时刻，决定噪声强度 | 远小于完整链长 $T$（如 $T$=1000 中取 $i$≈50-200） |
| $\bar\alpha_i$ | 累积信噪比，越大信号保留越多 | 由 cosine/linear schedule 决定 |

---

### Q2："强场景交互 decoder 保证轨迹可行性"体现在哪部分 feature？

Decoder 的场景交互通过**三类 feature 的 cross-attention** 实现：

```text
┌─────────────────────────────────────────────────────────┐
│  候选轨迹 query（noisy trajectory waypoints）            │
│       │                                                  │
│       ├──── Spatial Cross-Attention ────→ BEV/PV 空间特征│
│       │     （轨迹 waypoint 坐标 → 读取局部空间证据）     │
│       │     体现：可行驶区域、道路边界、障碍物位置         │
│       │                                                  │
│       ├──── Agent Cross-Attention ─────→ Object Queries  │
│       │     （与动态目标实例级交互）                       │
│       │     体现：周围车辆位置/速度/意图、行人运动         │
│       │                                                  │
│       └──── Map Cross-Attention ───────→ Map Queries     │
│             （与道路结构交互）                             │
│             体现：车道线、拓扑连接、停止线、转向约束       │
└─────────────────────────────────────────────────────────┘
```

具体对应的 feature 来源：

| Feature 类型 | 来源 | 提供的信息 | 交互方式 |
|---|---|---|---|
| Dense BEV/PV spatial feature | 感知 backbone（如 Transfuser 的 BEV encoder） | 局部空间环境：道路形状、障碍物、可行驶区域 | Deformable cross-attention，轨迹 waypoint 坐标作为 reference point |
| Agent/Object queries | 感知检测头或 query-based decoder（如 SparseDrive 的 instance queries） | 动态目标：位置、速度、类别、运动状态 | Standard cross-attention |
| Map queries | 地图编码模块（如 SparseDrive 的 map queries） | 静态拓扑：车道中心线、边界、连接关系 | Standard cross-attention |

"保证轨迹可行性"的机制是：每条候选轨迹根据**自身当前 waypoint 位置**去查询周围环境——左变道的候选读取左侧信息，直行的候选读取前方信息——然后 decoder 据此修正轨迹使其避障、贴合车道、遵守拓扑。

---

### Q3：训练时每个样本的 noisy anchor trajectories 和 diffusion timestep embedding 是什么？

**Noisy anchor trajectories 的生成过程：**

每个训练 iteration，对一个 batch 中的每个样本：

```python
# 1. anchor set 是固定的（训练前聚类好的 K 条轨迹）
anchors = fixed_anchor_set  # shape: [K, T_f, 2]

# 2. 随机采样一个截断范围内的 timestep
i = randint(1, i_max)  # i_max 是截断上界，远小于完整 T

# 3. 查表得到该 timestep 对应的 noise schedule 参数
alpha_bar_i = noise_schedule[i]  # 累积 ᾱᵢ

# 4. 采样随机噪声
epsilon = randn_like(anchors)  # shape: [K, T_f, 2]

# 5. 前向加噪，得到这个样本的 noisy anchor trajectories
noisy_anchors = sqrt(alpha_bar_i) * anchors + sqrt(1 - alpha_bar_i) * epsilon
# shape: [K, T_f, 2]，即 K 条带噪声的轨迹
```

注意：同一个 batch 内不同样本可以采样不同的 timestep $i$，这样模型学会处理不同噪声等级。

**Diffusion timestep embedding 是什么：**

Timestep embedding 的作用是告诉去噪网络"当前输入的噪声有多大"，让同一个网络能处理不同噪声等级。

具体实现通常是 **sinusoidal positional encoding**（与 Transformer 中的位置编码同源）：

$$
\text{emb}(i) = [\sin(i/10000^{0/d}), \cos(i/10000^{0/d}), \sin(i/10000^{2/d}), \cos(i/10000^{2/d}), \dots]
$$

然后通过 MLP 投影到模型隐藏维度：

```python
# 典型实现
timestep_embed = sinusoidal_encoding(i)       # [d_embed]
timestep_embed = MLP(timestep_embed)           # [d_model]
# 然后以 加法/拼接/FiLM调制 方式注入 decoder 各层
```

它的核心意义：同一个 decoder 在 $i=1$（几乎无噪声，做精细修正）和 $i=100$（噪声较大，做粗结构恢复）时应该有不同行为，timestep embedding 提供这个条件信号。

---

### Q4：推理时 "固定 anchor set" 是怎么获得的？是随机加载吗？

**不是随机加载。** Anchor set 是训练前通过离线聚类**一次性确定**的，之后在训练和推理中始终使用同一组 anchor。

**完整获取流程：**

```text
Step 1: 数据收集
    遍历训练集所有样本，提取每个样本的 ego future trajectory
    → 得到 N 条轨迹（N = 训练集样本数）

Step 2: 坐标统一
    将所有轨迹转换到 ego-centric 坐标系
    （以当前时刻 ego 位置为原点，朝向为 y 轴正方向）

Step 3: K-Means 聚类
    对 N 条轨迹（每条是 T_f×2 维向量）执行 K-Means
    → 得到 K 个聚类中心 {a₁, a₂, ..., a_K}

Step 4: 保存
    将 K 个 anchor 保存为文件（如 anchors.npy 或 anchors.pt）

Step 5: 训练和推理时加载
    模型初始化时加载这个固定文件
    推理时每一帧都使用全部 K 个 anchor 作为起点
```

**关键澄清：**

- 推理时**每一帧都使用全部 K 个 anchor**，不是随机选子集
- Anchor set 在模型生命周期内不变，除非数据分布变了需要重新聚类
- 这些 anchor 是**模型的一部分**（类似 learned codebook），只不过它们是通过 K-Means 初始化而非端到端学习得到的
- 不同数据集/场景应该分别聚类各自的 anchor set（论文验证了 NAVSIM anchor 迁移到 CARLA 仍可用，但理想情况下应针对目标域重新聚类）

---

## 16. 参考资料

1. DiffusionDrive GitHub repository: <https://github.com/hustvl/DiffusionDrive>
2. DiffusionDrive arXiv: <https://arxiv.org/abs/2411.15139>
3. CVPR 2025 paper page: <https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html>
4. CVPR 2025 PDF: <https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.pdf>
5. Supplemental material: <https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liao_DiffusionDrive_Truncated_Diffusion_CVPR_2025_supplemental.pdf>
6. NAVSIM metrics documentation: <https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md>
