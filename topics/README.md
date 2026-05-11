# 研究笔记索引

---

## 01-扩散模型

扩散模型的理论基础、训练细节、与强化学习的融合。

| 文件 | 说明 |
|------|------|
| [**扩散模型：从生成理论到决策智能.md**](01-diffusion-models(扩散模型)/notes/理论基础/扩散模型：从生成理论到决策智能.md) | 整合文档，涵盖数学原理、训练优化、RL 融合、机器人/自驾应用全景 |
| [`原始素材/扩散模型详解.md`](01-diffusion-models(扩散模型)/notes/原始素材/扩散模型详解.md) | 原始笔记：基础理论 + 算法演进 + 机器人/自驾应用 |
| [`原始素材/扩散模型图像生成：数据构建、训练、损失与优化.md`](01-diffusion-models(扩散模型)/notes/原始素材/扩散模型图像生成：数据构建、训练、损失与优化.md) | 原始笔记：DDPM 训练流程、损失函数、优化器配置 |
| [`原始素材/扩散模型与强化学习的关系.md`](01-diffusion-models(扩散模型)/notes/原始素材/扩散模型与强化学习的关系.md) | 原始笔记：RL 与扩散模型的四种融合范式 |

## 02-VLA 与基础模型

Physical Intelligence 系列论文及 VLA 模型族研究。

| 文件 | 说明 |
|------|------|
| [Physical_Intelligence_VLA_Reading_Report.md](02-robot-manipulation(机器人操控)/notes/VLA/Physical_Intelligence_VLA_Reading_Report.md) | 9 篇 VLA 论文综合阅读报告（PI 核心 4 篇 + 学术界改进 5 篇） |
| [PI_Series_Deep_Reading_Report.md](02-robot-manipulation(机器人操控)/notes/VLA/PI_Series_Deep_Reading_Report.md) | PI 技术演进线深度解析：π₀ → π₀.5 → FAST → π₀-FAST |

## 03-UMI 项目

UMI 系列 34 篇论文读书报告（按类别拆分）+ 专题笔记。

| 文件 | 说明 |
|------|------|
| [**0-UMI原论文与全景概览.md**](02-robot-manipulation(机器人操控)/notes/UMI/0-UMI原论文与全景概览.md) | UMI 1.0 原论文详解 + 34 篇论文分类索引 + 技术脉络总结 |
| [1-UMI系列核心扩展（9篇）.md](02-robot-manipulation(机器人操控)/notes/UMI/1-UMI系列核心扩展（9篇）.md) | UMI-FT、FastUMI、MV-UMI、UMI-on-Air、UMI on Legs、ActiveUMI、DexUMI、HoMMI |
| [2-硬件改进与触觉感知（6篇）.md](02-robot-manipulation(机器人操控)/notes/UMI/2-硬件改进与触觉感知（6篇）.md) | TacUMI、FARM、UMIGen、exUMI、TacThru-UMI、Hoi! |
| [3-数据策略与学习框架（6篇）.md](02-robot-manipulation(机器人操控)/notes/UMI/3-数据策略与学习框架（6篇）.md) | RDT2、RwoR、Label-UMI、REVER、农业应用、Pro-HOI |
| [4-相关系统与基础设施（11篇）.md](02-robot-manipulation(机器人操控)/notes/UMI/4-相关系统与基础设施（11篇）.md) | ALOHA/Mobile ALOHA、π₀、Diffusion Policy、DP3、MimicGen、RoboVerse 等 |
| [UMI-FT 论文解析：21D动作向量与Diffusion Policy的价值.md](02-robot-manipulation(机器人操控)/notes/UMI/UMI-FT%20论文解析：21D动作向量与Diffusion%20Policy的价值.md) | UMI-FT 专题深度解析 |
| [UMI论文Q&A.md](02-robot-manipulation(机器人操控)/notes/UMI/UMI论文Q&A.md) | UMI 论文细节问答 |
| [`_原始素材/UMI论文读书报告.md`](02-robot-manipulation(机器人操控)/notes/UMI/_原始素材/UMI论文读书报告.md) | 原始未拆分的完整报告（2232 行） |

## 03-自动驾驶

世界模型、强化学习、扩散模型在自动驾驶中的应用。

### 世界模型

| 文件 | 说明 |
|------|------|
| [World_Model_Survey_Autonomous_Driving.md](03-autonomous-driving(自动驾驶)/notes/world-models/World_Model_Survey_Autonomous_Driving.md) | 自动驾驶世界模型综述 |
| [综合对比报告_WorldModel.md](03-autonomous-driving(自动驾驶)/notes/world-models/综合对比报告_WorldModel.md) | 世界模型综合对比报告 |
| [A_视频生成型_report.md](03-autonomous-driving(自动驾驶)/notes/world-models/A_视频生成型_report.md) | A 类：视频生成型世界模型 |
| [B_3D占用神经场景型_report.md](03-autonomous-driving(自动驾驶)/notes/world-models/B_3D占用神经场景型_report.md) | B 类：3D 占用神经场景型 |
| [C_端到端AD集成型_report.md](03-autonomous-driving(自动驾驶)/notes/world-models/C_端到端AD集成型_report.md) | C 类：端到端 AD 集成型 |
| [D_仿真导向型_report.md](03-autonomous-driving(自动驾驶)/notes/world-models/D_仿真导向型_report.md) | D 类：仿真导向型 |
| [E_基础工业模型_report.md](03-autonomous-driving(自动驾驶)/notes/world-models/E_基础工业模型_report.md) | E 类：基础工业模型 |
| [F_NeRF_3DGS型_report.md](03-autonomous-driving(自动驾驶)/notes/world-models/F_NeRF_3DGS型_report.md) | F 类：NeRF / 3DGS 型 |
| [G_LLM_VLM融合型_report.md](03-autonomous-driving(自动驾驶)/notes/world-models/G_LLM_VLM融合型_report.md) | G 类：LLM / VLM 融合型 |
| [H_经典基础_report.md](03-autonomous-driving(自动驾驶)/notes/world-models/H_经典基础_report.md) | H 类：经典基础世界模型 |

### 强化学习

| 文件 | 说明 |
|------|------|
| [Survey_RL_自动驾驶.md](03-autonomous-driving(自动驾驶)/notes/RL/Survey_RL_自动驾驶.md) | RL 在自动驾驶中的应用综述 |
| [Survey_潜空间RL_自动驾驶.md](03-autonomous-driving(自动驾驶)/notes/RL/Survey_潜空间RL_自动驾驶.md) | 潜空间 RL 在自动驾驶中的应用综述 |
| [DreamerAD_论文解读.md](03-autonomous-driving(自动驾驶)/notes/RL/DreamerAD_论文解读.md) | DreamerAD 论文深度解读 |

### 扩散模型 × 自动驾驶

| 文件 | 说明 |
|------|------|
| [DiffusionDrive_整合深度解读.md](03-autonomous-driving(自动驾驶)/notes/diffusion-for-AD/DiffusionDrive_整合深度解读.md) | DiffusionDrive 整合深度解读 |
| [RAD_RAD-2_深度系统解读.md](03-autonomous-driving(自动驾驶)/notes/diffusion-for-AD/RAD_RAD-2_深度系统解读.md) | RAD / RAD-2 深度系统解读 |

## 04-3D 视觉与重建

3D 视觉、神经辐射场与高斯重建相关研究。

| 文件 | 说明 |
|------|------|
| [ffn_vggt_gs_reconstruction_report.md](04-3d-vision(3D视觉与重建)/notes/ffn_vggt_gs_reconstruction_report.md) | FFN / VGGT / 高斯重建综合报告 |

---

> 四个主题的逻辑关系：**扩散模型**是底层算法基础 → **VLA 基础模型**将扩散模型（Flow Matching）与视觉语言模型结合 → **UMI 项目**是扩散策略在真实机器人操控中的具体落地；**3D 视觉**为机器人感知和自动驾驶提供空间理解能力。
