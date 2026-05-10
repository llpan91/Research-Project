# UMI 项目学习笔记索引

---

## 01-扩散模型

扩散模型的理论基础、训练细节、与强化学习的融合。

| 文件 | 说明 |
|------|------|
| **扩散模型：从生成理论到决策智能.md** | 整合文档，涵盖数学原理、训练优化、RL 融合、机器人/自驾应用全景 |
| `_原始素材/扩散模型详解.md` | 原始笔记：基础理论 + 算法演进 + 机器人/自驾应用 |
| `_原始素材/扩散模型图像生成：数据构建、训练、损失与优化.md` | 原始笔记：DDPM 训练流程、损失函数、优化器配置 |
| `_原始素材/扩散模型与强化学习的关系.md` | 原始笔记：RL 与扩散模型的四种融合范式 |

## 02-VLA 与基础模型

Physical Intelligence 系列论文及 VLA 模型族研究。

| 文件 | 说明 |
|------|------|
| Physical_Intelligence_VLA_Reading_Report.md | 9 篇 VLA 论文综合阅读报告（PI 核心 4 篇 + 学术界改进 5 篇） |
| PI_Series_Deep_Reading_Report.md | PI 技术演进线深度解析：π₀ → π₀.5 → FAST → π₀-FAST |

## 03-UMI 项目

UMI 系列 34 篇论文读书报告（按类别拆分）+ 专题笔记。

| 文件 | 说明 |
|------|------|
| **0-UMI原论文与全景概览.md** | UMI 1.0 原论文详解 + 34 篇论文分类索引 + 技术脉络总结 |
| 1-UMI系列核心扩展（9篇）.md | UMI-FT、FastUMI、MV-UMI、UMI-on-Air、UMI on Legs、ActiveUMI、DexUMI、HoMMI |
| 2-硬件改进与触觉感知（6篇）.md | TacUMI、FARM、UMIGen、exUMI、TacThru-UMI、Hoi! |
| 3-数据策略与学习框架（6篇）.md | RDT2、RwoR、Label-UMI、REVER、农业应用、Pro-HOI |
| 4-相关系统与基础设施（11篇）.md | ALOHA/Mobile ALOHA、π₀、Diffusion Policy、DP3、MimicGen、RoboVerse 等 |
| UMI-FT 论文解析：21D动作向量与Diffusion Policy的价值.md | UMI-FT 专题深度解析 |
| UMI论文Q&A.md | UMI 论文细节问答 |
| `_原始素材/UMI论文读书报告.md` | 原始未拆分的完整报告（2232 行） |

---

> 三个主题的逻辑关系：**扩散模型**是底层算法基础 → **VLA 基础模型**将扩散模型（Flow Matching）与视觉语言模型结合 → **UMI 项目**是扩散策略在真实机器人操控中的具体落地。
