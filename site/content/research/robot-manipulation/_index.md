---
title: "机器人操控 (Robot Manipulation)"
description: "UMI 系统、Diffusion Policy 与 VLA 大模型"
ShowToc: true
---

## 概述

研究低成本、高泛化性的机器人操控方法。以 **UMI（Universal Manipulation Interface）** 为核心，探索从人类示教到机器人策略的完整流水线。

关键方向：
1. **UMI 系统** — 手持夹爪数据采集、跨机体部署
2. **Diffusion Policy** — 扩散模型作为视觉运动策略
3. **VLA 模型** — π0 系列（Vision-Language-Action）
4. **硬件扩展** — 触觉感知、灵巧手、移动平台

## UMI 系列论文

{{< paper-card title="UMI" authors="Chi et al." year="2024" description="Universal Manipulation Interface — 手持夹爪 + Diffusion Policy 的数据采集与部署框架" >}}

{{< paper-card title="UMI-FT" authors="2024" description="力/扭矩感知扩展，21D 动作向量" >}}

{{< paper-card title="FastUMI" authors="2024" description="简化硬件设计，降低成本" >}}

{{< paper-card title="DexUMI" authors="2024" description="将 UMI 范式扩展到灵巧手操控" >}}

{{< paper-card title="UMI on Legs" authors="2024" description="移动平台上的 UMI 操控" >}}

{{< paper-card title="UMI-on-Air" authors="2024" description="跨具身形态的策略迁移" >}}

## VLA 大模型

{{< paper-card title="π0" authors="Physical Intelligence" year="2024" arxiv="2410.24164" description="Vision-Language-Action Flow Model，通用机器人控制" >}}

{{< paper-card title="π0.5" authors="Physical Intelligence" year="2025" description="π0 升级版，更强泛化能力" >}}

{{< paper-card title="π0-FAST" authors="Physical Intelligence" year="2025" description="基于 FAST tokenizer 的高效动作表示" >}}

## 相关框架

- ALOHA / Mobile ALOHA — 低成本双臂遥操作
- MimicGen — 数据增广框架
- RoboVerse / InfiniteWorld — 仿真平台
