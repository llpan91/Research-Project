---
title: "扩散模型 (Diffusion Models)"
description: "从理论基础到决策智能的生成模型研究"
ShowToc: true
---

## 概述

扩散模型通过逐步加噪-去噪的过程实现高质量生成，是当前最强大的生成模型范式之一。我的研究聚焦于：

1. **理论基础** — DDPM、DDIM、Score SDE 的数学推导
2. **加速采样** — 一致性模型、蒸馏方法
3. **条件生成** — Classifier-Free Guidance、Latent Diffusion
4. **决策应用** — Diffusion Policy、Diffuser、DPPO

## 核心论文

{{< paper-card title="DDPM" authors="Ho et al." year="2020" arxiv="2006.11239" description="Denoising Diffusion Probabilistic Models — 奠基工作" >}}

{{< paper-card title="DDIM" authors="Song et al." year="2020" arxiv="2010.02502" description="确定性采样加速，统一 DDPM 与 Score Matching" >}}

{{< paper-card title="Score SDE" authors="Song et al." year="2021" arxiv="2011.13456" description="统一连续时间框架，SDE/ODE 双视角" >}}

{{< paper-card title="Classifier-Free Guidance" authors="Ho & Salimans" year="2022" arxiv="2207.12598" description="无分类器引导，条件生成的标准范式" >}}

{{< paper-card title="Latent Diffusion (LDM)" authors="Rombach et al." year="2022" arxiv="2112.10752" description="潜空间扩散，Stable Diffusion 基础" >}}

{{< paper-card title="Consistency Models" authors="Song et al." year="2023" arxiv="2303.01469" description="单步/少步生成，一致性训练与蒸馏" >}}

{{< paper-card title="Flow Matching" authors="Lipman et al." year="2023" arxiv="2210.02747" description="直线路径条件流匹配，训练更稳定" >}}

{{< paper-card title="Diffusion Policy" authors="Chi et al." year="2023" arxiv="2303.04137" description="将扩散模型作为机器人策略表示" >}}

{{< paper-card title="Diffuser" authors="Janner et al." year="2022" arxiv="2205.09991" description="扩散模型用于轨迹规划" >}}

{{< paper-card title="DPPO" authors="Ren et al." year="2024" arxiv="2409.00588" description="扩散策略的在线强化学习微调" >}}

## 学习资源

- 数学导论翻译（基于 Calvin Luo 教程）
- DDPM/DDIM 对比笔记
- 采样加速：从 DDPM 到一步生成的算法演进
