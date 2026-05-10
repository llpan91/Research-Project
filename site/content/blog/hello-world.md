---
title: "Hello World: 研究博客启动"
date: 2026-05-10
draft: false
tags: ["meta"]
summary: "记录本站搭建的动机与技术选型。"
---

## 为什么建这个站

在研究过程中积累了大量笔记和论文阅读记录，分散在本地 Markdown 文件中。建立这个网站是为了：

1. **系统化整理** — 将零散笔记组织成可导航的知识结构
2. **公开分享** — 帮助同领域的研究者
3. **倒逼输出** — 写作是最好的学习方式

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 静态站点生成器 | Hugo | 编译速度快，Go 模板灵活 |
| 主题 | PaperMod | 极简、暗色模式、中文友好 |
| 数学公式 | KaTeX | 比 MathJax 快 10x |
| 部署 | Docker (nginx) | 生产镜像 ~25MB |

## 内容规划

目前规划四个研究主题：

- 扩散模型（Diffusion Models）
- 机器人操控（Robot Manipulation）
- 自动驾驶（Autonomous Driving）
- 3D 视觉（3D Vision）

每个主题下包含论文阅读笔记、理论推导和实践记录。

## 数学公式测试

扩散模型的前向过程：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

ELBO 分解中的去噪目标：

$$L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2\sigma_t^2} \| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2 \right]$$

如果这些公式能正确渲染，说明 KaTeX 配置成功。
