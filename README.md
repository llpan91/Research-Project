# Research Notes

Multi-topic research project covering diffusion models, robot manipulation, autonomous driving, and 3D vision.

## Project Structure

```
topics/
├── 01-diffusion-models(扩散模型)/       # 扩散模型理论基础
│   ├── papers/                          # DDPM, DDIM, Flow Matching, Consistency Models 等
│   └── notes/                           # 理论笔记、数学推导、问题记录
│
├── 02-robot-manipulation(机器人操控)/   # 机器人操控与策略学习
│   ├── papers/
│   │   ├── UMI/                         # UMI 系列 (硬件扩展、灵巧手、移动操控)
│   │   ├── VLA(Physical-Intelligence)/  # Pi0 系列 + VLA 改进工作
│   │   └── diffusion-policy/            # Diffusion Policy 基础框架
│   └── notes/
│       ├── UMI/                          # UMI 论文解读与翻译
│       └── VLA/                          # VLA/Pi0 深度报告
│
├── 03-autonomous-driving(自动驾驶)/     # 自动驾驶全栈
│   ├── papers/
│   │   ├── world-models/                # A-H 八大类 (视频生成、3D占用、端到端、仿真等)
│   │   ├── diffusion-for-AD/            # DiffusionDrive, RAD, MagicDrive 等
│   │   └── RL/                          # DreamerAD 等强化学习方法
│   └── notes/
│       ├── world-models/                # World Model Survey + 分类报告
│       ├── diffusion-for-AD/            # DiffusionDrive/RAD 解读
│       └── RL/                          # RL for AD surveys
│
└── 04-3d-vision(3D视觉与重建)/         # 3D 视觉、重建、NeRF/3DGS
    ├── papers/
    └── notes/
```
