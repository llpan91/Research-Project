# A_视频生成型 论文分析报告（Part 3）

---

## [9] Panacea: Panoramic and Controllable Video Generation for Autonomous Driving

**机构/会议**：USTC / MEGVII Technology / Mach Drive，CVPR 2024

**核心贡献**：
1. 首个面向自动驾驶的全景可控多视角视频生成方法，兼顾时序一致性与跨视角连贯性
2. 提出分解式4D注意力机制（Intra-View + Cross-View + Cross-Frame），在计算可行范围内建模多视角时空关系
3. 两阶段生成流水线：先生成多视角图像帧，再沿时间轴扩展为视频

**方法要点**：基于Latent Diffusion Model，引入ControlNet注入BEV布局序列（含3D边界框、路网、相机姿态）实现细粒度布局控制；CLIP文本提示提供粗粒度天气/时间/场景属性控制；4D注意力分解避免了完整HWVT注意力的显存瓶颈。

**主要结果**：在nuScenes上验证，合成数据使BEV感知NDS指标提升+2.3（纯图像增强场景）和+5.8（视频增强场景），有效提升下游感知模型性能。

**优缺点**：优点是首次实现全景多视角可控视频生成，数据增强效果显著；缺点是空间分辨率偏低（256×512），帧率受限于2Hz注释频率，与高帧率真实需求仍有差距。

---

## [10] Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability

**机构/会议**：HKUST / Shanghai AI Lab / University of Tübingen，NeurIPS 2024

**核心贡献**：
1. 提出可泛化驾驶世界模型，支持跨域零样本泛化，在多数据集上超越同类方法
2. 设计两个新损失函数：动态增强损失（关注运动区域）与结构保留损失（高频细节），显著提升高分辨率预测保真度
3. 统一多模态动作控制接口，同时支持高层指令（Command/Goal Point）与低层操控（Trajectory/Angle/Speed）

**方法要点**：基于Stable Video Diffusion（SVD）初始化；引入潜变量替换（Latent Replacement）注入历史帧作为动态先验，支持长时自回归预测；两阶段训练：先学高保真预测，再冻结权重学习动作控制（LoRA）。

**主要结果**：在nuScenes上FID降低55%、FVD降低27%（对比最佳驾驶世界模型）；在70%以上对比测试中优于通用视频生成器；分辨率达576×1024、10Hz。

**优缺点**：优点是泛化能力强、分辨率高、动作控制多样；缺点是训练数据规模（1740h）仍低于GAIA-1（4700h），且多模态动作联合训练的一致性有待深入评估。

---

## [11] WoVoGen: World Volume-aware Video Generation for Driving Scenarios

**机构/会议**：arXiv 2024（具体机构见论文正文，未在前5页完整列出）

**核心贡献**：
1. 引入"世界体积"（World Volume）三维隐式表示，统一建模多视角空间几何结构
2. 将3D世界体积投影到各相机视角，从根本上保证多视角几何一致性
3. 在世界体积空间中进行时序建模，实现跨帧跨视角联合生成

**方法要点**：构建4D（空间+时间）体积特征场，通过相机内外参将体积特征投影到各视角2D特征图；在此统一空间中用扩散模型完成视频生成，避免了单纯依赖注意力机制维持多视角一致性的局限性；分辨率256×448，帧率2Hz。

**主要结果**：在nuScenes数据集上生成质量优于Drive-WM等方法，多视角几何一致性明显改善；但受限于较低分辨率与帧率，与更新方法相比有差距。

**优缺点**：优点是从3D几何层面解决多视角一致性，思路根本且优雅；缺点是不支持稀疏条件输入，分辨率和帧率偏低，推理效率有待提升。

---

## [12] DriveScape: Towards High-Resolution Controllable Multi-View Driving Video Generation

**机构/会议**：Tsinghua University / Sensetime Research / Northeastern University，arXiv 2024

**核心贡献**：
1. 首个支持高分辨率（576×1024）、高帧率（2~10Hz）、稀疏条件控制的多视角驾驶视频生成端到端框架，无需后处理
2. 提出双向调制Transformer（BiMOT），通过双向交叉注意力实现多种3D道路结构信息的精确对齐与协同
3. 统一模型同时学习多视角和时序一致性，支持key-view与neighbor-view的分层推理

**方法要点**：基于LDM，输入条件包括BEV地图、3D边界框、BEV关键帧、邻近相机视频；训练时混合多帧率（2~10Hz）数据并学习无条件帧嵌入；推理分两轮（key-view并行 → neighbor-view）实现高效跨视角一致生成；BiMOT包含两个方向的交叉注意力层与一个时序自注意力层。

**主要结果**：在nuScenes上达到SOTA，FID=8.34、FVD=76.39，显著优于WoVoGen、MagicDrive等方法；同时在感知任务上表现出色。

**优缺点**：优点是分辨率和帧率突破行业瓶颈，稀疏条件控制实用性强；缺点是仍为arXiv预印本，neighbor-view推理依赖key-view质量，存在误差传播风险。

---

## [13] WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens

**机构/会议**：GigaAI / Tsinghua University，arXiv 2024

**核心贡献**：
1. 首个面向通用场景（非仅驾驶）的世界模型，将世界建模重新定义为无监督视觉序列掩码预测任务
2. 提出空间时序分块Transformer（STPT），以局部时空窗口内的分块注意力高效捕获视频动态
3. 统一框架支持text-to-video、image-to-video、video inpainting、video stylization、action-to-video等多任务

**方法要点**：用VQGAN将视觉信号离散化为token；随机掩码部分token，STPT预测掩码token（类似BERT的并行解码，约10步即完成，比扩散模型快3~20倍）；文本用T5编码，动作（速度/方向）用MLP编码，通过空间交叉注意力融入模型；训练三元组为Visual-Text-Action数据。

**主要结果**：在自然场景和驾驶场景视频生成上均表现出色；推理速度约为扩散模型的3~20倍（10步 vs ~200步）；支持无条件/单条件/多条件灵活生成。

**优缺点**：优点是通用性强、推理高效、多任务统一；缺点是离散token化损失部分细节精度，视频生成质量在精细纹理上不及最新扩散模型，驾驶场景的可控精度弱于专用模型。
