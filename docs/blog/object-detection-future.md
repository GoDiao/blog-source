---
hide:
  - toc
---

# 目标检测何去何从

从 Viola-Jones 到 DETR，从滑动窗口到 Transformer，目标检测经历了二十年的范式更迭。站在 2025 年的节点回望，这个领域正处在一个微妙的十字路口：CNN 与 Transformer 的路线之争尚未终结，基础模型（Foundation Model）又在重新定义「检测」这件事的边界。

---

## 1. 一段简短的编年史

要理解「何去何从」，先要理解「从何而来」。

### 1.1 滑动窗口时代（2001–2013）

目标检测的起点可以追溯到 Viola-Jones 检测器（2001）——Haar 特征 + AdaBoost + 级联分类器，实时检测人脸。之后 HOG + SVM（Dalal & Triggs, 2005）将检测能力扩展到行人，DPM（Deformable Part Model, 2008–2010）引入可变形部件建模，在 PASCAL VOC 上把 mAP 推到了 30–40%。

这些方法的共同特征：**手工特征 + 判别式分类器 + 滑动窗口穷举**。计算量大、泛化性差、只能检测特定类别。

### 1.2 双阶段革命（2014–2017）

深度学习改变了游戏规则。

**R-CNN**（Girshick et al., 2014）开创了「区域提案 + CNN 分类」的范式：先用 Selective Search 生成 ~2000 个候选区域，再用 CNN 提取特征并分类。虽然 mAP 大幅提升，但速度极慢（一张图 47 秒）。

**SPPNet** 和 **Fast R-CNN**（2014–2015）通过共享卷积特征图，将速度提升了 100 倍以上。

**Faster R-CNN**（Ren et al., 2015）是真正的里程碑——用 RPN（Region Proposal Network）替代了 Selective Search，实现了端到端的区域提案和分类。检测速度首次逼近实时，mAP 在 COCO 上超过了 40%。

双阶段方法的核心优势是**精度高**——第一阶段筛选候选，第二阶段精细分类和回归。但两次前向传播的开销使得实时应用受限。

### 1.3 单阶段的崛起（2016–2020）

**YOLO**（Redmon et al., 2016）提出了一个激进的想法：把检测当作一个回归问题，一次前向传播同时预测所有边界框。YOLOv1 在 45 FPS 下达到了 63.4% mAP（VOC 2007）——速度碾压双阶段，但精度（尤其是小目标）还有差距。

**SSD**（Liu et al., 2016）在不同尺度的特征图上做检测，改善了多尺度性能。

**YOLOv2/v3**（2017–2018）引入了 anchor boxes、多尺度预测、Darknet-53 等改进，逐步缩小了与双阶段的精度差距。

**RetinaNet**（Lin et al., 2017）提出了 Focal Loss，解决了单阶段检测器中正负样本极度不均衡的核心问题——这是单阶段方法第一次在精度上真正超越双阶段。

**YOLOv4/v5**（2020）进一步将工程优化推向极致：CSPDarknet、SPP、PANet、Mosaic 增强、CIoU Loss……YOLOv5 成为了工业部署的事实标准。

### 1.4 无 Anchor 的探索（2018–2021）

Anchor-based 方法依赖预设的 anchor 尺度和比例，需要针对不同数据集调参，且 anchor 与 ground truth 的匹配机制（IoU 阈值）引入了额外的超参数。

**CornerNet**（2018）将检测转化为预测边界框的一对角点，完全去掉了 anchor。

**CenterNet**（2019）更进一步——把目标看作其中心点，预测中心点热力图 + 尺寸回归，架构极度简洁。

**FCOS**（Tian et al., 2019）用全卷积的方式逐像素预测：每个特征图位置直接回归到边界框四条边的距离，配合 centerness 分支抑制低质量预测。FCOS 证明了 anchor-free 可以在精度上与 anchor-based 方法持平甚至超越。

---

## 2. Transformer 登场：DETR 的冲击

2020 年，FAIR 发表了 [*End-to-End Object Detection with Transformers*](https://arxiv.org/abs/2005.12872)（Carion et al., 2020）。这篇论文的目标不仅是提升性能，而是从根本上**简化检测流程**。

### 2.1 DETR 的核心思想

传统检测器（无论是 anchor-based 还是 anchor-free）都需要一系列手工设计的组件：

- Anchor 设计 / 候选区域生成
- 正负样本分配策略（IoU 阈值、top-k 等）
- NMS（非极大值抑制）后处理
- 多尺度特征融合策略

DETR 用 Transformer 的**集合预测（Set Prediction）**范式替代了所有这些：

1. CNN backbone 提取特征
2. Transformer encoder 增强特征表示
3. Transformer decoder 用 $N$ 个 **object queries** 并行预测 $N$ 个检测结果
4. **二分图匹配（Hungarian Matching）** 将预测和 ground truth 做一对一匹配
5. 匹配后的预测直接计算分类和回归损失

没有 anchor，没有 NMS，没有手工的正负样本分配。网络自己学会「哪些 query 对应哪些目标」。

### 2.2 DETR 的优雅与代价

从理论角度，DETR 是目标检测领域最优雅的架构之一。它把检测问题干净地形式化为集合预测问题，消除了大量启发式设计。

但代价也很明显：

**收敛极慢。** 原始 DETR 在 COCO 上需要 500 个 epoch 才能达到与 Faster R-CNN 持平的精度。相比之下，Faster R-CNN 只需要 12–36 个 epoch。核心原因是 object queries 需要通过训练逐渐学会「关注图像的哪个区域」——在训练初期，query 的注意力模式是弥散的。

**小目标检测弱。** DETR 在大目标上表现出色（AP_L），但在小目标上（AP_S）明显弱于多尺度特征金字塔方法。原因是 DETR 只在最后一个特征图上做检测，没有显式的多尺度机制。

**计算开销大。** Transformer 的自注意力机制复杂度是 $O(N^2)$（$N$ 为 token 数量），在高分辨率特征图上计算量巨大。

### 2.3 DETR 的进化路线

DETR 的提出引发了后续三年的密集研究：

**Deformable DETR**（Zhu et al., 2021）是第一个重要改进——用可变形注意力替代全局注意力，每个 query 只关注少量学习到的参考点。收敛速度从 500 epoch 降至 50 epoch，同时支持多尺度特征。

**DAB-DETR**（Liu et al., 2022）将 object query 显式地建模为动态 anchor boxes（位置 + 尺寸），让 queries 有更好的空间先验，进一步加速收敛。

**DN-DETR**（Li et al., 2022）引入去噪训练（Denoising Training），通过向 queries 添加噪声并让网络恢复，加速匹配稳定化。

**DINO**（Zhang et al., 2022）结合了 DAB-DETR 的动态 queries、DN-DETR 的去噪训练、以及对比学习的去重策略，成为 DETR 家族的新 SOTA——在 COCO 上以 ResNet-50 backbone 达到了 63.2 AP，用 SwinL 达到了 63.7 AP。

**SAM-DETR++** 提出语义对齐匹配，仅 12 个训练周期即达到 44.8% AP，收敛速度提升超过 97%。

**Stable-DINO** 通过位置监督损失解决 DETR 训练不稳定问题，显著加速收敛并提升性能。

这些工作的共同趋势是：**在不牺牲 DETR 架构简洁性的前提下，通过各种先验注入加速收敛和提升性能**。

---

## 3. YOLO 王朝：极致工程的胜利

与 DETR 的学术优雅形成鲜明对比的，是 YOLO 系列的工程驱动路线。

### 3.1 从 YOLOv1 到 YOLOv8：十年进化

YOLO 的核心理念始终没变：**一个网络、一次前向、所有检测**。但每个版本都在架构、损失函数、数据增强上做了细致优化：

| 版本 | 年份 | 关键改进 | COCO AP |
|:-----|:-----|:---------|:--------|
| YOLOv1 | 2016 | 统一检测框架 | — (VOC) |
| YOLOv2 | 2017 | Anchor boxes, Darknet-19 | 48.1 |
| YOLOv3 | 2018 | 多尺度预测, Darknet-53 | 55.3 |
| YOLOv4 | 2020 | CSP, SPP, PANet, Mosaic | 62.8 |
| YOLOv5 | 2020 | 工程优化, PyTorch 原生 | 67.2 |
| YOLOv6 | 2022 | RepVGG, 解耦头 | 64.4 |
| YOLOv7 | 2022 | E-ELAN, 辅助头 | 69.7 |
| YOLOv8 | 2023 | C2f, Anchor-free, 分布式焦点损失 | 69.4 |

### 3.2 YOLOv8 的架构演进

YOLOv8（Ultralytics, 2023）代表了 YOLO 系列的一个重要转折：

**Anchor-free 设计。** YOLOv8 彻底放弃了 anchor boxes，采用 task-aligned 正负样本分配策略，简化了部署流程。

**C2f 模块。** 替代了之前的 C3 模块，在保持梯度流的同时减少了计算量。C2f（Cross Stage Partial Bottleneck with 2 convolutions, fused）借鉴了 CSPNet 的分流思想，但更加轻量。

**解耦检测头（Decoupled Head）。** 分类和回归分支使用独立的卷积层，避免了两个任务之间的特征竞争。回归分支使用 DFL（Distribution Focal Loss），将边界框回归建模为概率分布而非点估计。

### 3.3 YOLO11 与 RT-DETR：2024 的新格局

**YOLO11**（Ultralytics, 2024）延续了 YOLO 系列的渐进式优化路线，引入了增强的注意力机制和优化的 C3k2 模块。提供 n/s/m/l/x 五种尺寸，在参数效率和精度上进一步平衡。

**RT-DETR**（BAAI, 2024）则代表了另一种思路：能否让 DETR 架构也达到实时速度？RT-DETR 通过高效的混合编码器和优化的 cross-attention 机制，在 COCO 上以 ResNet-50 达到了 53.1 AP @ 108 FPS（T4 GPU），首次证明了 Transformer-based 检测器可以与 YOLO 在速度上竞争。

---

## 4. 基础模型：重新定义「检测」

2023–2024 年，目标检测领域出现了一个新的变量：**视觉基础模型（Vision Foundation Models）**。

### 4.1 开放词汇检测（Open-Vocabulary Detection）

传统检测器只能在训练时见过的类别上工作。如果一个模型在 COCO（80 类）上训练，它永远检测不出第 81 类物体。

开放词汇检测（OVD）试图打破这个限制：用视觉-语言模型（VLM）将检测能力扩展到任意类别。

**Grounding DINO**（Liu et al., 2023）结合了 DINO 的检测能力和 GLIP 的 grounding 能力——输入一段文本描述（如「桌上的红色杯子」），模型就能定位对应的目标。它将目标检测从「闭集分类」变成了「开集定位」。

**VLDet** 用 VL-PUB 模块和三明治式对齐损失，实现了高效的开放词汇目标检测，无需针对新类别重新训练。

**Florence-2**（Microsoft, 2024）更激进——它是一个统一的视觉基础模型，通过 prompt-based 的方式同时处理检测、分割、描述、grounding 等多种任务。训练数据 FLD-5B 包含 50 亿标注，覆盖了前所未有的任务和场景多样性。

### 4.2 SAM 与检测的关系

**SAM（Segment Anything Model）**（Kirillov et al., 2023）本身不是一个检测器——它是一个分割模型。但它的出现深刻影响了检测领域：

- SAM 可以将检测结果（边界框）转换为精确的分割掩码，实现「检测 + 分割」的零成本集成
- SAM 的 promptable segmentation 范式（点击、框、文本 → 分割）启发了新的交互式检测流程
- SAM 在零样本分割上的强大泛化能力，让人开始思考：是否可以用 SAM 替代传统的 NMS 后处理？

### 4.3 检测即语言（Detection as Language）

最前沿的趋势是将目标检测重新定义为**视觉语言理解**任务：

- **传统范式**：输入图像 → 输出 `{class_id, bbox}` 的集合
- **新范式**：输入 `(图像, 文本描述)` → 输出匹配文本描述的区域

这种范式的好处是：

1. **天然支持开放词汇**——只要你换一段文字，就能检测新的类别
2. **支持指代表达理解（Referring Expression）**——「左边的猫」vs「右边的猫」
3. **统一多种视觉任务**——检测、分割、captioning、VQA 共享同一个模型

代价是：推理速度通常比纯视觉检测器慢一个数量级（需要文本编码器 + 视觉-语言交互），且对语言的依赖可能在对抗性场景下引入新的脆弱性。

---

## 5. 三条路线的对比

把当前目标检测的三条主要路线放在一起对比：

| 维度 | YOLO 系列 | DETR 系列 | 基础模型 |
|:-----|:----------|:----------|:---------|
| **精度（COCO AP）** | 50–70 | 55–65 | 40–60（开放设定） |
| **速度（FPS）** | 30–300+ | 10–100 | 1–30 |
| **类别限制** | 闭集 | 闭集 | 开集 |
| **后处理** | NMS | 无需 NMS | 需要 NMS |
| **训练数据** | 中等（COCO 级） | 中等 | 海量（亿级标注） |
| **部署难度** | 低 | 中 | 高 |
| **典型应用** | 嵌入式、实时 | 服务端、研究 | 通用视觉理解 |

没有哪条路线是银弹。选择取决于你的具体约束：

- **需要极致速度和确定性延迟？** YOLO 系列。从边缘设备（Jetson Nano）到服务器都能流畅运行。
- **需要干净的架构和不需要 NMS？** DETR 系列。特别适合后处理成为瓶颈的流水线。
- **需要检测未见过的类别？** 基础模型 / OVD 方法。在研究和探索性场景下无可替代。

---

## 6. 当前挑战与未来方向

### 6.1 评估体系的局限

COCO AP 已经统治目标检测评估十年了。但它在几个方面存在盲区：

**类别分布偏斜。** COCO 的 80 个类别远不能代表真实世界的物体多样性。一个在 COCO 上达到 60 AP 的模型，可能完全无法检测到日常场景中的大多数物体。

**尺度假设。** COCO 的图像分辨率和目标尺度分布相对集中，但实际应用中可能遇到极端尺度（卫星图像中的车辆、显微镜下的细胞）。

**忽视公平性。** 不同人群、不同文化背景下的物体分布差异巨大，但 COCO AP 不衡量这些。

### 6.2 数据瓶颈

高质量标注数据的获取成本是目标检测领域最大的瓶颈之一。标注一个边界框看似简单（几秒钟），但标注准确、一致的边界框需要训练有素的标注员和严格的质量控制流程。

几个可能的突破方向：

- **弱监督和半监督检测**——用图像级标签或少量精确标注 + 大量无标注数据来训练
- **合成数据**——用 3D 引擎或扩散模型生成带完美标注的合成数据
- **自动标注**——用强模型（如 Florence-2）为弱模型生成伪标签

### 6.3 3D 与时序检测

2D 边界框的检测能力已经趋于成熟，但真实世界是 3D 的、动态的：

- **3D 目标检测**（点云、BEV）在自动驾驶中至关重要，但数据获取成本远高于 2D
- **视频目标检测**需要处理运动模糊、遮挡、外观变化等时序挑战
- **时序一致性**——逐帧独立检测会导致闪烁和不稳定，如何保持跨帧一致性仍是开放问题

### 6.4 端侧部署

尽管 YOLO 系列已经非常轻量，但在真正的边缘设备（手机、无人机、IoT 设备）上部署仍然面临挑战：

- 量化后的精度损失（INT8 / FP16）
- 硬件特异性优化（NPU、DSP、GPU 架构差异巨大）
- 动态场景下的鲁棒性（光照变化、遮挡、运动模糊）

### 6.5 统一视觉模型

最根本的方向可能是：**目标检测不再是一个独立的任务**。

随着 Florence-2、GPT-4V、Gemini 等统一视觉语言模型的出现，「检测」正在被吸收为更通用的视觉理解能力的一部分。未来的模型可能不再区分「检测」、「分割」、「描述」、「推理」——它们只是同一个模型在不同 prompt 下的不同输出。

这对检测领域意味着什么？短期来看，专用的检测模型（如 YOLO）在特定部署场景下仍然不可替代。但长期来看，「检测」可能不再是一个独立的研究方向，而是统一视觉智能的一个子能力。

---

## 7. 写在最后

目标检测领域的演进，本质上是三条路线的竞争与融合：

1. **工程驱动的极致优化**（YOLO 系列）——追求速度与精度的帕累托最优
2. **架构驱动的范式创新**（DETR 系列）——追求简洁与理论的优雅
3. **数据驱动的规模涌现**（基础模型）——追求通用性与零样本泛化

这三条路线不是互斥的——RT-DETR 就是 DETR 架构 + YOLO 级速度的融合尝试；Grounding DINO 是 DETR + 语言模型的融合。未来的突破很可能出现在这些路线的交叉地带。

对于研究者而言，最重要的是理解每条路线的设计哲学和隐含假设，而不是盲目追逐 SOTA 数字。因为数字反映的是特定数据集和评估协议下的表现，而真实世界的需求远比 COCO benchmark 复杂。

---

## 参考资料

- Redmon, J., et al. *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016.
- Ren, S., et al. *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS 2015.
- Carion, N., et al. *End-to-End Object Detection with Transformers*. ECCV 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- Zhu, X., et al. *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR 2021.
- Zhang, H., et al. *DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection*. ICLR 2023.
- Lin, T.-Y., et al. *Focal Loss for Dense Object Detection*. ICCV 2017.
- Liu, S., et al. *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*. ECCV 2024.
- Kirillov, A., et al. *Segment Anything*. ICCV 2023.
- Xiao, B., et al. *Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks*. CVPR 2024.
- Zhao, Y., et al. *DETRs Beat YOLOs on Real-Time Object Detection*. CVPR 2024.

---

*写于 2025 年 1 月 · 最后更新 2025 年 4 月。*
