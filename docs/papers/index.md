# 论文笔记

这里记录了我阅读的论文及分析报告。

---

## 论文列表

### 计算机视觉 - 目标检测

- [**DINO: 无监督知识蒸馏解锁视觉Transformer潜力**](DINO/paper_analysis_zh.md)  
  自监督学习框架,通过动量教师和多裁剪策略训练 ViT,实现无监督物体分割能力

- [**Stable-DINO: 基于位置监督损失的DETR稳定匹配方法**](Stable_DINO/paper_analysis_zh.md)  
  通过位置监督损失解决DETR训练不稳定问题,显著加速收敛并提升性能

- [**SAM-DETR++: 通过语义对齐匹配加速DETR收敛**](SAM_Detr/paper_analysis_zh.md)  
  提出语义对齐器模块,仅12个训练周期即达到44.8% AP,收敛速度提升超过97%

- [**VLDet: 多粒度视觉语言对齐的开放词汇目标检测**](VLDet/paper_analysis_zh.md)  
  通过VL-PUB模块和三明治式对齐损失,实现高效的开放词汇目标检测

### Transformer架构优化

- [**Revisiting [CLS]: 视觉Transformer中令牌解耦的架构再思考**](revisiting_CLS/paper_analysis_zh.md)  
  通过令牌特化设计解耦[CLS]与图像块令牌,在密集预测任务上显著提升性能

- [**Native Sparse Attention: 硬件对齐的原生可训练稀疏注意力**](Naive_Sparse_Attention/paper_analysis_zh.md)  
  提出NSA机制实现硬件友好的线性复杂度计算,在64K长序列上实现11.6倍加速

### 大语言模型训练

- [**OPUS: 面向大语言模型的优化器感知动态数据选择**](Opus/paper_analysis_zh.md)  
  首个优化器感知的数据选择框架,通过Ghost技巧和CountSketch投影实现高效效用评分

---

*论文报告会持续更新，欢迎关注。*
