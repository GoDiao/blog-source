---
hide:
  - toc
---

# 项目

## 研究项目

### [RGB-to-RAW 图像重建](./rgb-to-raw.md)

重建基于 Unprocessing 的逆 ISP 管道，设计轻量级 ReRAW 风格模型。

**关键成果**：PSNR 52.81 → 56.98 | SSIM 0.91 → 0.95

---

### [Vision-Language-Action (VLA)](./vla-raw-to-action.md)

评估 SpatialVLA 并在 Bridge 数据集上进行微调，实现端到端 RAW-to-action 训练。

**关键成果**：成功率 34.5% → 49.5% | 端到端性能 0.21 → 0.36

---

### [LMCache on Ascend 910B](./lmcache-ascend.md)

集成 CacheBlend 技术到 DeepResearch 推理管道，在华为 Ascend 910B 上实现高效 KV Cache 管理。

---

### [AgentDriver](./agentdriver.md)

迁移完整自主驾驶栈到华为 Ascend 平台，集成大语言模型进行决策。

---

## 开源项目

### [Paper Reader Agent](./my_opensource/paper-reader.md) ⭐

分层多智能体系统，用于学术论文深度分析和AI驱动的网络研究。

**GitHub**：[GoDiao/Paper-Reader](https://github.com/GoDiao/Paper-Reader)

---

### [AI Interview Agent](./my_opensource/ai-interview-agent.md) ⭐

智能面试助手系统，通过持久化候选人档案实现跨会话的成长追踪。

**GitHub**：[GoDiao/ai-interview-agent](https://github.com/GoDiao/ai-interview-agent)

---

## 竞赛项目

### [LLM 20 Questions](./llm-20-questions.md)

构建 GRPO 训练环境，实现多智能体协作架构。

**关键成果**：准确率 42% → 83%

---

### [WSDM Cup](./wsdm-cup.md)

多语言聊天机器人竞赛，排名 154/950。

---

## 技能总结

| 类别 | 内容 |
|------|------|
| **深度学习** | PyTorch, PEFT, MindSpeed-LLM |
| **LLM 框架** | vLLM, LMCache, Qwen, LLaMA |
| **计算机视觉** | Inverse ISP, RAW Processing, SpatialVLA |
| **AI 系统** | Agentic AI, Multi-Agent, RLAIF |
| **硬件平台** | Ascend 910B, GPU 优化 |
| **编程语言** | Python, SQL, TCL, JavaScript |
