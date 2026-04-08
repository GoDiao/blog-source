# 项目

## RGB-to-RAW 图像重建

研究基于 Unprocessing 的逆 ISP 管道，重建设计轻量级 ReRAW 风格模型，将 PSNR 从 52.81 提升至 56.98，SSIM 从 0.91 提升至 0.95。

技术栈：Python, PyTorch, 图像处理, Inverse ISP, RAW 图像处理

## VLA / RAW-to-Action

评估 SpatialVLA 并在 Bridge 数据集上进行微调，将成功率从 34.5% 提升至 49.5%。共同设计 RawAdaptationModule（可学习的通道投影），将 4 通道 RAW 桥接至 RGB 预训练编码器，实现端到端 RAW-to-action 训练（0.21→0.36）。

技术栈：Python, PyTorch, 具身智能, SpatialVLA, Domain Adaptation

## Autonomous String AI Agent

开源项目，设计多智能体自动驾驶系统，采用闭环 Perception-Memory-Reasoning-Planning 架构。 Perception Agent 使用检测和地图工具，Memory Agent 维护驾驶规则数据库，Planning Agent 输出 6 点轨迹表示未来 3 秒路径。

技术栈：Python, vLLM, Qwen, LLaMA, Multi-Agent Systems

## vLLM KV Cache 优化

集成 CacheBlend 技术到 DeepResearch 推理管道，在 Ascend 910B 上实现比 vLLM prefix cache 更高的 KV hit rate。

技术栈：Python, vLLM, LMCache, Ascend 910B, KV Cache

## LLM 20 Questions

构建 GRPO 训练环境，实现多智能体协作架构（Questioner/Answerer/Analyzer），将准确率从 42% 提升至 83%。开发 RLAIF 奖励函数结合规则约束和 AI 反馈。

技术栈：Python, GRPO, LLaMA, Multi-Agent, RLAIF
