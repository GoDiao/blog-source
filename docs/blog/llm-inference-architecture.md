---
hide:
  - toc
---

# LLM 推理架构解析：从单机到集群的完整图景

大语言模型的推理远不止「把模型加载到 GPU 上跑 forward」那么简单。从单机单卡的离线推理，到支持数千并发请求的在线服务，再到跨数据中心的多模态推理集群——每一步都涉及复杂的系统设计和工程权衡。这篇文章尝试从第一性原理出发，梳理 LLM 推理架构的全貌。

---

## 1. 推理的基础：为什么是 Memory-Bound？

在进入架构细节之前，先理解一个基本事实：**LLM 推理不是算力问题，而是显存带宽问题**。

以一个 70B 参数模型为例，FP16 精度的权重占用 140 GB 显存。生成一个 token 时，需要将整份权重从 HBM 加载到计算单元，一次完整的权重读取约为 140 GB。如果 GPU 的显存带宽是 2 TB/s，一次权重读取耗时约 70 ms。而 H100 的峰值算力是 989 TFLOPS——即使只做一次矩阵乘法，算力也远未饱和。

```
预填充阶段（Prefill）：计算密集型
├── 一次性处理长 prompt
├── 大量并行计算
└── GPU 利用率 80–95%

解码阶段（Decode）：显存带宽密集型
├── 每次只处理 1 个 token
├── 权重反复从显存加载
└── GPU 利用率 1–10%
```

这解释了为什么 LLM 推理优化的核心不是「加速计算」，而是「减少权重和 KV Cache 的加载次数」。

---

## 2. 推理系统的三层架构

一个完整的 LLM 推理系统通常分为三层：

```
┌──────────────────────────────────────────────┐
│              接入层 (Gateway)                │
│  负载均衡 · 请求路由 · 认证鉴权 · 限流熔断   │
├──────────────────────────────────────────────┤
│              调度层 (Scheduler)              │
│  请求队列 · 连续批处理 · 抢占 · 填充策略     │
├──────────────────────────────────────────────┤
│              计算层 (Engine)                 │
│  模型并行 · KV Cache 管理 · 算子优化         │
└──────────────────────────────────────────────┘
```

### 2.1 接入层：流量的入口

接入层负责将外部请求分发到推理节点。最简单的场景是 DNS 轮询，但生产环境中通常需要：

- **Token-Based 路由**：根据请求的 token 数量分配到不同实例（长请求去大显存实例，短请求去小实例）
- **Session Affinity**：同一个对话的多轮请求始终路由到同一个后端，避免 KV Cache 的重复传输
- **智能负载均衡**：实时监测各实例的显存使用率、队列深度，动态调整路由

### 2.2 调度层：效率的关键

调度层决定了请求的排队、批处理和资源分配策略。这是整个系统中最具工程挑战的部分。

### 2.3 计算层：硬件的极限

计算层负责实际的模型前向传播，包括张量并行、KV Cache 分页管理、算子融合等底层优化。

---

## 3. 调度层核心：连续批处理（Continuous Batching）

### 3.1 为什么传统批处理不适用

在图像分类等传统任务中，批处理很简单——把 32 张图拼成一个大张量，一次 forward 出 32 个结果。但 LLM 推理有两个根本差异：

1. **序列长度不固定**：32 个请求的 prompt 长度从 10 token 到 10K token 不等
2. **自回归生成**：每个请求每次只产生 1 个 token，然后才能决定下一个 token 是什么——不同请求可能提前结束（如模型输出了 `<|endoftext|>`）

如果等所有请求都结束才释放 batch，短的请求会被长的请求拖累（"tail latency" 问题）。

### 3.2 连续批处理原理

Orca（Yu et al., 2022）提出了连续批处理（Iteration-Level Batching）——每个 iteration（即生成一个 token 的周期）都重新调度：

```
Iteration 0:  Batch [R1, R2, R3, R4]  → 各自生成第 1 个 token
Iteration 1:  Batch [R1, R2, R3, R4]  → R2 结束（生成 EOS）
Iteration 2:  Batch [R1, R3, R4, R5]  → R5 是新到达的请求
Iteration 3:  Batch [R1, R3, R4, R5, R6] → R6 加入
```

关键设计：**每个 iteration 结束都检查是否有请求完成或新请求到达，动态重组 batch**。这避免了传统批处理的 tail latency 问题。

### 3.3 调度策略的权衡

连续批处理中，调度器需要决定「哪些请求进入下一个 batch」。常见策略：

| 策略 | 优点 | 缺点 |
|:-----|:-----|:-----|
| FCFS（先来先服务） | 简单、公平 | 短请求被长请求阻塞 |
| SJF（短作业优先） | 最小化平均延迟 | 长请求可能饿死 |
| 优先级调度 | 付费用户优先 | 配置复杂 |
| 分桶调度（Chunked Prefill） | Prefill 和 Decode 交错 | 实现复杂 |

**vLLM 采用 FCFS + 动态抢占**：新到达的长 prompt 请求可以抢占正在 decode 的请求（将其 KV Cache 换出到 CPU），保证 GPU 不会闲置。

---

## 4. 请求分块：Chunked Prefill

### 4.1 Prefill 和 Decode 的矛盾

Prefill（处理 prompt）是计算密集的——一个 8K 的 prompt 需要大量矩阵乘法，GPU 几乎满载。Decode（生成 token）是带宽密集的——一次只处理一个 token，GPU 大部分时间在等数据。

如果 prefilling 和 decoding 在同一个 batch 中混合，就会出现问题：

```
场景 1: [P1(8K) + D2 + D3 + D4]
├── P1 占用了全部 GPU 算力做 prefill
└── D2/D3/D4 被迫等待 —— GPU 未充分利用

场景 2: [D1 + D2 + D3 + D4]
├── 所有请求都在 decode
└── GPU 算力大量闲置，带宽是瓶颈
```

### 4.2 Chunked Prefill

Sarathi（Agrawal et al., 2023）提出的方案：将长 prefill 拆成小块（chunks），和 decode 请求交错执行：

```
Iteration 0: [P1_chunk0 + D2 + D3 + D4] → P1 前 512 token 做 prefill
Iteration 1: [P1_chunk1 + D2 + D3 + D4] → P1 后 512 token 做 prefill
...直到 P1 prefill 完成

然后： [D1 + D2 + D3 + D4] → 所有请求一起 decode
```

这样 GPU 始终有充足的计算量，不会因为纯 decode 阶段而利用率暴跌。SGLang 和 vLLM 的最新版本都支持这种调度策略。

---

## 5. 并行策略：TP、PP 与 DP

### 5.1 张量并行（Tensor Parallelism, TP）

将模型的每一层切分到多个 GPU 上并行计算。以 Transformer 为例：

```
单层 MLP: 输入 x → 线性投影 W₁ → 激活 → 线性投影 W₂ → 输出

TP 切分:
GPU 0: x → W₁[:,0:d/2] → 激活 → W₂[0:d/2,:] → partial_out_0
GPU 1: x → W₁[:,d/2:d] → 激活 → W₂[d/2:d,:] → partial_out_1

AllReduce: partial_out_0 + partial_out_1 = 最终输出
```

TP 的通信开销是 AllReduce，每层的激活值需要在 GPU 间同步。TP 数量通常不超过 8（对应 NVLink 的拓扑）。

### 5.2 流水线并行（Pipeline Parallelism, PP）

将模型按层切分，每个 GPU 负责连续的若干层：

```
GPU 0:  Layer 0–9   → 输出传给 GPU 1
GPU 1:  Layer 10–19 → 输出传给 GPU 2
GPU 2:  Layer 20–29 → 输出传给 GPU 3
GPU 3:  Layer 30–39 → 最终输出
```

PP 的通信开销是点对点传输（P2P），比 AllReduce 小得多。但 PP 有「流水线气泡」问题——当 batch size 不够大时，GPU 会有空闲等待期。

**Micro-Batching** 是缓解气泡的方法：将一个 batch 分成多个 micro-batch，让它们像流水线一样交错执行。

### 5.3 数据并行（Data Parallelism, DP）

将完整模型复制到多个节点上，每个节点处理不同的请求：

```
GPU 0/1/2/3:  完整模型副本 A —— 处理请求组 1
GPU 4/5/6/7:  完整模型副本 B —— 处理请求组 2
GPU 8/9/10/11: 完整模型副本 C —— 处理请求组 3
```

DP 没有跨节点通信（每个副本独立），但显存占用是模型大小的倍数。

### 5.4 组合策略

实际部署中，三种并行方式组合使用：

```
以 405B 模型、8×8 GPU 集群为例：

TP = 8（每节点内 NVLink 互联）
PP = 8（跨节点）
DP = 8（共 64 个副本）

总 GPU = 8 × 8 × 8 = 512
```

```
以 70B 模型、8×GPU 单节点为例：

TP = 8（单节点内）
PP = 1
DP = 1

总 GPU = 8
```

选择并行策略时，核心权衡：

- **TP 增加**：通信 AllReduce 变重，但单请求延迟降低
- **PP 增加**：流水线气泡变多，但可支持更大模型
- **DP 增加**：总吞吐线性增长，但显存线性增长

---

## 6. 投机解码（Speculative Decoding）：用带宽换延迟

### 6.1 核心洞察

Decode 阶段每次只生成 1 个 token，但权重读取量是整个模型。如果能一次猜多个 token，然后让大模型一次验证，就能「均摊」权重读取的开销。

### 6.2 基本原理

```
草稿模型 M_draft（小模型，如 1–7B）:
1. 一次生成 K 个候选 token: [t₁, t₂, ..., t_K]

目标模型 M_target（大模型，如 70B）:
2. 一次前向传播，验证这 K 个 token
3. 找到第一个不匹配的位置 j
4. 接受 token [t₁, ..., t_{j-1}]，从位置 j 重新生成
```

**关键**：验证步骤可以在一次前向传播中完成（通过树状注意力）。如果草稿模型质量够高，平均每次前向能生成 2–3 个 token。

### 6.3 草稿模型选择

| 方案 | 优势 | 劣势 |
|:-----|:-----|:-----|
| 小模型（如 Llama-1B） | 通用性强 | 需要额外显存 |
| N-gram 回溯 | 零额外开销 | 只适用重复模式 |
| 模型内草稿（Medusa/EAGLE） | 不增加显存 | 需要训练 |

**Medusa**（Cai et al., 2024）的方法是在大模型顶部添加多个预测头，并行预测未来多个 token 的位置。所有预测头共享大模型的中间表示，不需要额外加载权重。

### 6.4 效果

在典型工作负载上，投机解码可以将 decode 延迟降低 1.5–2.5×。收益在生成长文本时更明显——因为 overhead（草稿模型的运行成本）被摊薄了。

---

## 7. 推理引擎对比

### 7.1 vLLM

- **核心创新**：PagedAttention，分页式 KV Cache 管理
- **优势**：高吞吐、显存碎片率低、开源生态最完善
- **适用**：通用在线服务、多并发场景
- **局限**：功能迭代快但稳定性仍在打磨中

### 7.2 TensorRT-LLM

- **核心创新**：英伟达官方优化，算子融合到极致
- **优势**：单请求延迟最低、FP8 性能最优、与硬件深度协同
- **适用**：延迟敏感型应用、英伟达硬件环境
- **局限**：闭源（部分）、功能迭代慢、调试困难

### 7.3 SGLang

- **核心创新**：结构化生成（Structured Generation）+ RadixAttention
- **优势**：KV Cache 复用效率极高（前缀树缓存）、支持复杂输出约束
- **适用**：Agent 场景（工具调用、多轮对话）、结构化输出
- **局限**：社区规模较小

### 7.4 DeepSpeed Inference

- **核心创新**：ZeRO-Inference 允许 GPU+CPU+NVMe 异构存储
- **优势**：可以用有限显存跑超大模型（如单卡 24GB 跑 175B）
- **适用**：显存受限环境、离线批处理
- **局限**：延迟较高、更适合吞吐而非在线服务

### 7.5 选择建议

| 场景 | 推荐引擎 | 理由 |
|:-----|:---------|:-----|
| 生产级在线服务 | vLLM | 生态成熟、功能全面 |
| 极致延迟要求 | TensorRT-LLM | 英伟达深度优化 |
| Agent/工具调用 | SGLang | 结构化生成 + 前缀复用 |
| 单卡跑大模型 | DeepSpeed | 异构存储支持 |

---

## 8. 多模态推理：从文本到视觉

多模态 LLM（如 GPT-4V、Claude 3、Qwen-VL）的推理架构在文本基础上增加了视觉编码器（通常是 ViT）：

```
文本输入 → Tokenizer → Text Embedding
                          ↘
图像输入 → ViT Encoder → Visual Embedding → Projector → Text Embedding
                          ↗
                          ↓
                   LLM Decoder（标准 Transformer）
                          ↓
                     输出文本
```

### 8.1 视觉编码器的瓶颈

ViT 处理高分辨率图像时，token 数量巨大：

- 224×224 图像、patch size 14 → $(224/14)^2 = 256$ 个视觉 token
- 1344×1344 图像、patch size 14 → $(1344/14)^2 = 9216$ 个视觉 token

视觉 token 的 prefill 阶段需要巨大的计算量。优化策略包括：

- **动态分辨率**：根据图像内容自动选择分辨率（如 Qwen-VL 的 dynamic resolution）
- **视觉 token 压缩**：用可学习的压缩层将视觉 token 数量从 N 降到 M（如 LLaVA 的 token merging）
- **延迟加载**：视觉编码器的权重可以常驻显存（较小），而 LLM 权重按需加载

### 8.2 视频推理

视频帧的连续处理需要考虑时间维度的 KV Cache 复用：

```
Frame 1: 视觉编码 → KV Cache [V₁]
Frame 2: 视觉编码 → KV Cache [V₁, V₂]（V₁ 可复用）
Frame 3: 视觉编码 → KV Cache [V₁, V₂, V₃]
```

但如果视频很长（如 1 小时的直播流），KV Cache 会无限增长。此时需要滑动窗口策略（只保留最近 N 帧的 KV）。

---

## 9. 部署模式：云、边、端

### 9.1 云端部署

```
云端部署（数百到数千 GPU）
├── 模型：70B–405B+
├── 并发：1000+ QPS
├── 延迟要求：<100 ms / token（首 token <2s）
├── 硬件：A100/H100 × 8–512
└── 框架：vLLM / TensorRT-LLM / TGI
```

核心挑战：弹性伸缩、多租户隔离、成本优化（ spot instance + checkpoint 恢复）。

### 9.2 边缘部署

```
边缘部署（单节点 4–8 GPU）
├── 模型：7B–70B
├── 并发：10–100 QPS
├── 延迟要求：<50 ms / token
├── 硬件：A10/L4 × 4–8
└── 框架：vLLM / TensorRT-LLM
```

核心挑战：网络带宽受限（相比云端内部网络）、稳定性要求更高（难以热迁移）。

### 9.3 端侧部署

```
端侧部署（单 GPU 或无 GPU）
├── 模型：1B–7B（量化后）
├── 并发：单用户
├── 延迟要求：流式输出（<20 ms / token 感知流畅）
├── 硬件：Apple M4 / Snapdragon 8 Gen 3 / RTX 4090
└── 框架：llama.cpp / MLX / MLC-LLM
```

核心挑战：内存极其有限（几 GB 到几十 GB）、功耗约束、需要极端量化（INT4/INT3 甚至混合精度）。

---

## 10. 总结

LLM 推理架构的设计本质是在多个约束之间找最优解：

- **延迟 vs 吞吐**：低延迟要求小 batch、大 TP；高吞吐要求大 batch、大 DP
- **显存 vs 精度**：更大的 KV Cache 意味着更长的上下文，但也意味着更少的并发
- **通用性 vs 性能**：通用框架（vLLM）方便但未必极致；专用优化（TensorRT-LLM）性能最好但灵活性差

回顾这篇文章的核心脉络：

1. **Decode 是带宽瓶颈** —— 优化的核心不是加速计算，而是减少内存访问量
2. **连续批处理是吞吐的关键** —— 每个 iteration 都动态重组 batch，避免 tail latency
3. **并行策略需要匹配场景** —— TP 降低单请求延迟，DP 提高总吞吐，PP 支撑大模型
4. **投机解码是延迟优化的利器** —— 用小模型的带宽换取大模型的 token 生成速度
5. **没有最好的引擎，只有最适合的引擎** —— 根据延迟要求、硬件环境、功能需求选择

推理系统的优化是一个永无止境的游戏。模型在变（MoE、多模态）、硬件在变（B200、TPUv6）、需求在变（Agent、长上下文）。理解底层原理，才能在这些变化中做出正确的架构决策。

---

## 参考资料

- Yu, G.-I., et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022.
- Kwon, W., et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.
- Agrawal, A., et al. *Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*. arXiv 2023.
- Leviathan, Y., et al. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
- Cai, T., et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. arXiv 2024.
- NVIDIA. *TensorRT-LLM User Guide*. 2024.
- Zheng, L., et al. *SGLang: Efficient Execution of Structured Language Model Programs*. arXiv 2023.
- Aminabadi, R.Y., et al. *DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale*. SC 2022.
- OpenAI. *GPT-4V System Card*. 2023.

---

*写于 2025 年 1 月 · 最后更新 2025 年 4 月。*
