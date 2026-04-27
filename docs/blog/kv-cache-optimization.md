---
hide:
  - toc
---

# KV Cache 优化：从原理到实践

大语言模型的推理瓶颈不在计算，而在显存。KV Cache 作为 Transformer 自回归解码的核心优化，将注意力计算的复杂度从 $O(n^2)$ 降到了 $O(n)$，但代价是随序列长度线性增长的显存占用。当上下文从 4K 扩展到 128K、甚至 1M token 时，KV Cache 成了最硬的那块天花板。这篇文章系统梳理 KV Cache 的原理、瓶颈，以及从量化、剪枝到架构重构的各类优化方案。

---

## 1. 为什么需要 KV Cache？

Transformer 的自注意力机制在生成第 $t$ 个 token 时，需要计算它和前面所有 $t-1$ 个 token 的注意力分数：

$$
\text{Attention}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}
$$

注意观察：每生成一个新 token，Key 和 Value 矩阵只是**追加了一行**——前 $t-1$ 行完全没有变化。如果不做缓存，每个时间步都要重新计算所有已生成 token 的 K 和 V，这是 $O(t \cdot d_{model} \cdot n_{layers})$ 的重复计算。

KV Cache 的做法很直接：**把每个时间步算出的 K 和 V 存下来，下一步直接复用**。

```
时间步 t=1:  计算 K₁, V₁ → 存入 cache
时间步 t=2:  计算 K₂, V₂ → 追加到 cache → 注意力只用 Q₂ 和 [K₁, K₂]
时间步 t=3:  计算 K₃, V₃ → 追加到 cache → 注意力只用 Q₃ 和 [K₁, K₂, K₃]
...
```

这样每步只需要计算新 token 的 Q、K、V，然后拿 Q 去和缓存的 K/V 做注意力。**计算量从 $O(n^2 \cdot d)$ 降到 $O(n \cdot d)$**，代价是显存占用线性增长。

---

## 2. KV Cache 的显存瓶颈

一个 token 的 KV Cache 占用多少显存？精确计算如下：

$$
\text{KV Cache Size} = 2 \times n_{layers} \times d_{model} \times \text{sizeof(dtype)}
$$

- **2**：Key 和 Value 各一份
- **$n_{layers}$**：每层都有自己的 KV Cache
- **$d_{model}$**：隐藏维度，即 K/V 向量的维度
- **sizeof(dtype)**：数据类型占用的字节数（FP16 = 2 bytes）

以 Llama 2-70B 为例：

| 参数 | 值 |
|:-----|:---|
| $n_{layers}$ | 80 |
| $d_{model}$ | 8192 |
| 精度 | FP16 (2 bytes) |
| 单 token KV Cache | $2 \times 80 \times 8192 \times 2 = 2.5$ MB |

单 token 就要 2.5 MB。一个 4096 长度的序列，KV Cache 就吃掉 **10 GB**。而 70B 模型本身的权重（FP16）也才 140 GB。更关键的是，在服务场景下你需要同时处理多个请求（batch），显存消耗成倍增长。

**这就是为什么 KV Cache 优化如此重要——它直接决定了你能服务多长的上下文、多少并发请求。**

---

## 3. 量化：用更少的位存储 KV

最直觉的优化：降低 KV Cache 的数值精度。

### 3.1 FP8 量化

将 KV Cache 从 FP16（16 bit）压缩到 FP8（8 bit），显存直接减半。FP8 有两种编码格式：

- **E4M3**：4 bit 指数 + 3 bit 尾数，动态范围较小但精度更高，适合存 K
- **E5M2**：5 bit 指数 + 2 bit 尾数，动态范围更大但精度较低，适合存 V（V 的数值分布比 K 更分散）

实现上，在注意力计算前将 FP8 KV 反量化回 FP16/FP32，计算完成后将新 KV 量化回 FP8 存入 cache。量化和反量化的计算开销很小（逐元素操作），几乎不影响推理速度。

### 3.2 INT8 / INT4 量化

比 FP8 更激进。INT8 可以进一步配合逐通道量化（per-channel quantization）来保持精度：

$$
K_{quantized} = \text{round}\left(\frac{K}{\text{scale}_K}\right), \quad \text{scale}_K = \frac{\max(|K|)}{127}
$$

INT4 更极致但精度损失更大，通常只对 V 使用（V 对精度的容忍度高于 K，因为 V 通过注意力权重加权求和，单个 V 的误差会被平滑掉）。

### 3.3 实践建议

| 精度 | 显存节省 | 精度影响 | 适用场景 |
|:-----|:---------|:---------|:---------|
| FP16 (baseline) | — | — | 短上下文、精度敏感 |
| FP8 | 50% | 几乎无损 | 大多数场景 |
| INT8 | 50% | 轻微下降 | 中长上下文 |
| INT4 | 75% | 可感知下降 | 超长上下文、对质量不敏感 |

**经验法则：FP8 是性价比最高的选择。** 在绝大多数任务上，FP8 KV Cache 与 FP16 相比几乎无精度损失，但显存直接减半。

---

## 4. 剪枝与淘汰：不是所有 Token 都同等重要

注意力机制有个特性：**不是每个历史 token 对当前生成都有贡献**。大量 token 的注意力权重接近零，它们的 KV Cache 其实可以安全丢弃。

### 4.1 滑动窗口（Sliding Window Attention）

最简单的策略：只保留最近 $W$ 个 token 的 KV Cache，更早的直接丢弃。

$$
\text{Attention}(Q_t, K_{t-W:t}, V_{t-W:t})
$$

Mistral、Phi 等模型采用了这种方法。窗口大小 $W$ 通常在 4096–8192 之间。

优点是显存占用固定为 $O(W)$，不随序列长度增长。缺点是模型失去了对远处上下文的「记忆」——如果第 5 个 token 的信息对第 50000 个 token 很重要，滑动窗口会把它丢掉。

### 4.2 注意力 Sink

滑动窗口有一个微妙的 bug：Transformer 的注意力机制倾向于把大量权重分配给前几个 token（尤其是第一个 token），这些 token 起到了「注意力汇聚点」的作用。如果你把前几个 token 也滑出窗口，模型性能会骤降。

**StreamingLLM**（Xiao et al., 2024）的解决方案是：始终保留前 $S$ 个「sink token」+ 最近 $W$ 个 token：

$$
\text{Cache} = \{K_1, ..., K_S\} \cup \{K_{t-W:t}\}
$$

这样 KV Cache 大小固定为 $S + W$，同时保持了模型的稳定性。

### 4.3 动态淘汰（Eviction Policy）

更精细的策略：根据注意力权重动态决定保留哪些 token。

**H2O（Heavy-Hitter Oracle）** 的核心观察是：在注意力分布中，少量 token 接收了大部分注意力权重——这些「heavy hitter」对生成质量至关重要。H2O 在 cache 满时，淘汰累计注意力分数最低的 token。

$$
\text{Score}_i = \sum_{t=1}^{T} \alpha_{t,i}, \quad \alpha_{t,i} = \text{softmax}\left(\frac{Q_t K^T}{\sqrt{d_k}}\right)_i
$$

淘汰 Score 最低的 token，保留 Score 最高的。

**Scissorhands** 采用类似思路，但基于「token 对未来预测的重要性」做淘汰决策——通过观察当前 token 被后续 token 关注的频率来估计其重要性。

### 4.4 各策略对比

| 方法 | 显存 | 远程依赖 | 实现复杂度 |
|:-----|:-----|:---------|:-----------|
| 滑动窗口 | $O(W)$ | 无 | 低 |
| StreamingLLM | $O(S+W)$ | 部分（sink tokens） | 低 |
| H2O | $O(B)$（$B$ 为预算） | 有（保留重要 token） | 中 |
| 全量缓存 | $O(n)$ | 完整 | — |

---

## 5. 架构级优化：从 GQA 到 MLA

如果改存储策略还不够，那就改模型架构本身。

### 5.1 Multi-Query Attention (MQA)

标准的多头注意力（MHA）中，每个注意力头都有自己独立的 K 和 V。如果 $h$ 个头都共享同一份 K 和 V，显存就降为原来的 $1/h$：

```
MHA:  Q₁ K₁ V₁  Q₂ K₂ V₂  Q₃ K₃ V₃  ...  (每头独立 K, V)
MQA:  Q₁ K V     Q₂ K V     Q₃ K V     ...  (所有头共享 K, V)
```

MQA 由 Shazeer（2019）提出，在推理速度上有巨大优势（K/V 只需从内存读一次），但训练时精度会有轻微下降。

### 5.2 Grouped-Query Attention (GQA)

GQA 是 MHA 和 MQA 的折中：将 $h$ 个注意力头分成 $g$ 组，每组共享一份 K 和 V。

$$
\text{KV Heads} = \frac{h}{g}, \quad \text{显存} = \frac{1}{g} \times \text{MHA 显存}
$$

Llama 2-70B 使用 $h=64, g=8$，即 8 个 KV 头——KV Cache 大小是 MHA 的 $1/8$。Llama 3 系列延续了这一设计。

这是目前最主流的方案，在精度和效率之间取得了很好的平衡。

### 5.3 Multi-Head Latent Attention (MLA)

DeepSeek-V2 提出了 MLA，思路更加激进：不是减少 KV 头的数量，而是**把 KV 压缩到一个低维潜空间**。

对每个 token，不再存储完整的 K 和 V 向量（维度 $d_{model}$），而是存一个低维压缩向量 $\mathbf{c}$（维度 $d_c \ll d_{model}$）：

$$
\mathbf{c}_t = W_{down} \cdot \mathbf{h}_t, \quad K_t = W_{up,K} \cdot \mathbf{c}_t, \quad V_t = W_{up,V} \cdot \mathbf{c}_t
$$

KV Cache 只存 $\mathbf{c}_t$（$d_c$ 维），在注意力计算时才通过上投影恢复 K 和 V。DeepSeek-V2 中 $d_c = 512$，而 $d_{model} = 5120$——压缩比达到 10 倍。

MLA 的精妙之处在于：压缩是在训练阶段学到的（而非推理时后处理的），所以模型能够自适应地保留最重要的信息。

### 5.4 架构对比

以隐藏维度 $d=8192$、64 个注意力头为例：

| 方法 | 每层 KV Cache (per token) | 压缩比 | 精度影响 |
|:-----|:-------------------------|:-------|:---------|
| MHA | $2 \times 64 \times 128 = 16$ KB | 1× | — |
| GQA (8 groups) | $2 \times 8 \times 128 = 2$ KB | 8× | 几乎无损 |
| MQA | $2 \times 128 = 256$ B | 64× | 轻微下降 |
| MLA ($d_c=512$) | $512 \times 2 = 1$ KB | 16× | 需要训练 |

---

## 6. 跨层共享：所有层都需要独立的 KV Cache 吗？

标准实现中，Transformer 的每一层都维护自己的 KV Cache。但研究表明，相邻层的 KV 表示高度相关。

**Cross-Layer Attention (CLA)** 的思路：让相邻层共享 KV Cache。具体来说，每隔 $k$ 层才计算并存储一份新的 KV，中间的层直接复用：

```
Layer 0: 计算 KV₀，存入 cache
Layer 1: 复用 KV₀
Layer 2: 复用 KV₀
Layer 3: 计算 KV₃，存入 cache
Layer 4: 复用 KV₃
...
```

这样 KV Cache 总量降为原来的 $1/k$。论文中 $k=2$ 或 $k=3$ 时，模型性能下降在可接受范围内。CLA 可以和 GQA/MLA 叠加使用，进一步压缩显存。

---

## 7. 分页 KV Cache：操作系统式的显存管理

在并发推理场景下，多个请求的 KV Cache 大小各不相同，且持续增长。传统的连续内存分配会导致严重的显存碎片——就像早期操作系统的内存管理问题。

**vLLM 的 PagedAttention** 借鉴了操作系统的虚拟内存分页机制：

- 将 KV Cache 划分为固定大小的「页」（page），每页存储固定数量 token 的 KV
- 维护一个页表（page table），记录每个序列的逻辑页到物理页的映射
- 新 token 的 KV 只需分配新的物理页，不需要连续的内存空间
- 序列结束后，其占用的页可以被立刻回收

```
序列 A:  [Page 3] → [Page 7] → [Page 12] → ...
序列 B:  [Page 1] → [Page 5] → [Page 8]  → ...
序列 C:  [Page 2] → [Page 4] → ...
                 ↑
           物理页可以不连续
```

PagedAttention 解决的核心问题：

- **显存碎片**：从 60–80% 的碎片率降到 <4%
- **共享前缀**：多个请求如果有相同的 system prompt，可以共享同一份物理页（Copy-on-Write）
- **动态调度**：可以根据当前显存使用情况灵活调度新请求

---

## 8. 离线与在线：Prefill 和 Decode 的不同优化策略

推理过程分为两个阶段，它们的计算特征完全不同：

### 8.1 Prefill 阶段

处理输入 prompt（可能很长，如 100K token），计算并缓存所有 token 的 KV。这个阶段是**计算密集型**的——大量 token 的 Q、K、V 可以并行计算，GPU 利用率很高。

优化重点是**吞吐量**：如何尽快处理完 prompt。常见技术包括：

- **Tensor Parallelism**：将注意力计算分布到多个 GPU
- **Chunked Prefill**：将长 prompt 分块处理，避免单次计算量过大导致 OOM
- **Flash Attention**：使用分块算法减少 HBM 访问次数，提升计算效率

### 8.2 Decode 阶段

逐个生成 token，每步只有 1 个新 token。这个阶段是**显存带宽密集型**的——计算量很小（只有 1 个 token 的矩阵乘法），但需要从显存中读取所有历史 token 的 KV Cache。

$$
\text{计算量} = O(d_{model}), \quad \text{内存读取量} = O(n \cdot d_{model})
$$

这就是所谓的 **Memory-Bound** 问题。GPU 的算力利用率可能只有 1–5%，瓶颈完全在显存带宽上。

优化重点：

- **Batch 推理**：同时处理多个请求，让 GPU 的每次内存读取都为多个请求服务
- **KV Cache 量化**：减少内存读取量（见第 3 节）
- **Speculative Decoding**：用小模型猜测多个 token，大模型一次验证多个，减少 decode 步数

---

## 9. 实际部署中的优化组合

在生产环境中，通常不会只用单一优化，而是多种技术叠加。以下是几种常见的组合策略：

### 9.1 高吞吐在线服务

```
GQA + PagedAttention + FP8 量化 + Continuous Batching
```

这是 vLLM / SGLang 等推理框架的标准配置。GQA 减少每 token 的 KV 大小，PagedAttention 高效管理显存，FP8 进一步压缩，Continuous Batching 最大化 GPU 利用率。

### 9.2 超长上下文推理

```
MLA + 滑动窗口 + H2O 动态淘汰 + Chunked Prefill
```

当上下文长度达到 128K+ 时，即使 GQA 也未必够用。MLA 提供更高压缩比，配合动态淘汰策略，将活跃 KV 控制在显存可承受范围内。

### 9.3 资源受限推理（单卡 / 消费级 GPU）

```
GQA + INT4 KV 量化 + CLA (k=2) + 滑动窗口
```

在 24 GB 显存的消费级 GPU 上运行 70B 级模型，需要最激进的压缩策略。

### 9.4 效果对比

以 Llama 3.1-70B（GQA, 80 层）为例，4096 长度序列的 KV Cache 显存占用：

| 优化组合 | KV Cache 显存 | 压缩比 |
|:---------|:-------------|:-------|
| 无优化 (MHA, FP16) | ~20 GB | 1× |
| GQA (8 KV heads) | ~2.5 GB | 8× |
| GQA + FP8 | ~1.25 GB | 16× |
| GQA + FP8 + CLA (k=2) | ~0.63 GB | 32× |
| GQA + INT4 + CLA (k=2) | ~0.31 GB | 64× |

从 20 GB 降到 0.31 GB——这意味着同样的 GPU 可以同时服务 64 倍多的请求。

---

## 10. 开放问题与未来方向

KV Cache 优化仍然是一个活跃的研究领域，几个值得关注的方向：

**跨请求 KV Cache 共享。** 在 RAG 场景中，多个请求可能共享相同的长文档前缀。目前 PagedAttention 已能实现物理页共享，但如何高效管理共享页的生命周期、处理前缀更新，仍需要工程上的打磨。

**训练感知的 KV 压缩。** 当前的量化、剪枝都是推理时的后处理。如果模型在训练时就知道 KV 会被压缩（如 MLA），它就能学到更具压缩性的表示。将压缩感知融入预训练是一个有前景的方向。

**长上下文评估方法论。** 如何评估 KV Cache 优化对模型能力的影响？当前的 NIAH（Needle in a Haystack）测试只检查信息检索能力，但实际应用中长上下文涉及推理、总结、多轮交互等更复杂的能力维度。一个全面的评估框架仍然缺失。

**硬件协同设计。** HBM 带宽是 decode 阶段的根本瓶颈。未来的推理芯片可能需要专门为 KV Cache 访问模式设计存储层级——例如更大量的片上 SRAM 直接缓存热点 KV 页。

---

## 总结

KV Cache 优化是大模型推理工程中最核心的课题之一。从原理上看，它是在「保留足够信息」和「最小化显存占用」之间寻找平衡点。

回顾整条优化路线：

- **量化**（FP8/INT4）是最简单、最通用的优化，几乎无理由不用
- **GQA** 是当前架构标配，在精度和效率间取得最佳平衡
- **动态淘汰**（H2O/StreamingLLM）是超长上下文的必选项
- **MLA** 代表了训练感知压缩的未来方向
- **PagedAttention** 是高效显存管理的工程基础

这些技术不是互斥的——最佳的部署方案是它们的精心组合。理解每个优化的原理和权衡，才能在实际场景中做出正确的选择。

---

## 参考资料

- Shazeer, N. *Fast Transformer Decoding: One Write-Head is All You Need*. arXiv 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie, J., et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP 2023.
- DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. 2024.
- Xiao, G., et al. *Efficient Streaming Language Models with Attention Sinks*. ICLR 2024.
- Zhang, Z., et al. *H2O: Heavy-Hit Oracle for Efficient Generative Inference of Large Language Models*. NeurIPS 2023.
- Kwon, W., et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.
- Brandon, W., et al. *Striped Attention: Faster Inference and Finetuning for Large Context Windows*. arXiv 2024.
- Liu, Z., et al. *Cross-Layer Attention Sharing*. arXiv 2024.

---

*写于 2025 年 1 月 · 最后更新 2025 年 4 月。*
