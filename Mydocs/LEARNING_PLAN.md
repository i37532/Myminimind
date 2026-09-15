# MiniMind 学习计划

> 适用人群：有深度学习基础、熟悉 PyTorch 和 Transformer（如 BERT/ViT 的 encoder 结构），但对大语言模型（LLM）本身不了解的学习者。
>
> 目标：借助 MiniMind 这套从零实现、代码精简的 LLM 项目，补齐语言模型的核心知识，打通「预训练 → SFT → 强化学习对齐」的完整链路。

---

## 前置背景：你已有的基础 vs 需要补齐的缺口

你熟悉 PyTorch 和 Transformer 的 encoder（BERT / ViT），但 LLM 是 **Decoder-Only 自回归** 模型，需要补齐的核心概念包括：

- **自回归生成 / next-token prediction / teacher forcing**
- **Tokenizer（BPE）**、vocab、special tokens
- **预训练 → SFT → RL 的三阶段训练范式**（区别于单一监督任务）
- **现代 LLM 的推理优化**：KV Cache、RoPE、GQA、SwiGLU
- **解码策略**：temperature / top-p / top-k / repetition penalty

> 核心代码集中在 `model/model_minimind.py`（约 19KB，实现完整 GPT 风格模型），训练脚本在 `trainer/` 目录下每阶段一个文件，非常适合逐文件精读。

---

## 学习路线总览

```
阶段0 跑通推理 ──> 阶段1 精读模型结构 ──> 阶段2 Tokenizer ──> 阶段3 预训练
                                        (核心)                     (核心)
                                                                      │
阶段7 强化学习对齐 <── 阶段6 LoRA/蒸馏 <── 阶段5 SFT <── 阶段4 推理与解码
   (进阶可选)
```

---

## 阶段 0：跑通一次推理（1 天）

**目标**：先看效果、建立全局观，不要一上来就啃代码。

- 按 README「快速开始」章节下载 `minimind-3` 权重，跑 CLI 推理（`scripts/chat_api.py`）。
- 动手：换成自己的 prompt，观察输出，体会「续写」与「问答」的区别。

**检验标准**：能成功跑出模型输出，并能用一句话说出「大模型本质上在做什么」。

---

## 阶段 1：精读模型结构（2–3 天）⭐ 核心

**文件**：`model/model_minimind.py`

> 这是整个项目的地基，建议逐行阅读。按以下顺序，每个类对应一个 LLM 关键概念。

| 类 / 函数 | LLM 概念 | 要搞懂的问题 |
|---|---|---|
| `MiniMindConfig` | 超参组织 | d_model、n_layers、q_heads/kv_heads 如何决定模型 |
| `RMSNorm` | Pre-Norm 层归一化 | 为什么 LLM 用 RMSNorm 而非 LayerNorm |
| `precompute_freqs_cis` / `apply_rotary_pos_emb` | RoPE 旋转位置编码 | 相对位置信息如何编码，为何支持长度外推 |
| `repeat_kv` | GQA（分组查询注意力） | 为何 kv_heads < q_heads 能省显存 |
| `Attention` | 因果注意力 + KV Cache | `past_key_value` 与 `attention_mask` 的作用 |
| `FeedForward` | SwiGLU | 与标准 FFN 的差异 |
| `MOEFeedForward` | MoE（稀疏专家） | top-1 routing 如何计算 |
| `MiniMindBlock` | Decoder Block | Pre-Norm + 残差连接的组装方式 |
| `MiniMindModel` | 主干网络 | embedding → 多层 block → 输出 |
| `MiniMindForCausalLM.generate` | 自回归解码 | 逐 token 生成的完整循环 |

**架构要点（minimind-3 Dense）**：

- Decoder-Only，对齐 Qwen3 生态
- Pre-Norm + RMSNorm，SwiGLU 激活，RoPE + YaRN 外推
- q_heads=8、kv_heads=4（GQA），max_position_embeddings=32768，rope_theta=1e6
- d_model=768、n_layers=8，约 64M 参数

**检验标准**：能不看代码画出从 `input_ids` 到 `logits` 的完整数据流。

---

## 阶段 2：Tokenizer（1 天）

**文件**：`trainer/train_tokenizer.py`、`model/tokenizer.json`、`model/tokenizer_config.json`

- 搞懂 BPE 原理、vocab size（6400）的含义、`<|endoftext|>` 等 special tokens 的作用。
- 动手：用 tokenizer 编码一段中文，观察它如何被拆成 token id。

**检验标准**：能解释「为什么同一个词在不同上下文下可能被切成不同的 token」。

---

## 阶段 3：预训练（2 天）⭐ 核心

**文件**：`trainer/train_pretrain.py`、`dataset/lm_dataset.py`

- 理解：预训练本质是**无监督 next-token prediction**，loss 为交叉熵。
- 搞懂 `lm_dataset.py` 如何把语料切成固定长度样本、如何构造 `labels`（右移一位做 teacher forcing）。
- 重点看训练循环：混合精度、梯度累积、学习率 warmup、checkpoint 保存。

**动手**：小规模跑几步，观察 loss 下降趋势。

**检验标准**：能解释「预训练阶段模型到底在学什么」，以及 labels 与 input 的 shift 关系。

---

## 阶段 4：推理与解码（1 天）

**文件**：`model/model_minimind.py` 的 `generate` 方法、`scripts/serve_openai_api.py`

- 搞懂 temperature / top-p / top-k / repetition_penalty 各自作用。
- 理解 KV Cache 为什么能让逐 token 生成变快。

**动手**：固定 prompt，分别改 temperature、top_p、top_k，观察输出的确定性/多样性变化。

**检验标准**：能说出「贪心解码 vs 采样解码」的差异，以及 KV Cache 的加速原理。

---

## 阶段 5：SFT（监督微调）（1–2 天）

**文件**：`trainer/train_full_sft.py`

- 搞懂预训练与 SFT 的本质区别：
  - **格式**：prompt → answer 的对话结构（chat template）
  - **loss 只计算 answer 部分**，prompt 部分不参与
- 理解 chat template（如 `<|im_start|>` 等）如何把多轮对话转成训练样本。

**检验标准**：能解释「为什么预训练模型不能直接对话，需要 SFT 才能对齐指令」。

---

## 阶段 6：LoRA 与知识蒸馏（1–2 天）

**文件**：`model/model_lora.py`、`trainer/train_lora.py`、`trainer/train_distillation.py`

- **LoRA**：低秩分解原理，只训练少量额外参数、冻结主模型。
- **知识蒸馏**：教师模型（大）指导学生模型输出分布（KL 散度）。

**检验标准**：能解释 LoRA 的 `W = W0 + BA` 分解，以及蒸馏 loss 的构成。

---

## 阶段 7：强化学习对齐（3–5 天）进阶可选

**公共模块**：`trainer/rollout_engine.py`、`trainer/trainer_utils.py`（先读这两个）

**建议顺序**（由易到难）：

| 脚本 | 方法 | 核心思想 |
|---|---|---|
| `trainer/train_dpo.py` | DPO | 直接优化偏好对，无需显式 reward model |
| `trainer/train_ppo.py` | PPO | 经典 RLHF，含 critic / reward model |
| `trainer/train_grpo.py` | GRPO | 省 critic，组内相对奖励（DeepSeek-R1 所用） |
| `trainer/train_agent.py` | Agentic RL | 工具调用 + 自适应思考 |

**检验标准**：能画出 DPO、PPO、GRPO 各自的优化目标差异，并说出「为什么 GRPO 能省掉 critic」。

---

## 学习建议

1. **抓住主线（阶段 1 和 3）**：这是 LLM 区别于你已有知识的地方，其余阶段都在这两个地基上叠加。
2. **善用 README 的「模型」与「实验」章节**：里面的架构取舍讨论（如「深而窄 vs 宽而浅」的 Scaling 讨论）帮你理解「为什么这样设计」，而不只是「代码长什么样」。
3. **动手原则**：每阶段读完代码后，改一个超参（如 `n_layers`、`top_p`）观察变化，比纯读更有效。
4. **顺序别跳**：阶段 7 的 RL 依赖前面全部阶段，没有预训练 + SFT 基础直接看 PPO 会很吃力。

---

## 推荐学习节奏

| 时间 | 阶段 | 累计天数 |
|---|---|---|
| 第 1 周 | 阶段 0–3（推理 + 模型结构 + Tokenizer + 预训练） | 6–7 天 |
| 第 2 周 | 阶段 4–6（解码 + SFT + LoRA/蒸馏） | 3–5 天 |
| 第 3 周起 | 阶段 7（强化学习对齐，可选） | 3–5 天 |

---

> 下一步：从**阶段 1** 开始，逐行精读 `model/model_minimind.py`，从 `MiniMindConfig` 讲起。
