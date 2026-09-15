# MiniMind

- 目标：完全从零开始，用约 3 块钱成本 + 2 小时训练，训练出一个约 64M 参数的迷你语言模型 MiniMind，体积约为 GPT-3 的 1/2700，让普通个人 GPU 也能跑通训练与复现。
- 全阶段覆盖：代码用 PyTorch 原生实现（不依赖 transformers/trl/peft 等高层封装），涵盖 MoE、数据清洗、预训练（Pretrain）、监督微调（SFT）、LoRA、RLHF（DPO）、RLAIF（PPO/GRPO/CISPO）、Tool Use、Agentic RL、模型蒸馏等完整训练链路。
- 扩展模型：另有多模态版本 MiniMind-V（视觉）、MiniMind-O（Omni）、MiniMind-dLM（扩散语言模型）、MiniMind-Linear 等。
- 定位：既是 LLM 全阶段的开源复现项目，也是一套面向 LLM 入门与实践的教程，强调"从 0 开始训练、理解每一行代码"，而非只做推理或微调。

## 阶段 0：跑通一次推理（1 天）

**目标**：先看效果、建立全局观，不要一上来就啃代码。

- 按 README「快速开始」章节下载 `minimind-3` 权重，跑 CLI 推理（`scripts/chat_api.py`）。
- 动手：换成自己的 prompt，观察输出，体会「续写」与「问答」的区别。

**检验标准**：能成功跑出模型输出，并能用一句话说出「大模型本质上在做什么」。



