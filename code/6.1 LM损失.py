# -*- coding: utf-8 -*-
"""
6.1 LM损失：语言模型（LM）的"下一个词预测"损失 —— LlamaForCausalLM
教材节选，对应 HF modeling_llama.py 中 LlamaForCausalLM 类；在 5.5 LlamaModel
（隐藏状态编码器）之上加一个"词表映射头" lm_head，并计算交叉熵损失。

【语言模型训练的本质】
给定序列 [w1, w2, ..., wt]，训练目标 = 用前 t-1 个词预测第 t 个词（自回归）。
因此损失计算前要做【移位（shift）】：
    logits[t]  预测的是  labels[t+1]  位置的词
  → shift_logits = logits[..., :-1, :]（去掉最后一个位置的预测，它没有"下一个词"）
  → shift_labels = labels[..., 1:]    （去掉第一个位置的目标，它没有"上文"）

【前向流程】
  input_ids → LlamaModel（5.5）→ hidden_states
            → lm_head（词表映射头）→ logits（每个位置对词表中每个词的分数）
            → 与 labels 移位对齐 → CrossEntropyLoss

【库依赖】
- torch / torch.nn —— PyTorch（nn.Linear 映射头、CrossEntropyLoss 交叉熵）
- transformers       —— HF：LlamaPreTrainedModel（基类）、CausalLMOutputWithPast
                        （输出容器；⚠ transformers 5.x 中位于 transformers.modeling_outputs）
- LlamaModel         —— 本书 5.5 文件中的同名类（本仓库节选，需补全才能运行）
- 无第三方额外依赖（transformers 5.2.0 已装）
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from torch.nn import CrossEntropyLoss
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        # config：LlamaConfig 配置对象
        super().__init__(config)
        # 主干编码器：隐藏状态提取（结构见 5.5，含词嵌入 + N 层解码器 + 末尾归一化）
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        # 词表映射头：把每个位置的隐藏状态映射为"词表中每个词的分数"
        #   nn.Linear(hidden_size, vocab_size, bias=False)：
        #     in_features  —— 隐藏维度（如 4096）
        #     out_features —— 词表大小（如 32000）
        #     bias=False   —— 无偏置（LLaMA 的 lm_head 惯例）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 将最后一层输出映射为词汇表中每个词元的概率

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 参数说明：
        #   input_ids      —— 词元 id 序列，形状 (batch, seq)
        #   attention_mask —— 注意力掩码
        #   position_ids   —— 位置序号（供 RoPE）
        #   labels         —— 训练标签；None 时只推理（只返回 logits，不计算损失）
        # 返回值：CausalLMOutputWithPast（含 loss 和 logits）

        # 首先，将输入送入LlamaModel中获得最后一层的隐含状态
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        # outputs[0] = last_hidden_state，形状 (batch, seq, hidden_size)
        hidden_states = outputs[0]
        # 之后，将隐含状态送入映射头中转化为词汇表中每个词元的概率
        #   lm_head(hidden_states) → (batch, seq, vocab_size)
        #   .float()：强制转 fp32 —— 混合精度训练（bf16/fp16）下，交叉熵在 fp32
        #   计算更稳定，避免低精度溢出
        logits = self.lm_head(hidden_states).float()
        # 之后，将隐含状态送入映射头中转化为词汇表中每个词元的概率

        loss = None
        if labels is not None:
            # 基于第1至t-1个词来计算第2至t个词的预测概率
            #   ★ 移位对齐（本文件核心）：
            #   logits[..., :-1, :] —— 只取第 1 ~ seq-1 个位置的预测
            #                          （最后一个位置预测"下一个词"，但序列已结束，丢弃）
            #   labels[..., 1:]      —— 只取第 2 ~ seq 个位置的标签
            #                          （第一个位置没有"上文"，无法作为预测目标）
            #   .contiguous()：确保内存连续（view/flatten 要求），防止切片产生非连续视图报错
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 将tokens展平
            loss_fct = CrossEntropyLoss()
            # 将同批次中不同序列的词元铺平来方便计算
            #   view(-1, vocab_size)：(batch × (seq-1), vocab_size) 展平为二维
            #   view(-1)             ：(batch × (seq-1),) 展平为一维
            #   这样 CrossEntropyLoss 一次性处理"所有样本所有位置"的预测，
            #   标签为 -100 的位置会被自动忽略（预训练数据里一般没有，SFT 数据里有）
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            # 计算交叉熵损失
            #   对每个位置：-log P(真实词)，再对所有位置取平均
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )


def demo_lm_loss():
    """LM 损失核心演示：移位（shift）+ 交叉熵（纯 torch 可运行）"""
    torch.manual_seed(0)
    vocab = 10
    seq = 5

    # 一条序列：输入 5 个词，标签 = 输入本身（预训练就是"预测下一个词"）
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = input_ids.clone()
    # 模拟模型的 logits：(batch, seq, vocab) 随机分数
    logits = torch.randn(1, seq, vocab)

    # ── 核心：移位对齐 ──
    shift_logits = logits[..., :-1, :].contiguous()   # (1, 4, 10)：丢弃最后位置的预测
    shift_labels = labels[..., 1:].contiguous()       # (1, 4)：丢弃第一个位置的目标
    print(f"shift_logits 形状: {tuple(shift_logits.shape)}  = (batch, seq-1, vocab)")
    print(f"shift_labels 形状: {tuple(shift_labels.shape)}  = (batch, seq-1)")
    print(f"位置对应: logits 第 t 个位置 预测 labels 第 t+1 个位置的词")

    # ── 计算损失（与类内完全一致）──
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, vocab), shift_labels.view(-1))
    print(f"LM 损失 = {loss.item():.4f}")

    # ── 验证移位逻辑：手工对照第一对 (输入, 目标) ──
    print(f"\n输入序列: {input_ids[0].tolist()}")
    print(f"标签序列: {labels[0].tolist()}  （标签 = 输入，靠移位实现'预测下一个词'）")
    print(f"实际参与: 用位置 1~4 的输出 预测 词 2~5")


if __name__ == "__main__":
    demo_lm_loss()
