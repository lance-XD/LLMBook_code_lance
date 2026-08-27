# -*- coding: utf-8 -*-
"""
5.6 LLaMALayer：LLaMA 的单个解码器层（LlamaDecoderLayer）
教材节选的精简版，对应 HF transformers 的 modeling_llama.py 中 LlamaDecoderLayer 类。

【单个解码器层的结构（Transformer 的"标准块" + LLaMA 特色）】
  hidden_states
    ↓
  ┌─ input_layernorm（RMSNorm，Pre-Norm 前置归一化，见 5.1）
  ├─ self_attn（注意力层：Q/K/V 投影 → 缩放点积注意力 → 输出投影）
  ├─ ⊕ 残差连接（residual + attention 输出）
  ├─ post_attention_layernorm（RMSNorm）
  ├─ mlp（前馈网络：gate 投影 + up 投影 → SiLU 激活 → down 投影）
  └─ ⊕ 残差连接（residual + mlp 输出）
  输出 = 最后一层残差的结果，形状与输入相同

【为什么用 Pre-Norm（归一化在子层之前）+ 残差？】
  Pre-Norm 让梯度能沿残差通路直达浅层，显著缓解深层 Transformer 的训练不稳定；
  这是 LLaMA 相对原始 Transformer（Post-Norm）的关键改进之一。

【库依赖】
- torch / torch.nn —— PyTorch
- typing.Optional / Tuple —— 标准库类型标注
- LlamaConfig / LlamaAttention / LlamaMLP / LlamaRMSNorm —— 配置类与子模块：
    本书中分别来自配套文件（5.1 RMSNorm；注意力与 MLP 在教材完整 LLaMA 代码中）
    ⚠ 本仓库只收录了 5.1~5.6 的节选，未包含 LlamaAttention / LlamaMLP 的独立文件，
      因此本片段无法独立运行（运行需补全这两个子模块）
- 无第三方额外依赖（torch 2.4.0 已装）
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers import LlamaConfig  # 配置类（类型注解需要；运行 demo 时仅注解用）
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        # 参数说明：
        #   config    —— LlamaConfig 配置对象（hidden_size、rms_norm_eps、层数等）
        #   layer_idx —— 当前层序号（0 ~ num_hidden_layers-1，部分实现按层号微调配置）
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)  # 注意力层
        self.mlp = LlamaMLP(config)  # 前馈网络层

        # 注意力层和前馈网络层前的RMSNorm
        #   两个归一化层的公式相同（见 5.1），分别服务"注意力"和"MLP"两个子块
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 参数说明：
        #   hidden_states —— 输入隐藏状态，形状 (batch, seq_len, hidden_size)
        #   attention_mask—— 因果掩码（由上层 5.5 传入，下三角允许、上三角 -inf）
        #   position_ids  —— 位置序号（供 5.2 的 RoPE 使用）
        # 返回值：元组，[0] 为输出隐藏状态（形状与输入相同）

        # ── 子块 1：注意力（Pre-Norm + 残差） ──
        residual = hidden_states        # ① 先保存残差（在归一化【之前】拷贝 → Pre-Norm）

        hidden_states = self.input_layernorm(hidden_states)
        # 注意力层前使用RMSNorm进行归一化
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        # 进行注意力模块的计算
        #   self_attn 返回 3 个值：输出隐藏状态、注意力权重、KV 缓存（推理用）
        hidden_states = residual + hidden_states
        # 残差连接：输出 = 输入 + 注意力输出（让梯度有"捷径"回流）

        # ── 子块 2：MLP（Pre-Norm + 残差） ──
        residual = hidden_states        # ② 再保存残差

        hidden_states = self.post_attention_layernorm(hidden_states)
        # 前馈网络层前使用RMSNorm进行归一化
        hidden_states = self.mlp(hidden_states)
        # 进行前馈网络层的计算
        #   LlamaMLP 内部：hidden → gate(×SiLU) 与 up 两条支路逐元素相乘 → down 投影
        hidden_states = residual + hidden_states
        # 残差连接：输出 = 输入 + MLP 输出

        outputs = (hidden_states,)
        return outputs
        # 返回元组（与 HF 接口一致），上层 5.5 取 [0] 得到输出隐藏状态


def demo_decoder_layer():
    """DecoderLayer 结构演示：Pre-Norm + 残差 的模式与形状流转（纯 torch）
    （用最简单的线性层模拟注意力/MLP，不实现真实子模块）"""
    torch.manual_seed(0)
    hidden = 8
    # 模拟组件：真实模型里是 LlamaAttention（5.5 配套）和 LlamaMLP
    fake_attn = nn.Linear(hidden, hidden)
    fake_mlp = nn.Linear(hidden, hidden)

    def rmsnorm(x):   # 与 5.1 相同的公式 RMSNorm
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    x = torch.randn(2, 4, hidden)         # (batch, seq, hidden)

    # ── 子块 1：注意力（Pre-Norm + 残差） ──
    residual = x
    h = rmsnorm(x)                        # Pre-Norm
    h = fake_attn(h)                      # 模拟注意力
    h = residual + h                      # 残差连接 ①

    # ── 子块 2：MLP（Pre-Norm + 残差） ──
    residual = h
    h = rmsnorm(h)                        # Pre-Norm
    h = fake_mlp(h)                       # 模拟 MLP
    h = residual + h                      # 残差连接 ②

    print("输入形状:", tuple(x.shape), "→ 输出形状:", tuple(h.shape), "（形状不变）")
    print("结构: Pre-Norm → 子层 → 残差相加（注意力块 + MLP 块各一次）")


if __name__ == "__main__":
    demo_decoder_layer()
