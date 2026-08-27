# -*- coding: utf-8 -*-
"""
5.5 LLaMA：LLaMA 模型的顶层结构（LlamaModel）
教材节选的精简版，对应 HF transformers 的 modeling_llama.py 中 LlamaModel 类。

【LlamaModel 的组成（自上而下）】
  input_ids（词元 id 序列）
    ↓  embed_tokens（词嵌入矩阵）
  (batch, seq, hidden) 词向量序列
    ↓  循环 num_hidden_layers 次
  LlamaDecoderLayer（5.6：注意力 + 前馈网络，带 Pre-Norm 与残差）
    ↓
  LlamaRMSNorm（5.1：末尾归一化）
    ↓
  last_hidden_state（模型的隐藏状态输出）

【库依赖】
- torch / torch.nn —— PyTorch（nn.Embedding、nn.ModuleList）
- transformers       —— HF 库：LlamaPreTrainedModel（预训练基类）、LlamaConfig（配置类）、
                        BaseModelOutputWithPast（输出容器；⚠ transformers 5.x 中该符号
                        位于 transformers.modeling_outputs，不再从顶层导出）
- LlamaDecoderLayer / LlamaRMSNorm —— 本书 5.6 / 5.1 文件中的同名类（非 HF 顶层导出）
- 无第三方额外依赖（transformers 5.2.0 已装）

【⚠ 节选说明（教材片段，无法独立运行）】
  1. forward 引用的 self._update_causal_mask(...) 在教材精简时被省略（HF 原版中用于
     生成因果掩码，依赖 attention_mask / cache_position 等）；
  2. __init__ 末尾的 causal_mask 赋值是未使用的残留代码（HF 原版不在此处生成掩码）；
  3. 原片段用 @add_start_docstrings_to_model_forward(Llama_INPUTS_DOCSTRING) 装饰器
     （HF 旧版的文档注入装饰器），transformers 5.x 已移除这两个内部符号，此处删除。
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from transformers import LlamaPreTrainedModel, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        # config：LlamaConfig 配置对象（含 vocab_size、hidden_size、num_hidden_layers、
        #         max_position_embeddings、rms_norm_eps 等全部结构超参）
        super().__init__(config)
        self.vocab_size = config.vocab_size
        # LLaMA的词表大小
        # 词嵌入矩阵：把"词元 id"映射成"词向量"
        #   nn.Embedding(vocab_size, hidden_size, padding_idx)：
        #     vocab_size —— 词表大小（行数）
        #     hidden_size—— 词向量维度（列数）
        #     padding_idx—— padding 位置的 id（该行保持全 0、不参与梯度更新）
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # LLaMA的词嵌入矩阵，将输入的id序列转化为词向量序列
        # 所有 Transformer 解码器层
        #   nn.ModuleList：模块列表容器（便于统一注册参数、遍历）
        #   [LlamaDecoderLayer(config, layer_idx) for ...]：创建 num_hidden_layers 个
        #   解码器层（每层的结构见 5.6 LLaMALayer.py），layer_idx 传入层序号
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 所有的Transformer解码器层
        # 末尾归一化层：LLaMA 用 RMSNorm（公式见 5.1），eps 取配置里的 rms_norm_eps
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # ↓ 教材节选残留代码：生成一个 (max_len, max_len) 的全 True 掩码但未被使用
        # （HF 原版通过 _update_causal_mask 动态生成，这里不生效，可忽略）
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings), fill_value=True, dtype=torch.bool
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,   # ⚠ 教材片段漏了该参数，此处按 HF 原版补回
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 参数说明：
        #   input_ids     —— 词元 id 序列，形状 (batch, seq_len)（整数张量）
        #   attention_mask—— 注意力掩码，1=有效、0=padding
        #   position_ids  —— 每个词元的位置序号（None 时由 attention_mask 推导）
        #   inputs_embeds —— 可直接传入词向量跳过 embedding（None 时由 input_ids 查表得到）
        # 返回值：BaseModelOutputWithPast（含 last_hidden_state）或元组
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # 将输入的input id序列转化为词向量序列
            #   embed_tokens(input_ids)：查词嵌入表 → (batch, seq, hidden_size)
        # 创建单向注意力的注意力掩盖矩阵
        #   _update_causal_mask(attention_mask, inputs_embeds)：
        #     由 2D 的 attention_mask 生成 4D 的因果掩码（下三角允许、上三角 -inf），
        #     教材片段省略了该方法实现（HF 原版中有）
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)
        # 创建单向注意力的注意力掩盖矩阵

        hidden_states = inputs_embeds

        # 逐层前向：把隐藏状态依次经过每一层解码器
        for decoder_layer in self.layers:
            # 用每个LLaMA解码器层对词元的隐含状态进行映射
            #   decoder_layer(...) 返回元组，[0] 取输出隐藏状态（见 5.6）
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )[0]

        # 对每个词元的隐含状态使用RMSNorm归一化
        #   最后一层输出后再过一层 RMSNorm（对比：每层内部也有自己的归一化，见 5.6）
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


def demo_llama_flow():
    """LLaMA 顶层结构流程演示：embedding → 层循环 → 末尾归一化 的形状流转（纯 torch）"""
    torch.manual_seed(0)
    vocab, hidden, layers, batch, seq = 32, 8, 2, 2, 6

    # 用最小组件模拟结构（真实模型里"层"是 5.6 的 LlamaDecoderLayer，"归一化"是 5.1 的 LlamaRMSNorm）
    embed_tokens = nn.Embedding(vocab, hidden)
    fake_layers = [nn.Linear(hidden, hidden) for _ in range(layers)]

    def rmsnorm(x):   # 与 5.1 相同的公式
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    input_ids = torch.randint(0, vocab, (batch, seq))
    hidden_states = embed_tokens(input_ids)          # ① 词嵌入
    print(f"input_ids {tuple(input_ids.shape)} → embed_tokens → {tuple(hidden_states.shape)}")
    for layer in fake_layers:                        # ② 逐层前向（真实模型为 LlamaDecoderLayer 循环）
        hidden_states = layer(hidden_states)
    hidden_states = rmsnorm(hidden_states)           # ③ 末尾 RMSNorm
    print(f"层循环 + 末尾归一化 → {tuple(hidden_states.shape)}")
    print("结构: input_ids → embed_tokens → N×LlamaDecoderLayer → LlamaRMSNorm → last_hidden_state")


if __name__ == "__main__":
    demo_llama_flow()
