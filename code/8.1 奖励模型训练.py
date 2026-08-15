# -*- coding: utf-8 -*-
"""
8.1 奖励模型训练：LlamaRewardModel —— 在 Llama 基座上加一个"奖励头"，
用对比式（pairwise）损失训练奖励模型（Reward Model, RM）

【库依赖关系】
- torch                              : PyTorch 基础库（张量运算、自动求导）
- torch.nn as nn                     : PyTorch 的神经网络子模块 ——
                                        nn.Linear（全连接层，奖励头）、nn.CrossEntropyLoss（交叉熵损失）
- torch.nn.functional as F           : PyTorch 函数式接口 ——
                                        F.binary_cross_entropy_with_logits（带 logits 的二分类交叉熵，内部自带 sigmoid）
- transformers.LlamaForCausalLM      : HuggingFace 的 Llama 因果语言模型类（基座）。
                                        本类直接继承它，复用：
                                          self.model   —— LlamaModel 主干（embedding + N 层 decoder，输出 last_hidden_state）
                                          self.lm_head —— nn.Linear(hidden_size, vocab_size)，词表投影
                                          self.config  —— 配置对象
- transformers.LlamaConfig           : Llama 的结构超参类（hidden_size、vocab_size、层数等），
                                        用于构造模型（本文件底部的演示用它创建极小型随机模型）

【奖励模型在 RLHF 中的位置（为什么需要它）】
  1. 先训练一个"评分器"：给定 (指令, 回复)，输出一个标量分数（奖励），衡量回复好坏；
  2. 再把它当作强化学习的奖励信号，去微调策略模型（后续 8.2 DPO / PPO 会用到）。
本文件实现第 1 步：把 Llama 的隐藏状态映射成标量奖励，
用"正例得分 > 负例得分"的对比损失 + 语言建模正则损失来训练。

【整体结构】
LlamaRewardModel(LlamaForCausalLM)
├── reward_head       : nn.Linear(hidden_size, 1) —— 奖励头：隐藏状态 → 奖励分
├── _forward_rmloss() : 只跑主干模型，输出 (batch, seq_len) 的逐词元奖励分
├── _forward_lmloss() : 跑主干 + lm_head，输出交叉熵损失（语言建模正则项）
└── forward()         : 计算 reward(正例) - reward(负例) 的对比损失 + LM 正则损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaForCausalLM


class LlamaRewardModel(LlamaForCausalLM):

    def __init__(self, config):
        # config: LlamaConfig 实例（含 hidden_size、vocab_size、num_hidden_layers 等结构超参）
        # 调用父类 LlamaForCausalLM.__init__，初始化并随机生成：
        #   self.model   —— LlamaModel 主干（负责把词元序列编码为隐藏状态）
        #   self.lm_head —— nn.Linear(hidden_size, vocab_size, bias=False)，把隐藏状态投影回词表
        #   self.config  —— 保存配置对象（后续代码通过 self.config.vocab_size 等读取）
        super().__init__(config)

        # 初始化线性变换层，将隐藏状态映射为标量，用于输出最终奖励
        #   nn.Linear(in_features=config.hidden_size, out_features=1, bias=False)
        #     in_features —— 输入维度：每个词元的隐藏状态大小（如 Llama-7B 的 4096）
        #     out_features—— 输出维度：1，即一个标量奖励分
        #     bias=False  —— 不使用偏置（奖励头通常不用偏置，保持简单稳定）
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)

    def _forward_rmloss(self, input_ids, attention_mask, **kargs):
        # 本函数只做"打分"：给定一个序列，输出每个词元位置的奖励分
        # input_ids      : 输入词元的符号序列，形状 (batch, seq_len)（整数张量）
        # attention_mask : 与输入对应的注意力掩码，形状 (batch, seq_len)，
        #                  1 = 有效词元，0 = padding（注意力自动忽略 0 的位置）
        # **kargs        : 吸收多余关键字参数（保持接口兼容，实际未使用）

        # 将输入词元通过大语言模型进行编码，转化为隐藏状态
        #   self.model —— 父类 LlamaForCausalLM 里的 LlamaModel 主干（不含 lm_head）
        #   forward(input_ids=..., attention_mask=..., return_dict=True, use_cache=False)
        #     return_dict=True —— 返回带属性的输出对象，可取 .last_hidden_state
        #     use_cache=False  —— 关闭 KV 缓存：训练时每个位置都要算梯度，不需要推理用的缓存加速
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False
        )
        # 使用线性变换层，将隐藏状态映射为标量
        #   output.last_hidden_state : 形状 (batch, seq_len, hidden_size)
        #   reward_head(...)         : 每个词元的隐藏状态过一层 Linear → (batch, seq_len, 1)
        #   squeeze(-1)              : 去掉最后一个长度为 1 的维度 → (batch, seq_len)
        #                              即"每个词元一个奖励分值"
        #   ⚠ 若想得到"每条样本一个标量奖励"（更常见的 RM 做法），需要改成池化：
        #       取最后有效词元（只看回复末尾）: logits[:, -1]
        #       或对序列求平均              : logits.mean(dim=1)
        logits = self.reward_head(output.last_hidden_state).squeeze(-1)
        return logits

    def _forward_lmloss(self, prompt_ids, lm_attn_mask, response_ids):
        # 本函数计算语言建模损失（LM loss），作为奖励模型训练的"正则项"，
        # 让奖励头在学"打分"的同时不破坏基座的语言能力
        # prompt_ids   : 输入词元和输出词元拼接后的符号序列（即完整序列）
        # lm_attn_mask : 对应的注意力掩码
        # response_ids : 计算交叉熵损失时目标的符号序列（与 prompt_ids 等长；
        #                其中 prompt 位置通常置为 IGNORE_INDEX(-100)，CrossEntropyLoss 会自动忽略）

        # 主干编码：与 _forward_rmloss 相同，得到完整序列的隐藏状态
        outputs = self.model.forward(
            input_ids=prompt_ids,
            attention_mask=lm_attn_mask,
            return_dict=True,
            use_cache=False,
        )
        # 使用交叉熵计算模型学习的损失，作为最终损失函数中的正则项
        hidden_states = outputs.last_hidden_state   # (batch, seq_len, hidden_size)
        logits = self.lm_head(hidden_states)        # (batch, seq_len, vocab_size) 词表投影
        loss_fct = nn.CrossEntropyLoss()            # 交叉熵损失（reduction='mean'，-100 目标自动忽略）
        logits = logits.view(-1, self.config.vocab_size)   # 展平成 (batch*seq_len, vocab_size)
        response_ids = response_ids.view(-1)               # 展平成 (batch*seq_len,)
        loss = loss_fct(logits, response_ids)              # 平均交叉熵，即语言建模损失
        return loss

    def forward(self, sent1_idx, attention_mask_1, sent2_idx, attention_mask_2, labels, prompt_ids, lm_attn_mask, response_ids, **kargs):
        # ★ 完全重写了父类的 forward：输入不再是 (input_ids, labels)，
        #   而是"正例 / 负例 / LM 序列"三组数据。因此该模型需配套自定义的
        #   DataLoader 与训练循环，不能直接用 Trainer 的标准 CausalLM 流程。
        # sent1_idx       : 输入词元和正例输出词元拼接后的符号序列
        #                   （正例 = 人工标注为"更好"的回复）
        # attention_mask_1: sent1_idx 对应的注意力掩码
        # sent2_idx       : 输入词元和负例输出词元拼接后的符号序列
        #                   （负例 = 被标注为"更差"的回复，如被人类拒绝的回复）
        # attention_mask_2: sent2_idx 对应的注意力掩码
        # labels          : 训练目标（均为 1，表示正例恒在 sent1_idx 中），
        #                   形状与 reward 输出对齐，参与对比损失计算
        # prompt_ids      : 输入词元和回复词元拼接后的完整序列（用于 LM 正则损失）
        # lm_attn_mask    : prompt_ids 对应的注意力掩码
        # response_ids    : 计算交叉熵损失时目标的符号序列
        # **kargs         : 吸收多余关键字参数

        # 计算正例输出的奖励值
        #   打分函数 _forward_rmloss → 形状 (batch, seq_len)（逐词元奖励分）
        reward0 = self._forward_rmloss(
            input_ids=sent1_idx,
            attention_mask=attention_mask_1
        )
        # 计算负例输出的奖励值
        reward1 = self._forward_rmloss(
            input_ids=sent2_idx,
            attention_mask=attention_mask_2
        )
        # 计算对比式训练方法的损失函数
        #   logits = reward0 - reward1 ：正例与负例的奖励差（越大越好）
        #   配合 labels=1 ⇒ 训练目标 = 让 reward0 - reward1 > 0，即"正例得分更高"
        #   F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        #     —— 二分类交叉熵（内部自带 sigmoid，数值比手动 sigmoid+BCE 更稳定）：
        #        logits 越大且越接近 target=1，损失越小；
        #        reduction="mean" —— 对 batch 内所有位置取平均
        logits = reward0 - reward1
        rm_loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction="mean")

        # 计算模仿学习的正则项的损失函数（语言建模损失，保持基座语言能力）
        # prompt_ids    = [101, 102, 103, 201, 202, 2]     # 完整序列 = 提示 + 回复 + EOS，喂给模型
        # response_ids  = [-100, -100, -100, 201, 202, 2]  # 前 3 个位置（提示）被换成 -100
        #               └──── 屏蔽：不学 ────┘└─ 回复：要学 ─┘
        lm_loss = self._forward_lmloss(prompt_ids, lm_attn_mask, response_ids)

        # 计算最终损失：对比损失（学打分）+ LM 正则损失（保语言能力）
        loss = rm_loss + lm_loss
        return loss


def demo_reward_model():
    """奖励模型前向 / 损失演示：构造一个极小的随机初始化 Llama，跑一遍完整 forward
    （无需下载任何模型权重，仅需 torch + transformers）"""
    # 极小的 Llama 结构配置（词表 64、隐藏 16 维、1 层 2 头），只为演示形状与损失计算
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    torch.manual_seed(0)
    model = LlamaRewardModel(config)          # 随机初始化的奖励模型
    print("reward_head 权重形状:", tuple(model.reward_head.weight.shape), "（hidden_size → 1）")

    batch, seq = 2, 8
    sent1_idx = torch.randint(0, config.vocab_size, (batch, seq))   # 正例序列
    attention_mask_1 = torch.ones(batch, seq, dtype=torch.long)
    sent2_idx = torch.randint(0, config.vocab_size, (batch, seq))   # 负例序列
    attention_mask_2 = torch.ones(batch, seq, dtype=torch.long)
    labels = torch.ones(batch, seq)                                  # 全 1：正例恒在 sent1
    prompt_ids = torch.randint(0, config.vocab_size, (batch, seq))   # LM 正则用的完整序列
    lm_attn_mask = torch.ones(batch, seq, dtype=torch.long)
    response_ids = torch.randint(0, config.vocab_size, (batch, seq))  # 目标序列（演示用随机 token）

    # 单次完整前向：返回 对比损失 + LM 正则损失 之和
    loss = model(
        sent1_idx, attention_mask_1,
        sent2_idx, attention_mask_2,
        labels, prompt_ids, lm_attn_mask, response_ids,
    )
    print("单次前向总损失:", loss.item())

    # 单独验证打分函数的输出形状（batch, seq_len）
    reward0 = model._forward_rmloss(sent1_idx, attention_mask_1)
    reward1 = model._forward_rmloss(sent2_idx, attention_mask_2)
    print("_forward_rmloss 输出形状:", tuple(reward0.shape), "（batch, seq_len）")


if __name__ == "__main__":
    demo_reward_model()
