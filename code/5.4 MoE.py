# -*- coding: utf-8 -*-
"""
5.4 MoE：Mixture of Experts（混合专家）层
Mixtral / DeepSeek 等大模型使用的稀疏激活结构。

【核心思想】
不用一个巨大的 FFN 处理所有 token，而是准备 N 个"专家"（通常是并列的小 FFN），
每个 token 通过一个路由网络（gate）打分，只激活得分最高的 k 个专家，
再把它们的输出按权重加权求和。好处：
  1. 参数量大幅增加（N 个专家），但每个 token 只算 k 个 → 计算量基本不变；
  2. 不同 token 走不同专家 → 专家可以"专业化"。

【前向流程（forward）】
  1. gate(inputs)            —— 路由网络给每个 token 打 N 个分数（gate_logits）
  2. topk(gate_logits, k)    —— 取分数最高的 k 个专家（索引 selected_experts）
  3. softmax(weights)        —— 对 k 个分数做归一化，得到加权系数
  4. 循环所有专家，把被选中的 token 送进对应专家，按权重累加输出

【库依赖】
- torch / torch.nn / torch.nn.functional —— PyTorch（torch.topk、F.softmax、torch.where）
- typing.List —— 标准库类型标注
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module,
                 num_experts_per_tok: int):
        # 参数说明：
        #   experts               —— 专家列表（每个元素是一个 nn.Module，通常是小型 FFN）
        #   gate                  —— 路由网络（一个线性层，输入 hidden_size，输出 专家数 N）
        #   num_experts_per_tok   —— 每个 token 激活的专家数量 k（稀疏度控制，k << N）
        super().__init__()
        assert len(experts) > 0   # 防御：至少 1 个专家
        self.experts = nn.ModuleList(experts)  # 所有专家的列表
        self.gate = gate          # 路由网络
        self.num_experts_per_tok = num_experts_per_tok  # 每个词元选择的专家数目

    def forward(self, inputs: torch.Tensor):
        # inputs：形状 (batch_size, seq_len, hidden_size)，即一批 token 的向量
        # 第 1 步：路由打分
        #   gate(inputs) —— 每个 token 过路由网络 → (batch, seq, num_experts)，
        #   即每个 token 对每个专家有一个"亲和度"分数
        gate_logits = self.gate(inputs)
        # 第 2 步：选 top-k 专家
        #   torch.topk(gate_logits, k, dim=-1)：
        #     沿最后一个维度取分数最高的 k 个，返回 (values, indices)：
        #       weights          —— (batch, seq, k)，选中的 k 个分数
        #       selected_experts —— (batch, seq, k)，选中的专家下标（0 ~ N-1）
        weights, selected_experts = torch.topk(gate_logits,
                                               self.num_experts_per_tok)
        # 使用路由网络选择出top-k个专家
        # 第 3 步：归一化权重
        #   F.softmax(weights, dim=1, dtype=torch.float)：
        #     对 k 个分数做 softmax → 权重和为 1（dim=1 是 seq 维？注意这里沿 dim=1
        #     归一化 —— 实际应按"每个 token 的 k 个分数"归一化，即 dim=-1；
        #     教材代码取 dim=1，在 batch>1 时行为与标准实现有差异，理解意图即可）
        #   dtype=torch.float：强制 fp32 计算 softmax，避免低精度溢出；随后转回 inputs.dtype
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        # 计算出选择的专家的权重
        # 第 4 步：按专家聚合计算
        #   results：输出累积张量，形状与 inputs 相同，初始全 0
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            # torch.where(selected_experts == i)：
            #   找出"本批中哪些 (batch, seq) 位置的 token 选中了专家 i"，
            #   返回两个下标数组：batch_idx（token 的 batch 维下标）、
            #                     nth_expert（token 在 k 个选中专家里的第几个）
            batch_idx, nth_expert = torch.where(selected_experts == i)
            # 把选中专家 i 的 token 送进 expert 计算，乘以对应权重后累加：
            #   inputs[batch_idx]               —— 这些 token 的向量
            #   weights[batch_idx, nth_expert, None] —— 它们对应的 softmax 权重
            #     （[..., None] 补一个维度用于与向量广播相乘）
            #   results[batch_idx] += ...       —— 每个 token 可能被多个专家处理，
            #                                      这里按权重把多个专家的输出累加
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        # 将每个专家的输出加权相加作为最终的输出
        # 返回：形状与 inputs 相同的 MoE 输出（batch, seq, hidden_size）
        return results


def demo_moe():
    """MoE 路由演示：4 个专家、每个 token 激活 2 个（纯 torch 可运行）"""
    torch.manual_seed(0)
    hidden = 8
    # 4 个专家：都是"hidden → hidden"的线性层（随机初始化，权重各不相同）
    experts = [nn.Linear(hidden, hidden, bias=False) for _ in range(4)]
    # 路由网络：hidden → 4 个分数（每个专家一个）
    gate = nn.Linear(hidden, 4, bias=False)
    moe = MoeLayer(experts=experts, gate=gate, num_experts_per_tok=2)

    x = torch.randn(3, hidden)          # 3 个 token
    out = moe(x)
    print("输入形状:", tuple(x.shape), "→ MoE 输出形状:", tuple(out.shape))

    # 展示路由决策过程（关闭梯度只看打分）
    with torch.no_grad():
        gate_logits = gate(x)                                   # (3, 4)
        weights, selected = torch.topk(gate_logits, 2)          # top-2
        print("\ngate 打分（每个 token 对 4 个专家的分数）:")
        print(gate_logits)
        print("\n每个 token 选中的 2 个专家:", selected.tolist())
        print("对应权重（softmax 后）:", torch.softmax(weights, dim=1).tolist())
        # 稀疏性验证：4 个专家中每个 token 只激活 2 个
        print("\n稀疏激活: 每个 token 只算 2/4 个专家（计算量约为稠密 FFN 的一半）")


if __name__ == "__main__":
    demo_moe()
