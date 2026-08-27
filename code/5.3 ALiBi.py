# -*- coding: utf-8 -*-
"""
5.3 ALiBi：Attention with Linear Biases（线性偏置注意力）
BLOOM / CodeGen 等模型使用的位置编码方案（论文：Train Short, Test Long, 2022）。

【核心思想】
不学位置 embedding，也不旋转向量（对比 5.2 RoPE），而是直接给注意力分数
加一个【随距离线性变化】的偏置：
    最终注意力分数(i, j) = q_i · k_j + 偏置(i, j)
其中偏置随"key 离 query 越远"而越小 → 模型天然偏向关注邻近 token，
且这种线性偏置结构天然支持"训练短序列、推理长序列"（长度外推）。

【每个头一个斜率（slope）】
头 h 的斜率 = 2^(-(h+1))（头数 ≤ 2 的幂时）：
    头 0 → 2^-1 = 0.5，头 1 → 0.25，头 2 → 0.125 ...（几何递减）
头数不是 2 的幂时，多余的头用"双倍基数"的奇数幂补上（见代码注释）。

【HF 实现的等价技巧（本代码即 HF build_alibi_tensor 原样实现）】
论文公式是 偏置 = slope·(j - i)（i 为 query 位置，j 为 key 位置）；
而 HF 实现只算了 slope·j（key 位置），因为 softmax 有【平移不变性】：
    softmax(l + c) = softmax(l)，c 是"该行所有元素相同的常数"。
对固定的 query 位置 i，slope·i 是常数，所以 slope·j 与 slope·(j-i) 的
softmax 结果完全一致（详见底部示例 3 的验证）。
HF 源码 docstring 原话："Alibi tensor is not causal ... relies on a translation
invariance of softmax"。

【库依赖】
- math / torch —— Python 标准库 + PyTorch（本机已装）
- 无其他外部依赖
"""
import math
import torch


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    # 参数说明：
    #   attention_mask —— 注意力掩码，形状 (batch_size, seq_len)，1=有效 token，0=padding
    #   num_heads      —— 注意力头数
    #   dtype          —— 输出张量的数据类型
    # 返回值：形状 (batch_size * num_heads, 1, seq_len) 的偏置张量
    batch_size, seq_length = attention_mask.shape

    # ---- 第 1 步：计算"小于等于 num_heads 的最大 2 的幂" ----
    #   math.log2(8) = 3.0 → floor → 2^3 = 8；log2(6) = 2.58 → floor → 2^2 = 4
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    # 基数：base = 2^(-(2^-(log2(power_of_2) - 3)))
    #   8 头时：log2(8)-3 = 0 → base = 2^-1 = 0.5
    #   16 头时：log2(16)-3 = 1 → base = 2^-0.5 ≈ 0.7071
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    # 幂次 1, 2, ..., closest_power_of_2
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    # 斜率 = base^powers：8 头时 = [0.5, 0.25, 0.125, 0.0625, ...]（几何递减序列）
    slopes = torch.pow(base, powers)
    # 计算各个头的惩罚系数

    if closest_power_of_2 != num_heads:
        # 如果头数不是2的幂次方，修改惩罚系数
        #   例：num_heads=6，已得 4 个斜率（2^-1~2^-4），还需补 2 个：
        #     额外基数用"2×closest_power_of_2=8"计算（仍是 0.5），
        #     幂次取奇数 1, 3, 5, ... → 补齐的斜率与已有序列交错（论文的插值方案）
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # ---- 第 2 步：构造"位置序号"张量（距离的载体） ----
    #   attention_mask.cumsum(-1)：逐位置累加 1 → [1, 2, 3, ...]（即"第几个有效 token"）
    #   - 1                      ：变成位置下标 [0, 1, 2, ...]
    #   × attention_mask        ：padding 位置乘 0 → 保持 0（防止给 padding 也加偏置）
    #   [:, None, :]            ：插入 head 维 → (batch, 1, seq_len)
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    # 计算相对距离
    #   slopes[..., None]：(num_heads, 1)  ×  arange_tensor：(batch, 1, seq)
    #   → (batch, num_heads, seq)，即 偏置[h, j] = slope_h × 位置 j
    alibi = slopes[..., None] * arange_tensor
    # 计算ALiBi施加的注意力偏置
    #   合并 batch 与 head 两维 → (batch*num_heads, 1, seq_len)；
    #   注意力模块里它直接加到 (batch*num_heads, q_len, k_len) 的分数上
    #   （1 与 q_len 广播，即"每个 query 行加同一份 key 位置偏置"，靠 softmax
    #    平移不变性与论文公式等价，见文件头）
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


def demo_alibi():
    """ALiBi 数值验证（纯 torch 可运行）"""
    print("=" * 60)
    print("示例 1：8 头模型的斜率序列（几何递减）")
    mask = torch.ones(1, 8, dtype=torch.long)
    alibi = build_alibi_tensor(mask, num_heads=8, dtype=torch.float32)
    print("  alibi 形状:", tuple(alibi.shape), "= (batch*heads, 1, seq)")
    slopes = alibi[:, 0, 1].tolist()          # slope = 位置 1 的偏置 - 位置 0 的偏置 = slope×1
    print("  各头斜率:", [round(s, 6) for s in slopes], "= 2^-1, 2^-2, ..., 2^-8")

    print("=" * 60)
    print("示例 2：偏置随 key 位置线性增长（Linear Biases 的由来）")
    print("  头 0（slope=0.5）的偏置序列:", alibi[0, 0].tolist())
    print("  头 7（slope=1/256）的偏置序列:", alibi[7, 0].tolist())

    print("=" * 60)
    print("示例 3：softmax 平移不变性 —— HF 为何能用 slope·j 代替 slope·(j-i)")
    torch.manual_seed(0)
    scores = torch.randn(1, 5)                     # 某 query 对所有 key 的注意力分数
    i = 2                                          # 固定 query 位置
    slope = 0.5
    paper_bias = slope * (torch.arange(5) - i)     # 论文公式 slope·(j-i)
    hf_bias = slope * torch.arange(5)              # HF 实现 slope·j
    # 两者只差常数 slope·i（对该行所有 key 相同），softmax 结果一致
    p_paper = torch.softmax(scores + paper_bias, dim=-1)
    p_hf = torch.softmax(scores + hf_bias, dim=-1)
    print("  论文公式 softmax:", [round(v, 4) for v in p_paper.tolist()[0]])
    print("  HF 实现 softmax :", [round(v, 4) for v in p_hf.tolist()[0]])
    print("  结果一致:", torch.allclose(p_paper, p_hf, atol=1e-6))

    print("=" * 60)
    print("示例 4：padding 位置偏置归零（× attention_mask 的作用）")
    mask2 = torch.tensor([[1, 1, 1, 0, 0]])        # 后两个位置是 padding
    alibi2 = build_alibi_tensor(mask2, num_heads=4, dtype=torch.float32)
    print("  带 padding 的偏置（头 0）:", alibi2[0, 0].tolist(), "← padding 位置为 0")


if __name__ == "__main__":
    demo_alibi()
