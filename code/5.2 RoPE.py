# -*- coding: utf-8 -*-
"""
5.2 RoPE：Rotary Position Embedding（旋转位置编码）
Llama 等模型使用的位置编码方式，替代 Transformer 原始的绝对位置 embedding。

【核心思想】
不把"位置"作为向量加进输入，而是把 q/k 向量在二维子空间里【旋转】：
  q 在位置 m 处旋转角度 θ_m，k 在位置 n 处旋转角度 θ_n。
旋转后的内积 q_m · k_n 只与【相对距离 (m - n)】有关，与绝对位置无关
 → 模型天然感知"相对位置"，且不增加任何可学习参数。

【公式】
每个二维子空间 (x1, x2) 旋转角度 θ（标准旋转矩阵）：
    x1' = x1·cosθ - x2·sinθ
    x2' = x1·sinθ + x2·cosθ
代码用等价形式实现：xT·cosθ + rotate_half(x)·sinθ,其中 rotate_half(x) = (-x2, x1)T
恰好实现"逆时针旋转 90°"。

【本文件 vs 完整实现】
本文件只包含"旋转应用"部分（rotate_half + apply_rotary_pos_emb）；
cos/sin 的预计算（θ_i = m / 10000^(2i/d)，i 为子空间下标）通常在调用方完成
（见 5.5 LLaMA.py 或 HF 的 modeling_llama.py）。

【库依赖】
- torch —— PyTorch（torch.cat / 张量切片；本机 2.4.0 已装）
- 无其他外部依赖
"""
import torch


def rotate_half(x):
    # 将向量每两个元素视为一个子空间
    #   x 的最后一维按 2 个元素为一组；把"后半组"取负后与"前半组"交换位置，
    #   实现 90° 旋转。例：x = [x1, x2, x3, x4]（两组子空间）
    #     x1 = [x1, x2]，x2 = [x3, x4] → 返回 [-x3, -x4, x1, x2]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # 参数说明：
    #   q / k        —— query/key 张量，形状 (batch, num_heads, seq_len, head_dim)
    #   cos / sin    —— 预计算的旋转角余弦/正弦表，形状 (max_seq_len, head_dim)：
    #                    HF 实现：cos[m, i] = cos(m·θ_{i mod (head_dim/2)})，
    #                    θ_i = 1/10000^(2i/head_dim)；
    #                    即角度表按 [θ0, θ1, ..., θ0, θ1, ...] 重复（cat 方式），
    #                    与 rotate_half"前一半 vs 后一半"的配对 (i, i+d/2) 严格对应
    #   position_ids —— 每个 token 的位置序号，形状 (batch, seq_len)
    # 返回值：旋转后的 (q_embed, k_embed)，形状不变
    # 获得各个子空间旋转的正余弦值
    #   cos[position_ids]：按位置序号"收集"对应的 cos 值 → (batch, seq_len, head_dim)
    #   unsqueeze(1)      ：插入 head 维度 → (batch, 1, seq_len, head_dim)，
    #                       与 q/k 的 (batch, num_heads, seq_len, head_dim) 广播对齐
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    # 将每个子空间按照特定角度进行旋转
    #   q·cos + rotate_half(q)·sin —— 旋转公式的向量化写法：
    #     设 q 中一个二维子空间为 (a, b)，则
    #       rotate_half → (-b, a)
    #       q·cos + rotate_half(q)·sin → (a·cos - b·sin, b·cos + a·sin) ✓ 即旋转矩阵
    #   k 同样旋转（v 不旋转 —— 位置编码只作用于 q/k，影响的是注意力分数）
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def demo_rope():
    """RoPE 核心性质演示（纯 torch 可运行）"""
    print("=" * 60)
    print("示例 1：rotate_half 的行为（90° 旋转）")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])          # 两组二维子空间 (1,2) 和 (3,4)
    print("  rotate_half([1,2,3,4]) =", rotate_half(x).tolist())   # [-3, -4, 1, 2]

    print("=" * 60)
    print("示例 2：旋转公式与标准旋转矩阵等价")
    v = torch.tensor([1.0, 2.0])                     # 一个二维子空间 (x1, x2)
    theta = 0.5
    cos_t, sin_t = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    rot_matrix = torch.tensor([[cos_t.item(), -sin_t.item()], [sin_t.item(), cos_t.item()]])
    rotated_formula = v * cos_t + rotate_half(v) * sin_t     # 代码里的公式
    rotated_matrix = rot_matrix @ v                          # 标准旋转矩阵
    print("  公式旋转 :", rotated_formula.tolist())
    print("  矩阵旋转 :", rotated_matrix.tolist())
    print("  两者一致 :", torch.allclose(rotated_formula, rotated_matrix))

    print("=" * 60)
    print("示例 3：核心性质 —— 旋转后内积只与相对距离 (m-n) 有关")
    # 预计算 cos/sin（与 HF LLaMA 完全一致）：
    #   θ_i = 1 / 10000^(2i/head_dim)，freqs[m, i] = m·θ_i
    #   ⚠ 关键配对细节：rotate_half 的分组是"前一半 vs 后一半"（配对 (0,2)、(1,3)），
    #     所以角度表必须用 torch.cat((freqs, freqs), dim=-1) 重复为 [θ0, θ1, θ0, θ1]，
    #     而不是 repeat_interleave 的 [θ0, θ0, θ1, θ1]（那样配对 (0,1)、(2,3)，旋转不成立）
    head_dim, max_pos = 4, 8
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (2,)
    freqs = torch.outer(torch.arange(max_pos).float(), inv_freq)                    # (8, 2)
    emb = torch.cat((freqs, freqs), dim=-1)   # (8, 4) = [mθ0, mθ1, mθ0, mθ1]
    cos = torch.cos(emb)
    sin = torch.sin(emb)

    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, head_dim)   # 单个 query 向量
    k = torch.randn(1, 1, 1, head_dim)   # 单个 key 向量
    dots = []
    for m, n in [(3, 1), (7, 5)]:        # 两组绝对位置，但相对距离都是 m-n=2
        q_rot, _ = apply_rotary_pos_emb(q, k, cos, sin, torch.tensor([[m]]))
        _, k_rot = apply_rotary_pos_emb(q, k, cos, sin, torch.tensor([[n]]))
        dots.append((q_rot @ k_rot.transpose(-1, -2)).item())
    print(f"  (m=3,n=1) 与 (m=7,n=5) 的内积: {dots[0]:.6f} vs {dots[1]:.6f}")
    print(f"  相对距离相同 → 内积相同: {abs(dots[0] - dots[1]) < 1e-5}")


if __name__ == "__main__":
    demo_rope()
