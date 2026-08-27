# -*- coding: utf-8 -*-
"""
5.1 RMSNorm：Root Mean Square Layer Normalization（均方根层归一化）
Llama 系列模型使用的归一化层，替换了 Transformer 原始论文中的 LayerNorm。

【公式】
    RMSNorm(x) = x / sqrt(mean(x²) + eps) × weight

【与 LayerNorm 的两点关键区别】
  1. 不减均值：LayerNorm 先减均值再除标准差（x - μ)/σ；
     RMSNorm 只除"均方根" RMS(x) = sqrt(mean(x²))。
     论文认为"减均值"对 Transformer 归一化并非必要，去掉后计算更省时间、效果相当。
  2. 无偏置：只有可学习的缩放系数 weight（LayerNorm 是 weight + bias 两个参数）。

【为什么先转 fp32 计算？】
混合精度训练（fp16/bf16）下，先转成 fp32 再算平方均值与开方，可避免低精度
累加误差；归一化完成后转回原 dtype 输出。

【库依赖】
- torch / torch.nn —— PyTorch（nn.Module 基类、nn.Parameter 可学习参数、torch.rsqrt 开方）
"""
import torch
import torch.nn as nn


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # weight：可学习的缩放系数（逐维），初始化为全 1。
        #   hidden_size —— 隐藏层维度，归一化沿最后一个维度进行
        #   nn.Parameter —— 把普通张量注册为"可训练参数"（参与反向传播更新）
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # eps：防止除零的小常数（当某行 x 全为 0 时，mean(x²)=0，开方后为 0，
        #      加上 eps 保证分母不为 0）
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # hidden_states：形状 (batch, seq_len, hidden_size)
        # 记录输入 dtype，归一化结束后转回
        input_dtype = hidden_states.dtype
        # 统一转 fp32 计算（见文件头说明）
        hidden_states = hidden_states.to(torch.float32)
        # 计算隐含状态的均方根
        #   pow(2)      —— 逐元素平方
        #   mean(-1, keepdim=True) —— 沿最后一个维度（hidden）求平均；
        #     keepdim=True 保留该维度 → 结果形状 (batch, seq_len, 1)，便于后续广播
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 将隐含状态除以其均方根后重新缩放
        #   torch.rsqrt(x) = 1 / sqrt(x)：即除以 sqrt(mean(x²) + eps)；
        #   等价写法：hidden_states / torch.sqrt(variance + eps)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 乘以可学习的缩放系数 weight（逐元素广播），并转回输入时的 dtype
        return self.weight * hidden_states.to(input_dtype)


def demo_rmsnorm():
    """RMSNorm 数值验证（纯 torch 可运行）"""
    torch.manual_seed(0)
    layer = LlamaRMSNorm(hidden_size=8)
    x = torch.randn(2, 5, 8)  # (batch, seq_len, hidden_size)

    # 示例 1：输出与手算公式一致（weight 初始为全 1）
    y = layer(x)
    manual = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    print("示例1  输出与手算公式一致:", torch.allclose(y, manual, atol=1e-6))

    # 示例 2：归一化后每一行的均方根 ≈ 1（weight=1 时的归一化效果）
    print("示例2  输出每行均方根 ≈", round(y.pow(2).mean(-1).mean().item(), 6))

    # 示例 3：与 LayerNorm 对比 —— RMSNorm 不减均值
    ln = nn.LayerNorm(8, elementwise_affine=False)   # LayerNorm（无仿射参数）
    print("示例3  LayerNorm 输出行均值 ≈", round(ln(x).mean(-1).mean().item(), 6), "（减了均值 → ≈0）")
    print("示例3  RMSNorm  输出行均值 ≈", round(y.mean(-1).mean().item(), 6), "（不减均值 → ≠0）")


if __name__ == "__main__":
    demo_rmsnorm()
