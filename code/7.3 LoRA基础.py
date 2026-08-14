# -*- coding: utf-8 -*-
"""
7.3 LoRA基础：LoRA（Low-Rank Adaptation，低秩适配）原理的最小实现

【库依赖关系】
- import torch                    : PyTorch 基础张量库（张量运算、自动求导）
- import torch.nn as nn           : PyTorch 的神经网络子模块，提供 nn.Linear（全连接层）、
                                    nn.Dropout（随机失活层）、nn.Module（所有网络层的基类）
- import torch.nn.functional as F : PyTorch 函数式接口，F.linear 与 nn.Linear 是同一个计算 y = xWᵀ + b

【核心思想】
在冻结的原始权重 W 旁边，并联一条"降维(A) → 升维(B)"的低秩旁路：
    输出 = xWᵀ + b + B(A(x))
微调时只训练 A、B（参数量通常不到原来的 1%），就能近似表达针对下游任务的权重修正量。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 模拟 7.4 实践代码中的超参对象（等价于 peft.LoraConfig(r=4, lora_dropout=0.1)）
class Config:
    """LoRA 超参配置"""

    lora_r = 8          # LoRA 的秩（rank）：低秩空间大小，通常 4~64，远小于输入/输出维度
    lora_dropout = 0.1  # LoRA 旁路输入的 Dropout 概率，用于正则化


# 继承 PyTorch 的线性变换类 nn.Linear
class LoRALinear(nn.Linear):

    def __init__(self, in_features, out_features, config, bias=True):
        # in_features : 输入特征维度（上一层的输出大小）
        # out_features: 输出特征维度（本层的输出大小）
        # in_features和out_features只负责改变输入数据的最后一维（特征维度），而其他维度（比如序列长度、批量大小）都会原封不动地保留
        # [batch, seq, in] → [batch, seq, out]
        # bias=True   : 是否创建偏置参数 b（形状 (out_features,)）
        # 调用父类 nn.Linear.__init__，内部会创建并随机初始化
        #   self.weight —— 形状 (out_features, in_features)，即原始的 W（冻结不训练）
        #   self.bias   —— 形状 (out_features,)（若 bias=True）
        super().__init__(in_features, out_features, bias=bias)

        # 从配置中获取LoRA的秩，这决定了低秩矩阵A和B的大小
        self.r = config.lora_r

        # 初始化A，将输入映射到低秩空间r
        # 形状：(r, in_features)，参数量 = in_features × r
        # bias=False：低秩矩阵不带偏置（LoRA 标准做法，少一半参数量且不影响效果）
        self.A = nn.Linear(in_features, self.r, bias=False)

        # 初始化B，将低秩空间映射回原始输出空间
        # 形状：(out_features, r)，参数量 = r × out_features
        self.B = nn.Linear(self.r, out_features, bias=False)

        # 初始化一个Dropout层，用于在输入传递给A之前进行正则化
        # p：随机失活概率，训练时以概率 p 把元素置 0，其余元素放大 1/(1-p) 倍
        self.dropout = nn.Dropout(p=config.lora_dropout)

        # 使用标准差为0.02的正态分布初始化A的权重
        # normal_ 是原地操作（_ 后缀，直接改内存）：每个元素从 N(0, 0.02²) 独立采样
        # 与 LLaMA 等模型的初始化量级一致，保证 LoRA 旁路的初始数值不会破坏原始输出
        self.A.weight.data.normal_(std=0.02)

        # B的权重初始化为零（LoRA 的关键技巧）：
        # 训练第 0 步时 B(A(x)) == 0，模型输出 = 原始模型输出，不破坏预训练权重；
        # 梯度仍能正常流经 A、B，训练过程中旁路从 0 开始缓慢"长出"需要的修正量
        self.B.weight.data.zero_()

    def forward(self, input):
        # input: 形状 (batch_size, in_features) 的输入张量
        # 原始分支：F.linear(input, self.weight, self.bias) = input @ Wᵀ + b
        #   self.weight / self.bias 是父类 nn.Linear 留下的"原始预训练权重"
        linear_output = F.linear(input, self.weight, self.bias)

        # LoRA分支：input → Dropout → A(降维到 r) → B(升维回 out_features)
        #   即 input @ Aᵀ @ Bᵀ，A、B 才是训练时真正更新的参数
        lora_output = self.B(self.A(self.dropout(input)))

        # 将标准线性输出与缩放后的LoRA输出相加，得到最终输出
        # 注：PEFT 库（见 7.4 的 LoraConfig）会再乘缩放因子 lora_alpha / r，
        #     此处为基础版本，省略缩放以讲清原理
        return linear_output + lora_output


if __name__ == "__main__":
    # ========== 示例 1：nn.Linear 与 F.linear 的等价关系 ==========
    torch.manual_seed(0)
    x = torch.randn(2, 3)                     # 2 个样本，每个 3 维特征
    layer = nn.Linear(3, 5, bias=True)        # 输入 3 维 → 输出 5 维
    y = layer(x)                              # 形状 (2, 5)
    y_manual = x @ layer.weight.T + layer.bias  # 数学定义：y = xWᵀ + b
    assert torch.allclose(y, y_manual)
    print("示例1：nn.Linear 与 xWᵀ+b 等价, weight 形状:", tuple(layer.weight.shape))

    # ========== 示例 2：LoRA 的参数量节省（100 维 → 100 维，r=4） ==========
    base = nn.Linear(100, 100)
    lora_layer = LoRALinear(100, 100, Config())
    w_params = base.weight.numel()                        # 100×100 = 10000
    ab_params = (sum(p.numel() for p in lora_layer.A.parameters())
                 + sum(p.numel() for p in lora_layer.B.parameters()))  # 100×4 + 4×100 = 800
    print(f"示例2：原始权重 {w_params} 个参数, LoRA 旁路 {ab_params} 个参数, "
          f"节省 {1 - ab_params / w_params:.1%}")

    # ========== 示例 3：B 零初始化 ⇒ 训练起点与原始模型完全一致 ==========
    x2 = torch.randn(4, 100)
    with torch.no_grad():
        lora_layer.weight.copy_(base.weight)  # 让两条分支的原始权重一致
        lora_layer.bias.copy_(base.bias)
    print("示例3：训练开始时两条分支输出一致:", torch.allclose(base(x2), lora_layer(x2)))

    # ========== 示例 4：冻结原始权重，只训练 A、B（这就是 LoRA 微调的全部） ==========
    for p in lora_layer.parameters():
        p.requires_grad = False
    lora_layer.A.weight.requires_grad = True
    lora_layer.B.weight.requires_grad = True
    trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    print(f"示例4：冻结后仅 {trainable} 个参数可训练（只含 A、B）")

    # ========== 示例 5：nn.Dropout 的行为（训练模式 vs 推理模式） ==========
    d = nn.Dropout(p=0.5)
    ones = torch.ones(1, 6)
    d.train()
    print("示例5：训练模式 dropout(ones) =", d(ones).tolist())
    d.eval()
    print("示例5：推理模式 dropout(ones) =", d(ones).tolist())
