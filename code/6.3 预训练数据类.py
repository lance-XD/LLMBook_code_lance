# -*- coding: utf-8 -*-
"""
6.3 预训练数据类：PTDataset —— 把纯文本语料变成"定长 token 块"数据集

【作用】
预训练不需要"指令-回复"配对（对比 7.2 的 SFTDataset），只需要大量原始文本：
  原始文本 → 分词 → 把【所有 token 链式拼接】成一条超长流 → 切成
  block_size（= model_max_length）的等长块 → 每块就是一条训练样本。
这样做的原因：语料各行长度参差不齐，直接按行训练会造成大量 padding 浪费；
拼接切块后所有样本等长、无 padding，还天然保留了跨行的上下文。

【为什么 labels = input_ids（copy）？】
预训练目标是"预测下一个词"，标签就是输入本身；真正的"移位对齐"
（用第 t 个位置的输出预测第 t+1 个位置的词）发生在模型的损失计算里
（见 6.1 LM损失.py 的 shift_logits / shift_labels），数据侧无需处理。

【库依赖】
- torch               —— PyTorch（torch.tensor / torch.stack）
- datasets            —— HuggingFace 数据集库（load_dataset 读取纯文本文件）
                        安装：pip install datasets（本机 5.0.1 已装）
- itertools.chain     —— 标准库：把多个可迭代对象首尾相连（展平列表的列表）

【数据流（process → group_texts）】
  文本文件 → load_dataset('text') → tokenizer 批量分词（encode）
  → 收集为一维张量列表 → group_texts 链式拼接 → 按 block_size 切块 → 返回
"""
import torch
from datasets import load_dataset
from itertools import chain


class PTDataset:

    def __init__(self, args, tokenizer):
        # args：Arguments 实例（读取 model_max_length 作为 block_size）
        # tokenizer：分词器
        self.args = args
        self.block_size = self.args.model_max_length   # 每个样本的固定长度（切块大小）
        self.tokenizer = tokenizer
        # 分词并切块：process 先分词，group_texts 再拼成定长块
        self.input_ids = self.process()
        self.input_ids = self.group_texts(self.input_ids)
        # 标签 = 输入本身（预测下一个词；移位在模型损失里做，见 6.1）
        self.labels = self.input_ids.copy()

    # 数据集长度
    def __len__(self):
        # 样本数 = 切块后的块数（Trainer 用 len() 决定训练步数）
        return len(self.input_ids)

    # 获取第i条数据
    def __getitem__(self, i):
        # 返回一条样本：input_ids 与 labels 是同一个定长块（形状 [block_size]）
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    # 数据分词
    def encode(self, examples):
        # examples：一个 dict，含键 "text"（值为字符串列表，由 dataset.map 批量传入）
        # tokenizer(文本列表, truncation=True)：
        #   批量分词 → {"input_ids": [...], "attention_mask": [...]}
        #   truncation=True：超过模型最大长度时截断（预训练数据按 block_size 切块，
        #   这里超长文本先截断一次，保证后续切块稳定）
        output = self.tokenizer(examples["text"], truncation=True)
        return output

    # 数据批次化处理
    def group_texts(self, examples):
        # examples：分词结果列表，每个元素是一个一维张量（长度不等的 token 序列）
        # ① 链式拼接：把所有样本的 token 首尾相连成一条超长流
        #   chain(*examples)：把"张量列表"展开成"逐 token 迭代器"
        #   （迭代张量 = 逐个取出其中的标量元素）→ list(...) 得到一维标量张量列表
        concatenated_examples = list(chain(*examples))
        # ② 截断到 block_size 的整数倍：
        #   比如总长 10000、block_size 2048 → 10000 // 2048 = 4 → 取前 8192 个 token
        #   （尾部不足一块的 token 直接丢弃）
        total_length = len(concatenated_examples)
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # ③ 按 block_size 切块并堆叠：
        #   range(0, total_length, block_size) 依次取 [0:2048)、[2048:4096) ...
        #   torch.stack(...)：把块内的标量张量堆叠成形状 [block_size] 的一维张量
        #   结果：等长样本列表，每块就是一条训练样本（无 padding）
        result = [
            torch.stack(concatenated_examples[i:i + self.block_size]) for i in range(0, total_length, self.block_size)
        ]
        return result

    # 调用数据集加载、分词、批次化
    def process(self):
        input_ids = []
        # 加载纯文本数据集：
        #   load_dataset('text', data_files=路径)：
        #     'text' —— HF 内置的"纯文本"数据集构造器（每行一个样本）
        #     data_files —— 本地文本文件路径（可传列表）
        #     ['train'] —— 取训练划分
        list_data_dict = load_dataset('text', data_files=self.args.dataset)['train']
        # 批量应用 encode 并删除原始文本列：
        #   map(函数, batched=True) —— 一次传入一批样本（加速）；
        #   remove_columns='text'   —— 分词后丢弃文本列，只保留 input_ids
        tokenized_dataset = list_data_dict.map(
            self.encode,
            batched=True,
            remove_columns='text',
        )
        # 收集所有样本的 input_ids（转成一维张量；跳过空样本）
        for example in tokenized_dataset:
            if len(example['input_ids']) > 0:
                input_ids.append(torch.tensor(example['input_ids']))
        return input_ids


def demo_group_texts():
    """group_texts 核心逻辑演示：链式拼接 + 定长切块（纯 torch，无需数据文件）"""
    block_size = 4
    # 模拟 4 条长度不等的分词结果（一维张量）
    examples = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6, 7, 8, 9]),
        torch.tensor([10]),
    ]
    # ① 链式拼接（chain 展开张量 → 逐 token 平铺）
    concatenated = list(chain(*examples))
    print("① 链式拼接:", [int(t) for t in concatenated], "（共 10 个 token）")
    # ② 截断到 block_size=4 的整数倍 → 10 → 8
    total_length = len(concatenated)
    total_length = (total_length // block_size) * block_size
    print(f"② 截断到 {block_size} 的整数倍: {total_length} 个 token（尾部不足一块的丢弃）")
    # ③ 切块：每 4 个 token 一块
    blocks = [torch.stack(concatenated[i:i + block_size]) for i in range(0, total_length, block_size)]
    print("③ 切块结果（每条样本等长、无 padding）:")
    for b in blocks:
        print("   ", b.tolist())
    print("labels = input_ids（copy）→ 损失计算靠模型内部移位（见 6.1）")


if __name__ == "__main__":
    demo_group_texts()
