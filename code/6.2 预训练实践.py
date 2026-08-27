# -*- coding: utf-8 -*-
"""
6.2 预训练实践：从零预训练（继续预训练）大语言模型的完整训练脚本
与 7.1 SFT 脚本结构几乎一致，核心区别在【数据侧】：
  SFT（7.1）  —— 用 SFTDataset + DataCollator（指令-回复配对，标签含 -100 掩码）
  预训练(本文件)—— 用 PTDataset（6.3：纯文本 → 分词 → 定长切块，标签 = 输入本身）

【整体流程】
解析参数 → 加载分词器 → 加载模型（FlashAttention-2）→ PTDataset 准备数据
→ Trainer 训练 → 保存模型与状态

【为什么预训练不需要 DataCollator？】
PTDataset 在 process() 里已把全部 token 切成固定长度 block_size 的等长块
（详见 6.3 的 group_texts），样本天然等长，无需再按批次 pad 对齐。

【库依赖】
- dataclasses / transformers / HfArg —— 同 7.1（transformers 5.2.0 已装）
- dataset.pt_dataset.PTDataset —— 6.3 的预训练数据类
  ⚠ 本仓库 dataset/ 目录下只有 sft_dataset.py，需按教材约定把 6.3 的类
    保存为 dataset/pt_dataset.py 才能运行本脚本
- 无其他第三方依赖
"""
from dataclasses import dataclass
from dataset.pt_dataset import PTDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
from transformers.hf_argparser import HfArg


# 用户输入超参数
# 继承 TrainingArguments（HF Trainer 的训练参数基类）：除内置的几十个训练参数外，
# 自定义了 5 个任务专属参数；命令行用 --参数名=值 传入（详见 7.1 文件的详细注释）
@dataclass
class Arguments(TrainingArguments):
    # 模型结构
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `meta-llama/Llama-2-7b-hf`",
    )
    # 训练数据集：纯文本文件路径（.txt / .jsonl 等，由 6.3 的 load_dataset('text') 读取）
    dataset: str = HfArg(
        default="",
        help="Setting the names of data file.",
    )
    # 上下文窗口大小：同时是 PTDataset 的切块长度 block_size
    model_max_length: int = HfArg(
        default=2048,
        help="The maximum sequence length",
    )
    # 只保存模型参数（不保存优化器状态等中间结果）
    save_only_model: bool = HfArg(
        default=True,
        help="When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.",
    )
    # 使用BF16混合精度训练
    bf16: bool = HfArg(
        default=True,
        help="Whether to use bf16 (mixed) precision instead of 32-bit.",
    )


def train():
    # 1. 解析命令行参数
    parser = HfArgumentParser(Arguments)
    #   HfArgumentParser(Arguments).parse_args_into_dataclasses()
    #     —— 解析命令行参数（如 --model_name_or_path=... --dataset=...），
    #        返回一个元组，取 [0] 得到我们自定义的 Arguments 实例 args，
    #        之后所有参数（含父类 TrainingArguments 的）都从 args 上读取。
    args = parser.parse_args_into_dataclasses()[0]
    # 2. 加载分词器（add_eos_token=False —— 预训练直接按原始文本切分，
    #    不需要像 SFT 那样手动控制 EOS 时机做标签掩码）
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=False,
    )
    # 3. 加载模型，并使用FlashAttention
    #    attn_implementation="flash_attention_2"：用 FlashAttention-2 算法计算注意力，
    #    降低显存占用并加速（同 7.1）
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2")
    # 4. 初始化训练器、准备训练数据并开始训练
    #    PTDataset(args, tokenizer)：
    #      —— 读纯文本 → 分词 → 把全部 token 链式拼接后切成 block_size 的等长块
    #        （详见 6.3；每个块就是一条"样本"，标签 = 输入本身）
    #    Trainer 参数（model/args/tokenizer/train_dataset）含义同 7.1；
    #    注意这里没有 data_collator —— 样本已等长（见文件头说明）
    kwargs = dict(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=PTDataset(args, tokenizer),
    )

    trainer = Trainer(**kwargs)
    trainer.train()
    trainer.save_model(args.output_dir + "/checkpoint-final")
    trainer.save_state()


if __name__ == "__main__":
    train()
