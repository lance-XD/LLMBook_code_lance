# -*- coding: utf-8 -*-
"""
7.4 LoRA实践：基于 PEFT 库的 LoRA 微调完整训练脚本
（在 7.1 SFT 脚本基础上加入 LoRA：LoraConfig 配置 → get_peft_model 注入旁路 → 训练 → 合并参数）

【库依赖关系】
- torch                        : PyTorch 基础库（张量运算、自动求导；DataCollator 的 pad_sequence 来自它）
- dataclasses.dataclass        : Python 标准库，把 Arguments 类声明为"数据类"（自动生成 __init__/__repr__）
- typing.Optional              : 类型标注，表示参数可为 None（LoRA 四个参数都有默认值，可不传）
- dataset.sft_dataset          : 本项目 7.2 章节自带的 SFT 数据集类（拼接指令模板、分词、构造标签）
- transformers                 : HuggingFace Transformers 库：模型(AutoModelForCausalLM)、分词器
                                (AutoTokenizer)、训练器(Trainer)、训练参数基类(TrainingArguments)、
                                命令行解析(HfArgumentParser)
- transformers.hf_argparser    : HfArg —— 命令行参数声明工具（给 dataclass 字段附加 help 说明，
                                命令行用 --参数名=值 传入）
- peft                         : ★ LoRA 的核心库（Parameter-Efficient Fine-Tuning，参数高效微调）：
    LoraConfig               —— LoRA 超参配置类（秩 r、缩放 alpha、dropout、任务类型）
    TaskType                 —— 任务类型枚举（CAUSAL_LM = 因果语言模型）
    get_peft_model           —— 把普通模型包装成带 LoRA 低秩旁路的 PEFT 模型（自动冻结原权重）
    AutoPeftModelForCausalLM —— 从 LoRA checkpoint 加载模型的自动类（能识别适配器结构）
- transformers.integrations.deepspeed : DeepSpeed ZeRO-3 辅助函数（合并参数前需解除 ZeRO-3 权重分片）

【与 7.1 的关系】
7.4 = 7.1 完整 SFT 脚本 + 4 处 LoRA 改动（文中 "..." 占位符 = 与 7.1 完全相同的部分，本文件已补全）：
  1. Arguments 增加 lora / lora_r / lora_alpha / lora_dropout 四个超参数
  2. 模型加载后包一层 get_peft_model（注入 A/B 低秩旁路并冻结原权重）
  3. 训练完成后用 merge_and_unload() 把 A·B 合并回原权重，导出为标准模型
  4. 合并前处理 DeepSpeed ZeRO-3 的分片权重

【命令行运行示例】（在项目根目录执行，需要 pip install peft）：
  python "code/7.4 LoRA实践.py" ^
      --model_name_or_path=meta-llama/Llama-2-7b-hf ^
      --dataset=./data/sft_data.json ^
      --output_dir=./output ^
      --per_device_train_batch_size=4 ^
      --num_train_epochs=3 ^
      --bf16 ^
      --lora --lora_r=8 --lora_alpha=16 --lora_dropout=0.05
"""
import os
import torch
from dataclasses import dataclass
from typing import Optional
from dataset.sft_dataset import SFTDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.hf_argparser import HfArg

# 加载PEFT模块相关接口
from peft import (
    LoraConfig,
    TaskType,
    AutoPeftModelForCausalLM,
    get_peft_model,
)
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    unset_hf_deepspeed_config,
)

# 标签中值为 -100 的位置不参与损失计算（详见 7.1 的 DataCollator 注释）
IGNORE_INDEX = -100


# ============================================================================
# 用户超参数定义
# ----------------------------------------------------------------------------
# 继承 TrainingArguments（HuggingFace Trainer 的训练参数基类）：除内置的几十个
# 训练参数（output_dir、per_device_train_batch_size、learning_rate、bf16 等，
# 命令行直接 --参数名=值 传入）外，自定义了本任务专属参数。
# HfArg(default=..., help=...) 等价于 dataclasses.field，附带命令行 --help 说明。
# ============================================================================
@dataclass
class Arguments(TrainingArguments):
    # 模型结构
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `meta-llama/Llama-2-7b-hf`",
    )
    # 训练数据集
    dataset: str = HfArg(
        default="",
        help="Setting the names of data file.",
    )
    # 上下文窗口大小
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

    # ★ LoRA 相关超参数（本文件核心，会原样传给 peft.LoraConfig）
    # lora —— LoRA 总开关：
    #   为 True ：模型加载后被 get_peft_model 包装成 LoRA 模型（冻结原权重，只训 A/B），
    #             训练结束后把 A/B 合并回原权重（见 train() 第 6 步）；
    #   为 False：等价于 7.1 的普通全参 SFT。
    lora: Optional[bool] = HfArg(default=False, help="whether to train with LoRA.")

    # lora_r —— LoRA 的秩（rank）：低秩矩阵 A、B 的维度，对应 LoraConfig.r。
    #   决定可训练参数量：新增参数 ≈ (in_features + out_features) × r × 注入层数。
    #   常用 4~64；r 越大拟合能力越强，但显存占用和过拟合风险也随之上升。
    lora_r: Optional[int] = HfArg(default=16, help='Lora attention dimension (the "rank")')

    # lora_alpha —— 缩放因子，对应 LoraConfig.lora_alpha：
    #   PEFT 实现中 LoRA 分支输出乘以 lora_alpha / lora_r（默认 16/16 = 1，保持数值量级不变）；
    #   调大 alpha 相当于放大低秩修正的幅度（详见文件底部 demo_loRA_concepts 示例 1）。
    lora_alpha: Optional[int] = HfArg(default=16, help="The alpha parameter for Lora scaling.")

    # lora_dropout —— LoRA 旁路输入的随机失活概率，对应 LoraConfig.lora_dropout，
    #   也就是 7.3 代码中的 nn.Dropout(p=config.lora_dropout)。用于对修正量做正则化，防过拟合。
    lora_dropout: Optional[float] = HfArg(default=0.05, help="The dropout probability for Lora layers.")


# ============================================================================
# 数据整理器（Data Collator）：把一批长度不等的样本 pad 成 [batch_size, max_len]
# 与 7.1 完全相同（详细示例见 7.1 文件）；train_dataset 由 SFTDataset 负责。
# ============================================================================
@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances):
        # instances: 本批样本列表，每个元素是 {"input_ids": 一维张量, "labels": 一维张量}
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 按本批最长样本右侧对齐：
        #   输入补 tokenizer.pad_token_id（模型注意力自动忽略 pad 位置），
        #   标签补 IGNORE_INDEX(-100)（该位置不参与 loss 计算）
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
        )


# ============================================================================
# 训练主流程：解析参数 → 加载分词器 → 加载模型 → ★LoRA 改造 → Trainer 训练 → ★合并参数
# ============================================================================
def train():
    # ------------------------------------------------------------------
    # 1. 解析命令行参数（与 7.1 相同）
    #    HfArgumentParser(Arguments).parse_args_into_dataclasses()
    #      —— 读取命令行（如 --model_name_or_path=... --lora --lora_r=8 --bf16），
    #         返回元组，取 [0] 得到 Arguments 实例 args，之后所有参数都从 args 读取
    # ------------------------------------------------------------------
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # ------------------------------------------------------------------
    # 2. 加载分词器（与 7.1 相同）
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=False,
    )

    # ------------------------------------------------------------------
    # 3. 加载预训练模型（与 7.1 相同）
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2")

    # ------------------------------------------------------------------
    # 4. ★ LoRA 改造：LoraConfig 配置 → get_peft_model 注入低秩旁路
    # ------------------------------------------------------------------
    if args.lora:
        # ① 构造 LoRA 配置对象（4 个参数的含义见上方 Arguments 类的详细注释）：
        #    task_type=TaskType.CAUSAL_LM —— 任务类型：因果语言建模（GPT/Llama 自回归）。
        #      TaskType 是 peft 的枚举类，PEFT 需要据此确定注入位置（Llama 注意力中的
        #      q/k/v/o 投影层）和 forward 的挂接方式；其他常见取值：SEQ_CLS、SEQ2SEQ。
        #    r=args.lora_r        —— 秩：低秩矩阵 A、B 的维度
        #    lora_alpha=...       —— 缩放因子：LoRA 分支输出乘以 alpha / r
        #    lora_dropout=...     —— 旁路输入的 dropout 概率（同 7.3 的 nn.Dropout）
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        # ② 把普通模型包装成 PeftModel：
        #    内部遍历模型所有 nn.Linear，在目标模块（默认注意力 q/k/v/o 投影层）旁挂上
        #    A、B 两个小矩阵（即 7.3 手写 LoRALinear 的旁路，peft 自动完成同样的注入），
        #    并把原模型全部参数 requires_grad=False（冻结）。
        #    返回值可直接交给 Trainer 训练；打印 model 会看到类似统计：
        #      trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.0622
        model = get_peft_model(model, peft_config)

    # ------------------------------------------------------------------
    # 5. 组装 Trainer 并训练（与 7.1 相同；model 此刻已是 PeftModel）
    # ------------------------------------------------------------------
    kwargs = dict(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=SFTDataset(args, tokenizer),
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )
    trainer = Trainer(**kwargs)
    trainer.train()
    trainer.save_model(args.output_dir + "/checkpoint-final")
    trainer.save_state()

    # ------------------------------------------------------------------
    # 6. ★ LoRA 参数合并：把学到的 A·B 加回原始权重，导出为标准模型
    #    数学本质：W' = W + (alpha / r) · B · A（详见文件底部 demo_loRA_concepts 示例 3）
    # ------------------------------------------------------------------
    if args.lora:
        # 若用 DeepSpeed ZeRO-3 训练，权重会被分片到多张卡，任何单卡只有权重的一部分，
        # 直接合并会得到不完整权重。因此先检查并解除 ZeRO-3 的分片配置：
        #   is_deepspeed_zero3_enabled() —— 判断当前是否处于 ZeRO-3 模式（返回 bool）
        #   unset_hf_deepspeed_config()  —— 清除 transformers 缓存的 DeepSpeed 配置，
        #                                    使后续 from_pretrained 按普通（非分片）方式加载完整权重
        if is_deepspeed_zero3_enabled():
            unset_hf_deepspeed_config()

        # 遍历输出目录，逐个处理每个 checkpoint 子目录（如 checkpoint-100 / checkpoint-final）
        subdir_list = os.listdir(args.output_dir)
        for subdir in subdir_list:
            if subdir.startswith("checkpoint"):
                print("Merging model in ", args.output_dir + "/" + subdir)
                # 用 LoRA 专用自动类加载 checkpoint：
                #   它会读取保存的 adapter_config.json（记录 r/alpha/target_modules 等），
                #   重建 PeftModel 结构并载入训练好的 A/B 权重；
                #   普通的 AutoModelForCausalLM 不认识适配器结构，无法直接加载。
                peft_model = AutoPeftModelForCausalLM.from_pretrained(args.output_dir + "/" + subdir)
                # 合并 A·B 进原始权重并卸载（unload）适配器结构，
                # 返回结构完全等价于原始模型的普通 PreTrainedModel：
                # 推理速度、显存占用与原始模型一致（不再需要额外的 A/B 分支）。
                merged_model = peft_model.merge_and_unload()
                # 保存合并后的标准模型（权重文件 + 分词器），可直接部署/继续推理
                save_path = args.output_dir + "/" + subdir + "-merged"
                merged_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)


# ============================================================================
# LoRA 核心概念演示（只需 torch，无需 peft 与预训练模型）
# ----------------------------------------------------------------------------
# 把 7.3 手写实现 / 本文件 peft 调用中的数学关系用纯张量运算验证一遍，
# 帮助建立"参数 → 行为"的具体认知。
# ============================================================================
def demo_loRA_concepts():
    print("=" * 60)
    print("示例 1：缩放因子 lora_alpha / lora_r 的取值")
    print("PEFT 中 LoRA 分支输出 = B(A(x)) × (alpha / r)")
    for r, alpha in [(16, 16), (4, 16), (8, 32)]:
        print(f"  r={r}, alpha={alpha} → 缩放 = {alpha / r}")

    print("=" * 60)
    print("示例 2：LoRA 新增参数量（对应 get_peft_model 注入的 A、B）")
    print("以 Llama-7B 为例：32 层 × 每层注意力 4 个投影层(q/k/v/o)，每层 4096→4096 维")
    in_f = out_f = 4096
    layers = 32 * 4
    for r in [8, 16, 64]:
        params = (in_f + out_f) * r * layers
        print(f"  r={r}: 新增参数 {params:,} ≈ {params / 1e6:.2f}M，占 7B 总参数 {params / 7e9:.3%}")

    print("=" * 60)
    print("示例 3：merge_and_unload 的数学本质  W' = W + (alpha/r)·B·A")
    torch.manual_seed(0)
    r, alpha = 4, 16
    W = torch.randn(2, 2)          # 原始权重（冻结，训练中不变）
    A = torch.randn(r, 2) * 0.02   # 训练后的 A
    B = torch.randn(2, r)          # 训练后的 B（此时已非零）
    W_merged = W + (alpha / r) * (B @ A)
    x = torch.randn(5, 2)
    # 合并前：输出 = xWᵀ + s·B(A(x))（两条分支相加，即训练时的 forward）
    out_before = x @ W.T + (alpha / r) * ((x @ A.T) @ B.T)
    # 合并后：输出 = x(W_merged)ᵀ（单层计算，即 merge 之后的推理）
    out_after = x @ W_merged.T
    print(f"  合并前(旁路相加) 与 合并后(单层) 输出一致: "
          f"{torch.allclose(out_before, out_after, atol=1e-6)}")


if __name__ == "__main__":
    train()

    # 【可选演示】LoRA 核心概念（纯 torch 即可运行，无需 peft 与模型）。
    # 取消下面一行的注释即可执行：
    # demo_loRA_concepts()
