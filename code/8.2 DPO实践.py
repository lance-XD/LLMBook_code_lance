# -*- coding: utf-8 -*-
"""
8.2 DPO实践：Direct Preference Optimization（直接偏好优化）完整训练脚本

【库依赖关系】
- dataclasses.dataclass          : Python 标准库，把 Arguments 声明为数据类
- datasets.load_dataset          : HuggingFace datasets 库，从 Hub 或本地加载数据集
                                    （如 Anthropic/hh-rlhf 人类偏好数据集）
- transformers                   : HuggingFace Transformers 库 ——
                                    AutoModelForCausalLM（策略/参考模型）、AutoTokenizer（分词器）、
                                    HfArgumentParser（命令行解析）、TrainingArguments（训练参数基类）
- transformers.hf_argparser      : HfArg —— 命令行参数声明工具
- trl.DPOTrainer                 : ★ TRL（HuggingFace 的强化学习训练库）中的 DPO 训练器，
                                   内部自动完成 DPO 损失的完整计算（对数概率、隐式奖励、sigmoid 等）

【DPO 是什么：一句话】
8.1 的 RLHF 路线需要"先训练奖励模型、再用强化学习微调"两步；
DPO 跳过奖励模型和 RL 循环，直接用人类偏好数据 (prompt, chosen, rejected) 一步训练完成。

【DPO 损失公式】
    L_DPO = -log σ( β·( log(πθ(chosen)/πref(chosen)) - log(πθ(rejected)/πref(rejected)) ) )  公式8.43
  - πθ         —— 策略模型（本文件的 model，训练中更新参数）
  - πref       —— 参考模型（本文件的 model_ref，冻结不动，通常是 SFT 后的模型）
  - chosen     —— 人类标注为"更好"的回复（y_w，winner）
  - rejected   —— 人类标注为"更差"的回复（y_l，loser）
  - β(beta)    —— 温度系数：越大 = 越不允许策略模型偏离参考模型（越保守）
  - log(π/πref)—— "隐式奖励"：策略模型比参考模型更喜欢该回复的程度
  直觉：当策略模型对 chosen 的隐式奖励 > 对 rejected 的隐式奖励时，
        括号内为正、σ 接近 1、损失趋近 0（训练目标达成）。

【整体流程】
解析参数 → 加载策略模型 model + 参考模型 model_ref（冻结）→ 分词器
→ 加载并切分偏好数据（把 chosen/rejected 拆成 prompt 与回复）
→ DPOTrainer 训练 → 保存训练状态

【运行前提】pip install trl datasets（本仓库未内置）
"""
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.hf_argparser import HfArg
from trl import DPOTrainer


@dataclass
class Arguments(TrainingArguments):
    # 模型结构
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `yulan-team/YuLan-Chat-12B-v3`",
    )
    # DPO 训练数据集：HuggingFace 数据集名或本地路径（偏好数据 = 每条含 chosen/rejected 两个回复）
    data_path: str = HfArg(
        default=None,
        help="The path of preference dataset, e.g., `Anthropic/hh-rlhf`",
    )
    # 上下文窗口大小
    model_max_length: int = HfArg(default=512, help="Maximum sequence length.")
    # 使用 BF16 混合精度训练
    bf16: bool = HfArg(
        default=True,
        help="Whether to use bf16 (mixed) precision instead of 32-bit.",
    )
    # DPO 中使用的超参数 beta（对应损失公式里的 β）
    #   β 越小 → 允许策略模型更大胆地偏离参考模型（更激进地追求偏好）；
    #   β 越大 → 惩罚偏离，模型行为越接近参考模型（更保守）。
    #   直观理解：β 把"策略 vs 参考"的偏好差异放大/缩小后再过 sigmoid。
    #   ⚠ 注意：本书代码声明了该参数，但下方 DPOTrainer(**kwargs) 未显式传入 beta，
    #     trl 会使用其默认值 0.1；若想让命令行 --beta 生效，需改为
    #     DPOTrainer(..., beta=args.beta)。
    beta: float = HfArg(
        default=0.1,
        help="The beta factor in DPO loss."
        "Higher beta means less divergence from the initial policy.",
    )


# 加载训练数据集，并处理成相应的格式
def get_data(split, data_path):
    # 加载偏好数据集
    #   load_dataset(path=..., split=...)
    #     path  —— 数据集名（从 HuggingFace Hub 下载）或本地目录，如 "Anthropic/hh-rlhf"；
    #     split —— 数据划分，如 "train" / "test"（本函数由调用方传入）
    #   返回 Dataset 对象，每条样本含 "chosen"（更优回复）和 "rejected"（更差回复）两个字段。
    #   Anthropic HH-RLHF 的字段格式是完整对话文本，例如：
    #     "Human: What is the capital of France?\n\nAssistant: The capital of France is Paris."
    #   需要把"最后一段 Assistant 回复"从整段对话中拆出来，得到 (prompt, 回复) 两部分。
    dataset = load_dataset(split=split, path=data_path)

    # 对每条样本执行的处理函数（随后由 dataset.map 批量应用）
    def split_prompt_and_responses_hh(sample):
        # 多轮对话中 "Human: ...\n\nAssistant: ..." 会重复出现，
        # 我们只关心"最后一段"回复，因此要找到最后一个 "\n\nAssistant:" 的位置。
        search_term = "\n\nAssistant:"
        # str.rfind(sub) —— 从字符串右侧开始查找子串，返回最后一次出现的下标；
        #   找不到返回 -1。（对比 str.find：只找第一次出现，多轮对话会找错位置）
        search_term_idx = sample["chosen"].rfind(search_term)
        # 防御性检查：数据格式不符合预期（没有 Assistant 标记）时立刻报错，避免静默出错
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        # 切片：[: 标记末尾] 部分作为 prompt（包含最后的 "Assistant:" 前缀，
        # 这样模型生成时能顺着前缀直接开始回复）
        prompt = sample["chosen"][:search_term_idx + len(search_term)]
        return {
            "prompt": prompt,
            # chosen / rejected 用【同一个 len(prompt)】切掉前缀：
            # 前提是 HH-RLHF 里 chosen 与 rejected 共享同一段对话前缀（prompt 相同、回复不同），
            # 这正是偏好数据对的构造方式
            "chosen": sample["chosen"][len(prompt):],
            "rejected": sample["rejected"][len(prompt):],
        }

    # Dataset.map(函数) —— 对数据集每一条样本调用该函数，返回新数据集：
    #   新数据集的每条样本在原字段基础上新增 prompt / chosen / rejected 三个字段，
    #   供 DPOTrainer 使用（DPOTrainer 要求训练集包含这三列）。
    return dataset.map(split_prompt_and_responses_hh)


def train():
    # 1. 解析命令行参数（同 7.1/7.4）
    #    HfArgumentParser(Arguments).parse_args_into_dataclasses() → 元组，取 [0] 得 args
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # 2. 加载策略模型 πθ（要被训练、要更新参数的那个）
    #    AutoModelForCausalLM.from_pretrained(模型名/路径) —— 加载因果语言模型
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # 3. 加载参考模型 πref（DPO 的"锚点"）
    #    与策略模型用【同一个初始权重】加载；DPO 要求在训练过程中 πref 完全不变，
    #    用来衡量"策略模型偏离了初始模型多远"（损失公式里的分母）
    model_ref = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # 4. 冻结参考模型
    #    model_ref.eval()：切到评估模式（关闭 Dropout 等训练特有行为，保证输出确定）
    model_ref.eval()
    for param in model_ref.parameters():
        # requires_grad=False：不为其计算/保存梯度，训练中参数保持恒定；
        # 同时省下一份显存（参考模型不参与反向传播）
        param.requires_grad = False

    # 5. 加载分词器
    #    add_eos_token=True：分词时自动在末尾追加 EOS（与 7.1 的 False 相反——
    #    7.1 需要手动控制 EOS 时机做标签掩码，这里直接整段切分、交给 DPOTrainer 处理）
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=True,
    )

    # 6. 准备训练数据：加载偏好数据并切分出 prompt / chosen / rejected 三列
    train_dataset = get_data("train", args.data_path)

    # 7. 初始化 DPO 训练器并开始训练
    #    DPOTrainer(model=..., ref_model=..., args=..., tokenizer=..., train_dataset=...)
    #      model        —— 策略模型 πθ（训练中更新）
    #      ref_model    —— 参考模型 πref（冻结，见第 4 步）
    #      args         —— 训练超参数（Arguments 实例，含父类 TrainingArguments 的全部配置）
    #      tokenizer    —— 分词器
    #      train_dataset—— 偏好数据集（须含 prompt / chosen / rejected 三列）
    #    训练器内部会：对 chosen/rejected 分别做前向 → 计算 log(πθ/πref) 隐式奖励 →
    #    套用 DPO 损失公式 → 反向传播更新策略模型
    kwargs = dict(
        model=model,
        ref_model=model_ref,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    dpo_trainer = DPOTrainer(**kwargs)
    dpo_trainer.train()
    # 保存训练状态（优化器、调度器等，支持断点续训；模型权重随 checkpoint 一起保存）
    dpo_trainer.save_state()


# ============================================================================
# DPO 核心概念演示（只需 torch / Python 标准库，无需 trl、datasets 与模型下载）
# ----------------------------------------------------------------------------
# 把数据切分逻辑和 DPO 损失的数学关系用最小例子验证一遍，
# 帮助把"参数 → 行为"建立成具体认知。
# ============================================================================
def demo_dpo_concepts():
    print("=" * 60)
    print("示例 1：HH-RLHF 偏好数据如何切分（复刻 split_prompt_and_responses_hh）")
    chosen = "Human: What is the capital of France?\n\nAssistant: The capital of France is Paris."
    rejected = "Human: What is the capital of France?\n\nAssistant: It's obviously Paris."
    search_term = "\n\nAssistant:"
    idx = chosen.rfind(search_term)               # 从右往左找最后一个分隔标记
    prompt = chosen[:idx + len(search_term)]      # 含 "Assistant:" 前缀的提示部分
    print(f"  prompt   = {prompt!r}")
    print(f"  chosen   = {chosen[len(prompt):]!r}   ← 人类更喜欢的回复")
    print(f"  rejected = {rejected[len(prompt):]!r} ← 人类不喜欢的回复")

    print("=" * 60)
    print("示例 2：多轮对话时为什么必须用 rfind（从右找）")
    multi = "Human: Hi\n\nAssistant: Hello\n\nHuman: How are you?\n\nAssistant: I'm fine, thanks."
    idx2 = multi.rfind(search_term)
    print(f"  最后一次 'Assistant:' 出现在下标 {idx2}")
    print(f"  prompt = {multi[:idx2 + len(search_term)]!r}")
    print(f"  回复   = {multi[idx2 + len(search_term):]!r}")
    print(f"  （若用 find 找第一次出现的标记，会把 'Hello' 之后的内容错当成回复）")

    print("=" * 60)
    print("示例 3：DPO 损失的数值直觉（beta 的作用）")

    def dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta):
        # 隐式奖励：策略模型相对参考模型对每条回复的偏好程度（log(π/πref)）
        implicit_r_w = policy_logp_w - ref_logp_w   # chosen 的隐式奖励
        implicit_r_l = policy_logp_l - ref_logp_l   # rejected 的隐式奖励
        logits = beta * (implicit_r_w - implicit_r_l)  # 越大 = 策略越偏好 chosen
        return -F.logsigmoid(logits)                 # 与 BCE-with-logits(目标=1) 等价

    # 情形 A：策略模型已学会偏好 chosen（chosen 的对数概率 -0.1 远高于 rejected 的 -3.0）
    loss_a = dpo_loss(policy_logp_w=-0.1, policy_logp_l=-3.0,
                      ref_logp_w=-0.2, ref_logp_l=-0.2, beta=0.1)
    # 情形 B：还没学会（两者概率相当）→ 无信号，损失恒为 ln2 ≈ 0.693
    loss_b = dpo_loss(policy_logp_w=-0.5, policy_logp_l=-0.5,
                      ref_logp_w=-0.5, ref_logp_l=-0.5, beta=0.1)
    # 情形 A 但 beta=1.0：同一个偏好程度，β 放大信号 → 损失显著下降
    loss_big_beta = dpo_loss(policy_logp_w=-0.1, policy_logp_l=-3.0,
                             ref_logp_w=-0.2, ref_logp_l=-0.2, beta=1.0)
    print(f"  已学会偏好 chosen（beta=0.1）：损失 = {loss_a:.4f}  （低于 ln2≈0.693，方向正确）")
    print(f"  未学会（持平，beta=0.1）    ：损失 = {loss_b:.4f}  （= ln2，无梯度信号）")
    print(f"  已学会偏好 chosen（beta=1.0）：损失 = {loss_big_beta:.6f}（β 越大，偏好信号越强）")


if __name__ == "__main__":
    train()

    # 【可选演示】DPO 核心概念（仅需 torch，无需 trl/datasets 与模型下载）。
    # 取消下面一行的注释即可执行：
    # demo_dpo_concepts()
