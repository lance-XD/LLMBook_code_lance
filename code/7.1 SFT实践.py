import torch
from dataclasses import dataclass
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

IGNORE_INDEX = -100


# ============================================================================
# 7.1 监督微调（SFT）完整训练脚本
# ----------------------------------------------------------------------------
# 整体流程（一句话）：读取训练数据 → SFTDataset 拼模板并分词构造标签 →
#                     DataCollator 按批次对齐 → Trainer 训练模型 → 保存 checkpoint。
#
# ▍一条数据的完整流转示例：
#   1. 数据文件 sft_data.json 中一条样本：
#        {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."}
#   2. SFTDataset.process() 把它拼成模板文本并分词：
#        s = "Below is an instruction ...\n### Instruction:\nWhat is the capital of France?\n### Output:\n"
#        t = "The capital of France is Paris."
#        → encode_src_tgt(s, t, tokenizer) 返回：
#            input_id = [100, 200, ..., EOS]            （提示 + 回复整体 token）
#            label    = [-100, -100, ..., 31, ..., EOS] （提示部分为 -100，回复部分保留）
#   3. DataCollatorForSupervisedDataset 把一批长度不等的样本 pad 成
#      [batch_size, max_len] 的矩形张量交给模型。
#   4. Trainer 用交叉熵损失训练，标签为 -100 的位置不参与 loss。
#
# ▍命令行运行示例（在项目根目录执行）：
#   python "code/7.1 SFT实践.py" ^
#       --model_name_or_path=meta-llama/Llama-2-7b-hf ^
#       --dataset=./data/sft_data.json ^
#       --output_dir=./output ^
#       --per_device_train_batch_size=4 ^
#       --num_train_epochs=3 ^
#       --bf16
# ============================================================================


# ============================================================================
# 用户超参数定义
# ----------------------------------------------------------------------------
# 通过继承 TrainingArguments（HuggingFace Trainer 的训练参数基类），在原有大量
# 内置参数（学习率、训练轮数、batch size、日志/保存频率、混合精度等）基础上，
# 自定义了本任务专属的 5 个参数。
# 每个字段用 HfArg 声明（等价于 dataclasses.field，额外携带 help 说明文本），
# 这样命令行执行时可用 --参数名=值 的方式传入，--help 可查看全部参数说明。
#
# ▍本类定义的参数一览：
#   model_name_or_path —— 预训练模型：填 HuggingFace 模型名（自动从 Hub 下载）
#                         或本地模型目录路径，例如 meta-llama/Llama-2-7b-hf
#   dataset            —— 训练数据文件路径（JSON 文件），SFTDataset 会读取它
#   model_max_length   —— 上下文窗口大小/最大序列长度：输入 + 输出统一截断/对齐到
#                         这个长度（作为 tokenizer.model_max_length 使用）
#   save_only_model    —— 保存 checkpoint 时只保存模型权重，不保存优化器、调度器、
#                         RNG 等状态（更省磁盘空间）
#   bf16               —— 是否使用 BF16 混合精度训练：以半精度（16 位）存储和计算
#                         中间张量，显著降低显存占用、加快训练；BF16 相比 FP16 更
#                         不易溢出，A100/H100 等新架构显卡支持
#
# ▍命令行传参示例：
#   --model_name_or_path=meta-llama/Llama-2-7b-hf --dataset=./data/sft_data.json
#   --output_dir=./output --bf16 --per_device_train_batch_size=4 --num_train_epochs=3
#   （其中 output_dir / per_device_train_batch_size / num_train_epochs 等来自父类
#    TrainingArguments，无需在本类重复定义）
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


# ============================================================================
# 数据整理器（Data Collator）：把一批样本"拼"成一个 batch 张量
# ----------------------------------------------------------------------------
# 为什么需要它：SFTDataset 里每条样本的 input_id / label 长度各不相同（不同指令
# 和回复的长短不同），无法直接堆成矩形张量喂给模型。Trainer 在每次取批次时都会
# 调用 collator 的 __call__，把本批样本按"最长的一条"对齐（右侧补位）。
#
# 参数说明：
#   tokenizer —— 分词器对象，__call__ 中用它取 pad_token_id（填充标记的 id）
#   instances —— 一个列表，每个元素是 SFTDataset.__getitem__ 返回的 dict：
#                {"input_ids": 一维张量, "labels": 一维张量}
#
# 返回值：dict(input_ids=..., labels=...)，两个张量形状均为 [batch_size, max_len]
#
#
# ▍具体示例：
#   假设本批有 2 条样本：
#     样本 A：input_ids=[1,2,3]         labels=[-100,-100,5]
#     样本 B：input_ids=[4,5,6,7,8]     labels=[-100,9,10,11,12]
#   按最长（5）对齐后得到：
#     input_ids = [[1, 2, 3, PAD, PAD],
#                  [4, 5, 6, 7, 8]]
#     labels    = [[-100, -100, 5, -100, -100],
#                  [-100, 9, 10, 11, 12]]
#   其中 PAD = tokenizer.pad_token_id（如 0）；label 里补的 -100 不会参与 loss。
# ============================================================================
# 批次化数据，并构建序列到序列损失
@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances):
        # 把本批所有样本按 "input_ids" / "labels" 两个键分别抽出来组成两个列表：
        #   input_ids = [样本A的input_ids张量, 样本B的input_ids张量, ...]
        #   labels    = [样本A的labels张量,   样本B的labels张量,    ...]
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 输入按最长对齐：右侧补 tokenizer.pad_token_id
        #   torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=...)
        #     —— 把一组长度不等的张量填充成等长（取本批最长者）后堆叠；
        #        batch_first=True → 返回形状为 [batch, max_len]（batch 在最前）；
        #        padding_value     → 填充时使用的数值：
        #                            输入用 pad_token_id（模型注意力会自行忽略 pad 位置），
        #                            标签用 IGNORE_INDEX(-100)（不参与 loss 计算）。
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # 标签按最长对齐：右侧补 IGNORE_INDEX(-100)，保证补出来的位置不参与 loss
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
        )


# ============================================================================
# 训练主流程：解析参数 → 加载分词器 → 加载模型 → 组装 Trainer → 训练并保存
# ----------------------------------------------------------------------------

def train():
    # 解析命令行参数
    #   HfArgumentParser(Arguments).parse_args_into_dataclasses()
    #     —— 解析命令行参数（如 --model_name_or_path=... --dataset=...），
    #        返回一个元组，取 [0] 得到我们自定义的 Arguments 实例 args，
    #        之后所有参数（含父类 TrainingArguments 的）都从 args 上读取。
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    # 加载分词器
    #   AutoTokenizer.from_pretrained(模型, model_max_length=..., padding_side=..., add_eos_token=...)
    #     —— 从本地路径或 HuggingFace Hub 加载分词器：
    #        model_max_length —— 设定模型最大长度（与 Arguments.model_max_length 一致）
    #        padding_side     —— 填充方向："right" = 在序列右侧补 pad token
    #        add_eos_token    —— False = 分词时不自动追加 EOS；
    #                            这样 EOS 的添加时机完全交给 SFTDataset 控制
    #                            （详见 sft_dataset.py 的 encode_src_tgt 第 2 步）
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=False,
    )
    # 加载模型，并使用FlashAttention
    #   AutoModelForCausalLM.from_pretrained(模型, attn_implementation="flash_attention_2")
    #     —— 加载因果语言模型（GPT/Llama 这类自回归模型）：
    #        attn_implementation —— 注意力实现方式；flash_attention_2 用 FlashAttention-2
    #                               算法计算注意力，大幅降低显存占用并加速训练
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2")

    kwargs = dict(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=SFTDataset(args, tokenizer),
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )
    # 初始化训练器、准备训练数据并开始训练
    #   Trainer(model=..., args=..., tokenizer=..., train_dataset=..., data_collator=...)
    #     —— HuggingFace 训练器，内部完成 batch 采样、前向/反向、梯度更新、日志等：
    #        model          —— 要训练的模型
    #        args           —— 训练超参数（Arguments 实例，含父类全部训练配置）
    #        tokenizer      —— 分词器
    #        train_dataset  —— 训练数据集（SFTDataset 实例，内部已完成分词与标签构造）
    #        data_collator  —— 批次整理器（把不等长样本 pad 成 batch，见上方类注释）
    trainer = Trainer(**kwargs)
    #     trainer.train()               —— 启动训练（按 args 里的 epoch/batch/学习率配置）
    trainer.train()
    #     trainer.save_model(路径)      —— 保存模型权重到 output_dir/checkpoint-final
    trainer.save_model(args.output_dir + "/checkpoint-final")
    #     trainer.save_state()          —— 保存训练状态（优化器、调度器、RNG 等，配合
    #                                      resume_from_checkpoint 可断点续训）
    trainer.save_state()

# 测试传参写法
# class Test_Trainer:
#     def __init__(self, model, args):
#         self.model = model
#         self.args = args
#     def test(self):
#         print(f"model: {self.model}, args: {self.args}")
if __name__ == "__main__":
    train()
    # test_dict = dict(model="Qwen", args=123)
    # trainer = Test_Trainer(**test_dict)
    # print(trainer.test())
