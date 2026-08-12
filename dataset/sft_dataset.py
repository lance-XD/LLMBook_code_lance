import json


class SFTDataset:
    IGNORE_INDEX = -100
    # 定义指令模板格式
    instruction_template = "\n### Instruction:\n"
    response_template = "\n### Output:\n"
    format_template = {
        "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. " +
                "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
                "{input}" + response_template
        ),
        "prompt_no_input": (
                "Below is an instruction that describes a task. " +
                "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
                response_template
        ),
    }

    def __init__(self, args, tokenizer):
        self.args = args
        self.block_size = self.args.model_max_length
        self.tokenizer = tokenizer
        self.input_ids, self.labels = self.process(self.tokenizer)

    # 数据集长度
    def __len__(self):
        return len(self.input_ids)

    # 获取第 i 条数据
    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    # ============================================================================
    # 对输入和输出进行分词并标记输出位置
    # ----------------------------------------------------------------------------
    # 一句话作用：把"提示 s + 答案 t"编码成模型输入 input_id，同时构造训练标签
    #             label，使模型在训练时【只对答案部分计算损失】、忽略提示部分。
    #
    # 参数说明：
    #   s —— 提示部分文本（Prompt），即拼好指令模板后的完整指令字符串，例如：
    #        "Below is an instruction ...\n### Instruction:\nWhat is the capital of France?\n### Output:\n"
    #   t —— 期望模型输出的标准答案（回复），例如 "The capital of France is Paris."
    #   tokenizer —— 分词器对象（HuggingFace transformers 的 PreTrainedTokenizer），
    #                负责把文本切分成 token，并把每个 token 映射成数字 id
    #
    # 返回值：(input_id, label)
    #   input_id —— 整条样本（s + t）分词后的 token id 张量，形状为 [seq_len]，作为模型输入
    #   label    —— 训练标签，同样是 [seq_len] 张量；提示部分全部被替换为
    #               IGNORE_INDEX(-100)，只有回复部分保留真实的 token id
    # ----------------------------------------------------------------------------
    # ▍tokenizer.encode() 的参数详解（本函数中用到 3 个）：
    #   text                —— 要分词的字符串（encode 的第一个位置参数，必填）
    #   max_length          —— 允许的最大 token 数；此处取 tokenizer.model_max_length，
    #                          即模型/分词器支持的最大序列长度（常见 512 / 1024 等）
    #   truncation=True     —— "截断"开关：
    #                          当文本分词后的 token 数超过 max_length 时，从序列尾部
    #                          丢弃多余的 token，强制把长度压到 max_length 以内；
    #                          设为 False 则超长文本不截断，超出部分可能导致后续
    #                          训练时报长度不匹配错误。
    #   return_tensors='pt' —— 指定返回值类型为 PyTorch 张量（pt = PyTorch）；
    #                          不指定时返回普通 Python 列表（list[int]）。
    #                          带批次维度的张量形状为 [1, seq_len]，所以取 [0] 去掉批次维。
    # ----------------------------------------------------------------------------
    # ▍完整示例（以下 token id 均为示意，真实数值由具体分词器决定）：
    #   假设样本：instruction = "What is the capital of France?"
    #            output      = "The capital of France is Paris."
    #   s = "Below is an instruction ...\n### Instruction:\nWhat is the capital of France?\n### Output:\n"
    #   t = "The capital of France is Paris."
    #
    #   第 1 步  对 s 单独分词（不追加 EOS）→ 假设 30 个 token：
    #     source_id = [100, 200, 300, ..., 900]        # 共 30 个，长度 L_s = 30
    #
    #   第 2 步  打开 add_eos_token 后对 s + t 整体分词 → 末尾自动追加 EOS，共 38 个 token：
    #     input_id  = [100, 200, 300, ..., 900, 31, 42, 53, 64, 75, 86, 97, EOS]
    #                 └─────── 前 30 个 = 提示部分 ────────┘└── 回复 7 个 ──┘└─┘
    #                 （注：s + t 的前 30 个 token 与 source_id 完全一致）
    #
    #   第 3 步  clone 出 label，把前 30 个位置全部改成 -100：
    #     label     = [-100, -100, -100, ..., -100, 31, 42, 53, 64, 75, 86, 97, EOS]
    #                 └────── 提示部分被屏蔽（不参与 loss）───────┘└── 回复保留 ──┘
    #   → 训练时模型读入整条 input_id，但只在"回复部分 + EOS"上计算交叉熵损失。
    # ============================================================================
    def encode_src_tgt(self, s, t, tokenizer):
        # ── 第 1 步：只对提示部分 s 分词 ──────────────────────────────────────────
        # 三个参数的含义：
        #   s            —— 待分词文本（提示部分）
        #   max_length   —— 长度上限，超过即截断
        #   truncation   —— True = 超长时从尾部截断
        # 不指定 return_tensors，返回普通 list；这里只用来量出提示部分占了多少个
        # token（len(source_id)），供第 4 步屏蔽标签时定位提示部分的范围。
        # 注意：此时 add_eos_token 未开启，提示部分末尾不会追加 EOS。
        source_id = tokenizer.encode(s, max_length=tokenizer.model_max_length, truncation=True)

        # ── 第 2 步：打开 add_eos_token，对 s + t 整体分词 ────────────────────────
        # add_eos_token=True 表示"在序列末尾自动追加一个 EOS 结束符"，
        # 让模型学会在答案结束时输出 EOS（停止生成的信号）。
        # return_tensors='pt' 使返回值为 PyTorch 张量，形状为 [1, seq_len]，
        # 取 [0] 去掉批次维度，得到一维张量 input_id（长度 = L_s + t 的 token 数 + 1 个 EOS）。
        tokenizer.add_eos_token = True
        input_id = tokenizer.encode(s + t, max_length=tokenizer.model_max_length, truncation=True,
                                    return_tensors='pt')[0]

        # ── 第 3 步：复位 add_eos_token ───────────────────────────────────────────
        # tokenizer 是全局共享对象，此处临时改了它的属性，用完必须还原，
        # 否则会影响后续其他样本（以及外部代码）的分词结果。
        tokenizer.add_eos_token = False

        # ── 第 4 步：构造标签并屏蔽提示部分 ───────────────────────────────────────
        # label 是 input_id 的深拷贝；把前 len(source_id) 个位置（提示部分）设为 -100。
        # 交叉熵损失（CrossEntropyLoss）会自动忽略标签为 -100 的位置，
        # 因此模型只学习预测回复部分，不会去预测提示词。
        # 前提：s + t 的前缀 token 与 s 单独分词的 token 一致（未发生截断时成立）。
        label = input_id.clone()
        label[:len(source_id)] = self.IGNORE_INDEX
        return input_id, label

    # ============================================================================
    # 数据集处理主流程：加载数据 → 构造提示模板 → 分词 → 生成标签
    # ----------------------------------------------------------------------------
    # 参数：tokenizer —— 分词器对象，原样传给 encode_src_tgt 使用
    # 返回：(input_ids, labels)
    #   input_ids —— 列表，每个元素是一条样本的输入张量（形状 [seq_len]）
    #   labels    —— 列表，每个元素是对应样本的标签张量（提示部分为 -100）
    # ----------------------------------------------------------------------------
    # ▍调用到的函数/方法说明：
    #   json.load(open(路径))  —— 读取 JSON 文件并解析为 Python 对象；
    #                             本数据集文件内容是一个列表，因此得到 list[dict]，
    #                             每个 dict 是一条样本（含 instruction / input / output 字段）
    #   dict.pop('output')     —— 取出并删除 'output' 键的值并返回该值；
    #                             这里把返回值赋给 example['response']，等效于"键改名"
    #   str.format_map(dict)   —— 用 dict 中的字段值填充字符串模板里的 {占位符}；
    #                             例：模板 "...{instruction}..." 配 dict={"instruction": "hi"}
    #                             → "...hi..."
    #   str.strip()            —— 去掉字符串首尾的空白字符（空格、\n、\t 等）
    # ----------------------------------------------------------------------------
    # ▍完整示例：
    #   假设数据集中一条样本：
    #     {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."}
    #   （样本没有 "input" 字段 → 选择 prompt_no_input 模板）
    #   s = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    #       + "\n### Instruction:\n" + "What is the capital of France?" + "\n### Output:\n"
    #   t = "The capital of France is Paris."          # output 去掉首尾空白
    #   然后调用 encode_src_tgt(s, t, tokenizer) 得到该样本的 (input_id, label)，
    #   依次收集进 input_ids / labels 两个列表。
    # ============================================================================
    def process(self, tokenizer):
        # 两个列表分别收集所有样本的输入张量和标签张量
        input_ids = []
        labels = []
        # 读取 JSON 数据集文件：文件内容是一个列表，每个元素是一条样本 dict
        list_data_dict = json.load(open(self.args.dataset))

        for example in list_data_dict:
            # 把样本中的 "output" 字段改名为 "response"：
            # pop('output') 取出该键的值并删除原键，再写回新键 response
            example['response'] = example.pop('output')
            # 根据样本是否带 "input" 字段二选一：
            #   带 input（有额外上下文）→ 使用 prompt_input 模板；
            #   不带 input（纯指令）   → 使用 prompt_no_input 模板。
            # format_map(example) 用样本的字段值填充模板里的 {instruction} / {input}
            # 占位符，拼出完整的提示文本 s。
            s = self.format_template["prompt_input"].format_map(example) if 'input' in example.keys(
            ) else self.format_template["prompt_no_input"].format_map(example)
            # t 是期望的标准答案，strip() 去掉首尾空白字符（换行、空格等）
            t = example['response'].strip()
            # 对 (提示, 答案) 分词并生成带 -100 掩码的标签（详见 encode_src_tgt）
            input_id, label = self.encode_src_tgt(s, t, tokenizer)
            input_ids.append(input_id)
            labels.append(label)
        # 返回全部样本的输入和标签，__init__ 中会据此初始化 self.input_ids / self.labels
        return input_ids, labels


if __name__ == "__main__":
    instruction_template = "\n### Instruction:\n"
    response_template = "\n### Output:\n"
    test_prompt = ("Below is an instruction that describes a task, paired with an input that provides further context. " +
    "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
    " {input}" + response_template)
    test_data = dict(instruction="test instruction 123", input="test input 123")
    res = test_prompt.format_map(test_data)
    print(res)
