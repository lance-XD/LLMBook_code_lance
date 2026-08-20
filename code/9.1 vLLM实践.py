# -*- coding: utf-8 -*-
"""
9.1 vLLM实践：用 vLLM 引擎做高性能大模型推理（Llama-2-7B-Chat 聊天示例）

【库依赖关系】
- vllm : 高性能 LLM 推理引擎（伯克利 SkyLab 团队开发）——
    vllm.LLM            —— 模型类：加载权重、管理 KV 缓存、执行批量生成
    vllm.SamplingParams —— 解码超参配置类（temperature、max_tokens、惩罚系数等）
  注意：vLLM 仅支持 Linux + CUDA（Windows 无法安装）。本机（win32）未安装时，
        文件会自动改跑底部的 demo_sampling_concepts() 纯 torch 概念演示。

【vLLM 为什么快（对比 HuggingFace 的 model.generate）】
1. PagedAttention（分页注意力）：把 KV 缓存按"页"管理，像操作系统虚拟内存一样
   消除显存碎片 → 显存利用率接近 100% → 可同时服务超大 batch；
2. Continuous Batching（连续批处理）：任一请求生成完毕立刻腾出位置给新请求，
   不像传统批处理必须等整批请求全部结束才换下一批；
3. 算子融合 / 量化 / 显存预分配等工程优化。
   综合效果：吞吐量通常比 HF 实现快数倍到数十倍。

【整体流程】
定义 prompts（Llama-2 Chat 格式）→ vllm.LLM 加载模型 → 配置 SamplingParams
→ model.generate 批量生成 → 逐个打印输入/输出
"""
import torch
import torch.nn.functional as F

# vllm 库：高性能 LLM 推理引擎（仅支持 Linux + CUDA）。
# 用 try/except 包裹：未安装 vllm 的环境（如本机 Windows）仍可运行
# 文件底部的 demo_sampling_concepts() 概念演示；真实推理需在 Linux 环境执行。
try:
    import vllm
except ImportError:
    vllm = None


def vllm_generation_demo():
    """vLLM 真实推理（需 Linux + GPU + pip install vllm，首次运行会下载 Llama-2-7B-Chat 权重）"""
    if vllm is None:
        print("vLLM 仅支持 Linux + CUDA，当前环境无法运行真实推理；可运行底部概念演示。")
        return

    # 符合LLaMA-2 Chat格式的三个提示
    #   [INST] ... [/INST] 是 Llama-2 的对话模板（INST = Instruction，指令）：
    #   用户输入夹在 [INST] 与 [/INST] 之间，模型会在此基础上续写 [/INST] 之后的回复。
    #   （Llama-3 等新模型改用 <|begin_of_text|> 等特殊 token 模板，可调用
    #     tokenizer.apply_chat_template() 自动套用，无需手写）
    prompts = [
        '[INST] How are you? [/INST]',
        '[INST] 1 + 1 = ? [/INST]',
        '[INST] Can you tell me a joke? [/INST]',
    ]

    # 初始化vLLM的模型
    #   vllm.LLM(model=模型名/路径)
    #     model —— 模型标识：HuggingFace 模型名（首次自动下载）或本地模型目录，
    #               如 'meta-llama/Llama-2-7b-chat-hf'（Llama-2 的对话微调版）
    #   其他常用参数（本书未列）：
    #     tensor_parallel_size   —— 用几张 GPU 做张量并行（如 2 = 权重切分到 2 卡）
    #     gpu_memory_utilization —— 显存占用比例（默认 0.9；0.85 可给其他程序留余量）
    #   初始化会加载权重并做预热，耗时较长
    model = vllm.LLM(model='meta-llama/Llama-2-7b-chat-hf')

    # 设置vLLM的解码参数
    #   vllm.SamplingParams(...) 各参数含义：
    #     temperature —— 采样温度：
    #                    0 = 贪心搜索（永远选概率最大的 token，结果确定）；
    #                    >0 越高越随机；公式 P(token) ∝ exp(logit / T)（见底部示例 1）
    #     max_tokens  —— 新生成 token 数的上限（不含 prompt 部分；vLLM 默认只有 16，
    #                    这里给足 2048，保证长回复不被截断）
    #     presence_penalty —— 存在惩罚：对"出现过的 token"施加固定惩罚（与其出现几次无关），
    #                        鼓励模型引入新词；实现 ≈ logit -= penalty（若该 token 出现过）
    #     frequency_penalty—— 频率惩罚：按"出现次数"成比例惩罚（logit -= penalty × 次数），
    #                        比 presence_penalty 更严厉，专治重复（见底部示例 2）
    #   其他常用参数：top_p / top_k（核/束采样）、stop（停止生成串）、repetition_penalty 等
    sampling_params = vllm.SamplingParams(
        temperature=0,  # 温度设置为0表示贪心搜索
        max_tokens=2048,  # 新生成token数上限
        presence_penalty=0,  # 存在惩罚系数
        frequency_penalty=0,  # 频率惩罚系数
    )

    # 调用vLLM的模型进行生成
    #   model.generate(prompts, sampling_params=sampling_params)
    #     prompts         —— 输入提示的【列表】（一次批量生成多条；vLLM 的连续批处理
    #                        正是为这种多请求高并发场景设计的）
    #     sampling_params —— 上面配置的解码参数对象
    #   返回值 out：与 prompts 等长的列表，每个元素是 RequestOutput：
    #     it.outputs       —— 生成结果列表（未指定 n 参数时只有 1 条）
    #     it.outputs[0].text —— 模型实际生成的文本（不含 prompt 部分）
    out = model.generate(prompts, sampling_params=sampling_params)
    # zip(prompts, out)：把"输入提示"和"对应输出"一对一配对后并行迭代，
    # 逐个打印 输入 → 输出
    for prompt, it in zip(prompts, out):
        print(f'input = {prompt!r}\noutput = {it.outputs[0].text!r}')

    # 样例1
    # input = '[INST] How are you? [/INST]'
    # output = ' I\'m just an AI, I don\'t have feelings or emotions like humans do, so I don\'t have a physical state
    # of being such as "good" or "bad." I\'m here to help answer your questions and provide information to the best
    # of my ability, so please feel free to ask me anything!'

    # 样例2
    # input = '[INST] 1 + 1 = ? [/INST]'
    # output = ' The answer to 1 + 1 is 2.'

    # 样例3
    # input = '[INST] Can you tell me a joke? [/INST]'
    # output = " Of course! Here's a classic one:\n\nWhy don't scientists trust atoms?\n\nBecause they make up
    # everything!\n\nI hope that made you smile! Do you want to hear another one?"


def demo_sampling_concepts():
    """采样参数概念演示（纯 torch，无需 vllm；对应 SamplingParams 各参数的含义）"""
    print("=" * 60)
    print("示例 1：temperature 如何改变概率分布（贪心 vs 随机）")
    # 假设模型对 4 个候选词打出的原始分数（logits）
    logits = torch.tensor([1.0, 2.0, 0.5, 1.5])
    # P(token) ∝ exp(logit / T)：T 越小分布越尖锐（趋近 one-hot），T 越大越平坦（趋近均匀）
    for t, note in [(0.1, "≈贪心：最大者概率≈1，其余≈0"), (1.0, "默认：正常采样"), (4.0, "高温：趋于均匀")]:
        p = F.softmax(logits / t, dim=-1)
        print(f"  T={t:<4} {note}\n     概率 = {[round(v, 4) for v in p.tolist()]}")

    print("=" * 60)
    print("示例 2：presence_penalty 与 frequency_penalty 的差别")
    # 假设词表前 3 个 token 在本序列中已出现 1 / 2 / 3 次
    counts = torch.tensor([1, 2, 3])
    logits = torch.tensor([1.0, 1.0, 1.0])
    frequency_penalty = logits - 0.5 * counts  # logit -= penalty × 出现次数
    presence_penalty = logits - 0.5 * (counts > 0).float()  # logit -= penalty（出现过即罚一次）
    print(f"  原始 logits          : {logits.tolist()}")
    print(f"  frequency(按次数×0.5) : {frequency_penalty.tolist()}  ← 重复越多扣分越多")
    print(f"  presence(出现过×0.5) : {presence_penalty.tolist()}  ← 出现 1 次与 3 次惩罚相同")


if __name__ == "__main__":
    if vllm is None:
        # 无 vllm 环境（如本机 Windows）：跑纯 torch 的概念演示（temperature / 惩罚系数）
        demo_sampling_concepts()
    else:
        # Linux + vllm 环境：执行真实推理（下载权重 → 加载模型 → 批量生成）
        vllm_generation_demo()
