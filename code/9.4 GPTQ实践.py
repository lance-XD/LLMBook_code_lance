# ============================================================================
# 9.4 GPTQ 量化实践：用 GPTQ 方法加载 4bit 量化模型
# ----------------------------------------------------------------------------
# 什么是 GPTQ：一种"训练后量化"（Post-Training Quantization, PTQ）算法，
# 与 9.3 节 bitsandbytes 的零样本量化不同：
#   · bitsandbytes（load_in_4bit）—— 加载时直接量化，无需校准数据
#   · GPTQ —— 需要先拿一小部分"校准数据"（这里用 C4 数据集）过一遍模型，
#             统计每层权重的真实分布，再逐层求解最优量化参数
#             （基于二阶 Hessian 信息最小化量化误差），因此 4bit 下精度更高
#
# ▍流程概览（本脚本 4 步）：
#   1. 加载分词器 tokenizer（GPTQ 校准阶段要用它把校准文本转成 token）
#   2. 构造 GPTQConfig 量化配置（位宽 4bit、校准数据集 C4、分词器）
#   3. 用 quantization_config 加载模型 → 加载过程中完成 GPTQ 量化
#   4. 打印加载后占用的显存
#
# ▍显存对比（13B 参数模型）：
#   fp16 ≈ 26GB；4bit（GPTQ）≈ 6.5GB —— 4bit 量化把显存需求降到约 1/4
#
# ⚠ 注意事项：
#   1. 需要 GPU（CUDA 环境），且需安装 auto-gptq 库（pip install auto-gptq）
#   2. 校准阶段会用 C4 数据集跑一遍模型，耗时比直接加载更长
#   3. 首次运行需联网下载模型权重和校准数据集
# ============================================================================

# GPTQ 实战
import torch  # 补充导入：脚本中使用 torch.cuda.memory_allocated() 需要显式导入 torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
name = "yulan-team/YuLan-Chat-2-13b-fp16"

# ============================================================================
# 4bit 模型量化（GPTQ）
# ----------------------------------------------------------------------------
# ▍第 1 步：加载分词器
#   AutoTokenizer.from_pretrained(name)
#     —— 加载与模型配套的分词器（文本 ↔ token id 的映射工具）；
#        在 GPTQ 中分词器不是给推理用的，而是【校准阶段】用来把 C4 校准
#        数据集里的文本转成 token 序列，喂给模型统计权重分布
# ============================================================================
# 4bit模型量化
tokenizer = AutoTokenizer.from_pretrained(name)

# ============================================================================
# ▍第 2 步：构造 GPTQ 量化配置
#   GPTQConfig 参数详解：
#     bits=4            —— 量化位宽：把权重压缩到 4bit（每个权重 0.5 字节），
#                          13B 参数 ≈ 6.5GB 显存
#     dataset="c4"      —— 校准数据集名称：C4（Common Crawl 清洗后的网页语料），
#                          GPTQ 算法会用它跑一小步前向过程，收集每层权重的
#                          Hessian 信息，从而为每个权重块求解最优量化参数
#                          （scale / 零点），最小化量化误差
#     tokenizer=...     —— 上面加载的分词器，用于把 C4 校准文本切分成 token
# ============================================================================
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

# ============================================================================
# ▍第 3 步：用量化配置加载模型
#   AutoModelForCausalLM.from_pretrained(name, device_map="auto", quantization_config=...)
#     —— 加载模型时按 quantization_config 的设定执行 GPTQ 量化：
#        · name                —— 模型名（从 HuggingFace Hub 下载 13B 权重）
#        · device_map="auto"   —— 自动把模型各层分配到所有可用 GPU / CPU
#        · quantization_config —— 传入上面的 GPTQConfig，触发量化加载流程
#   量化过程中模型会先以 fp16 载入 → 用 C4 校准数据统计分布 →
#   逐层量化为 4bit 并替换权重 → 最终得到的 model 即可直接用于推理
# ============================================================================
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", quantization_config=quantization_config)

# ============================================================================
# ▍第 4 步：打印显存占用
#   torch.cuda.memory_allocated()
#     —— 返回当前进程在 GPU 上已分配的显存字节数，除以 1000^3 换算成 GB。
#        （此处数值含 PyTorch 缓存，实际占用以 nvidia-smi 为准）
#   预期结果：4bit 量化后约为 fp16 版本的 1/4（13B 模型 ≈ 6.5GB 左右）
# ============================================================================
print(f"memory usage: {torch.cuda.memory_allocated()/1000/1000/1000} GB") 