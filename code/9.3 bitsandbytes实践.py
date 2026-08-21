# ============================================================================
# 9.3 bitsandbytes 量化实践：加载 8bit / 4bit 量化模型
# ----------------------------------------------------------------------------
# 什么是 bitsandbytes：一个低比特量化库（需要 GPU 的 CUDA 环境），
# 它可以在【加载模型时】就把权重量化成 8bit 或 4bit，而不是先加载完整
# fp16 模型再转换，从而让超大模型能在有限的显存里跑起来。
#
# ▍为什么量化能省显存（以 13B 参数模型为例）：
#   fp16（16 位）：13B × 2 字节 ≈ 26 GB
#   8bit（1 字节）：13B × 1 字节 ≈ 13 GB   （省一半）
#   4bit（0.5 字节）：13B × 0.5 字节 ≈ 6.5 GB （省 3/4）
#   —— 这就是本示例只加载 8bit / 4bit 版本的原因。
#
# ▍两个加载参数的作用：
#   device_map="auto" —— 让 transformers 自动把模型各层分配到所有可用设备
#                        （多张 GPU 按显存分配，放不下时落到 CPU），
#                        避免手动指定设备
#   load_in_8bit=True / load_in_4bit=True —— 启用 bitsandbytes 的 8bit / 4bit
#                        量化加载；模型权重在加载过程中即被量化
#
# ⚠ 注意事项：
#   1. 需要 GPU（CUDA 环境），CPU 上无法使用 bitsandbytes 量化
#   2. 首次加载会从 HuggingFace Hub 下载模型权重，需要联网
# ============================================================================

# bitsandbytes 实战
from transformers import AutoModelForCausalLM
name = "yulan-team/YuLan-Chat-2-13b-fp16"
import torch
# ============================================================================
# 8bit 模型量化加载
# ----------------------------------------------------------------------------
# AutoModelForCausalLM.from_pretrained(模型名, device_map="auto", load_in_8bit=True)
#   —— 加载一个【权重已量化为 8bit】的因果语言模型：
#      · 模型名 name：从 HuggingFace Hub 按名称下载（yulan-team/YuLan-Chat-2-13b-fp16，
#        13B 参数的对话模型，fp16 原始权重约 26GB）
#      · device_map="auto"：自动把模型各层分配到所有可用 GPU / CPU
#      · load_in_8bit=True：启用 bitsandbytes 8bit 量化加载，
#        每个权重只占 1 字节，13B 参数 ≈ 13GB 显存
#       （底层会对每列权重计算 scale/零点做分块仿射量化，推理时再反量化计算）
# 结果：model_8bit 即为可用的量化模型，可直接用于推理/微调
# ============================================================================
# 8bit模型量化
model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
# 打印当前 GPU 已分配的显存：
#   torch.cuda.memory_allocated() —— 返回当前进程在 GPU 上分配的显存字节数，
#   除以 1000^3 换算成 GB 便于阅读
#   注意：这里的数值只是"已分配"的显存，可能包含 PyTorch 缓存，
#   实际占用以 nvidia-smi 为准
print(f"memory usage: {torch.cuda.memory_allocated()/1000/1000/1000} GB")


# ============================================================================
# 4bit 模型量化加载
# ----------------------------------------------------------------------------
# 与 8bit 同理，load_in_4bit=True 把权重量化为 4bit：
#   每个权重只占 0.5 字节，13B 参数 ≈ 6.5GB 显存（比 8bit 再省一半）
# 原理：4bit 量化通常使用"块状量化"——把权重矩阵按小块分别计算
#       scale / zero_point（即 NF4 或 FP4 格式），用更细的粒度降低
#       量化误差，使 4bit 也能保持较好的精度
# 适用场景：单卡显存有限（如 24GB / 16GB）却要跑 13B / 70B 级模型时，
#           4bit 加载几乎是唯一选择（常配合 LoRA 微调，见 10.x 章）
# ============================================================================
# 4bit模型量化
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_4bit=True)
print(f"memory usage: {torch.cuda.memory_allocated()/1000/1000/1000} GB")