# -*- coding: utf-8 -*-
"""
4.1 质量过滤：用 fastText 语言识别模型过滤非目标语言文本（数据预处理的第一步）

【背景：为什么要做语言过滤】
预训练语料来自互联网，混杂上百种语言。若直接训练，模型会把大量容量浪费在
不关心的语言上。质量过滤的第一步就是"只保留目标语言"（如中文/英文）的文本。

【库依赖关系】
- utils.evaluator.LangIdentifier —— 本书配套工具模块：
    封装 fastText 语言识别模型（lid.176.bin，可识别 176 种语言），提供：
      evaluate_single_text(text) → (labels, scores)
        labels —— 语言标签列表，按置信度【从高到低】排序（ISO 639 代码，如 en/fr/zh）
        scores —— 与 labels 一一对应的置信分数（0~1，全部语言分数之和为 1）
    运行前提：pip install fasttext-wheel
- 标准库：无其他外部依赖

【过滤逻辑（filter_single_text 的三种判定）】
输入文本 → fastText 打分 → 依次判断：
  1. 最高置信分 < 阈值 0.5（reject_threshold）→ 文本过短/语言混杂，判为"未知语言"(uk) → 丢弃
  2. 最高分语言不在期望列表 accept_lang_list 中 → 非目标语言 → 丢弃
  3. 最高分语言在期望列表中 → 保留
返回语义：True = 应过滤掉（丢弃），False = 保留。
"""
import os

from utils.evaluator.LangIdentifier import LangIdentifier


class FilterPassageByLangs:
    def __init__(self) -> None:
        # 防御检查：工具模块缺失时给出明确提示，而不是等到调用时报难以理解的错
        if LangIdentifier is None:
            raise RuntimeError(
                "缺少 utils.evaluator.LangIdentifier：请先安装 fasttext 并准备 "
                "utils/evaluator.py 与 utils/models/fasttext/lid.176.bin 模型文件；"
                "或运行底部 demo_filter_logic() 查看过滤判定逻辑。"
            )
        # 使用LangIdentifier模块加载已经训练好的fastText模型,详情见utils/evaluator/LangIdentifier.py
        self.language_identifier = LangIdentifier()
        # 置信度阈值：低于该值的最高分视为"模型拿不准"，文本判为未知语言
        self.reject_threshold = 0.5

    def filter_single_text(self, text: str, accept_lang_list: list) -> bool:
        # 参数说明：
        #   text             —— 待判断的文本（一段话/一行网页文本）
        #   accept_lang_list —— 期望保留的语言列表（如 ["zh", "en"]，通常来自配置文件）
        # 返回值 bool：True = 应过滤掉（丢弃该文本）；False = 保留
        # 调用方通常这样用：if filter.filter_single_text(t, langs): continue  # 丢弃

        # 使用fastText模型给text打分，每种语言生成一个置信分数
        #   evaluate_single_text(text) 返回两个列表（长度 = 模型支持的语言数）：
        #     labels —— 语言标签按置信度【从高到低】排序（如 ["fr", "en", "de", ...]）
        #     scores —— 与 labels 一一对应的置信分（如 [0.88, 0.07, 0.05, ...]）
        #   因此 labels[0] / scores[0] 就是"模型认为最可能的语言"及其置信度
        labels, scores = self.language_identifier.evaluate_single_text(text)
        # 如果text所有语言的分数均比reject_threshold要低，则直接定义为未知语言
        #   最高置信度都 < 0.5 → 文本太短、多语言混杂或模型拿不准 →
        #   把标签直接替换为未知标记 "uk"（unknown）
        #   （注：ISO 639-1 中 "uk" 本义是乌克兰语代码，此处按本书约定当作"未知"使用）
        if scores[0] < self.reject_threshold:
            labels = ["uk"]
        # 期望语言列表统一转小写：与 fastText 返回的小写 ISO 标签对齐，
        # 防止 "EN" 与 "en" 这种大小写不一致导致误判
        accept_lang_list = [each.lower() for each in accept_lang_list]
        # 如果分数最高的语言标签不在配置文件期望的语言列表中，则丢弃该文本
        #   labels[0] ∈ accept_lang_list → 目标语言 → 返回 False（保留）
        #   labels[0] ∉ accept_lang_list → 非目标语言（或未知）→ 返回 True（丢弃）
        if labels[0] not in accept_lang_list:
            return True
        return False


def demo_filter_logic():
    accept_lang_list = ["en", "zh", "fr"]
    text_lst = ["I love openai too much! It invented ChatGPT and GPT4 such tramendous inventions!!",
                "这是一段文本",
                "I lov\n\ne open\ni too much! It in  ve\tnted ChatG PT and GPT4 such tra me\tndous inv\tenti\tons!!",
                "適\n\n\t\n湜①\n葮焱\t暒 妏",
                "这是一段文本",
                "I love openai too <br> much! It invented ChatGPT and GPT4 such </br> tramendous inventions!!", ]
    filter_ = FilterPassageByLangs()
    for text in text_lst:
        res = filter_.filter_single_text(text, accept_lang_list)
        if res:
            print(f"{text}:被过滤！")
        else:
            print(f"{text}:被保留。")


if __name__ == "__main__":
    demo_filter_logic()
