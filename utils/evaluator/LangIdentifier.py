# -*- coding: utf-8 -*-
"""
utils/evaluator/LangIdentifier.py —— 语言识别评估器（fastText 封装）

【作用】
给一段文本打出"语言标签 + 置信分数"，是 4.1 质量过滤（FilterPassageByLangs）的底层工具：
    evaluate_single_text(text) → (labels, scores)
      labels —— 语言标签列表（如 ['en']），按置信度从高到低排序
      scores —— 与 labels 一一对应的置信分数（如 [0.83]）

【库依赖关系】
- utils.evaluator.evaluator_base.EvaluatorBase —— 本项目评估器基类：
    定义统一接口（evaluate_single_text / evaluate_single_pair / evaluate_pairwise_pairs），
    本类继承它并【重写】evaluate_single_text：从"恒返回 1.0 的占位实现"变成真实的 fastText 打分
- fasttext —— 官方 fastText 库（安装：pip install fasttext-wheel，依赖 numpy/scipy/pybind11）：
    fasttext.load_model(path) —— 加载 .bin 格式的预训练模型
    model.predict(text)       —— 预测文本语言，返回 (labels, scores)：
                                   labels 形如 ('__label__en',)，scores 形如 (0.83,)；
                                   默认 k=1 只返回置信度最高的 1 种语言
"""
import os

from utils.evaluator.evaluator_base import EvaluatorBase
# 安装依赖：fasttext 依赖于 numpy, scipy 和 pybind11。通常它们已存在，若没有可运行：pip install numpy scipy pybind11
# 安装 fasttext：pip install fasttext-wheel
import fasttext

class LangIdentifier(EvaluatorBase):
    def __init__(self, model_path: str = "../models/fasttext/lid.176.bin"):
        # model_path —— fastText 语言识别模型文件（lid.176.bin）路径；
        #               传空字符串则不加载模型（self.model = None，供无模型场景使用）
        # 下面两个属性来自父类 EvaluatorBase.__init__（统一接口约定），这里重新赋值
        super().__init__()
        self.input_path = ""
        self.output_path = ""
        # 置信度阈值：自身不直接使用，供上层（如 4.1 FilterPassageByLangs）读取，
        # 低于该值的最高分文本应被判定为"未知语言"
        self.reject_threshold = 0.5
        if model_path:
            # fasttext.load_model(path) —— 加载 fastText 预训练模型（.bin 格式）
            #   lid.176.bin：官方语言识别模型，覆盖 176 种语言，
            #   文件约百 MB（位于仓库 utils/models/fasttext/ 下），加载耗时数秒
            self.model = fasttext.load_model(model_path)

        else:
            # 获取当前脚本 LangIdentifier.py 所在的绝对目录
            # 即：D:/Pycharm/Projects/LLMBook_code_lance/utils/evaluator/
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # 从 evaluator 目录回到上一级 utils，再进入 models/fasttext
            # 构建出完整准确的绝对路径
            abs_model_path = os.path.normpath(os.path.join(script_dir, "..", "models", "fasttext", "lid.176.bin"))

            self.model = fasttext.load_model(abs_model_path)

    def _regularize_text(self, text: str) -> str:
        # 文本预处理：把多行文本压成一行
        #   先拷贝一份再修改，避免影响调用方的原始字符串（str 本身不可变，这里
        #   ret = text 后通过 replace 生成新字符串返回，不修改原 text）
        ret = text
        replace_list = ['\n']   # 需要清除的字符列表（当前只有换行符，可按需扩充）
        for replace_char in replace_list:
            # str.replace(旧字符, 新字符)：把文本中所有 replace_char 替换为 ''（即删除）。
            # 作用：fastText 模型按"一行一句话"训练，换行符会干扰语言判断，
            #       先把换行去掉，让模型面对的是干净的单行文本
            ret = ret.replace(replace_char, '')
        return ret

    def evaluate_single_text(self, text: str) -> tuple:
        # ★ 重写父类的同名方法：从"恒返回 1.0 的占位实现"变成真正的 fastText 打分
        # 参数 text —— 待识别语言的文本（任意长度，内部会先做 _regularize_text 预处理）
        # 返回值 (labels, scores)：
        #   labels —— 语言标签【列表】（按置信度降序，如 ['en']）
        #   scores —— 与 labels 一一对应的置信分数（如 [0.83]）
        text = self._regularize_text(text)
        # model.predict(text) —— fastText 预测文本语言，返回两个元组：
        #   labels —— 形如 ('__label__en',)，带 '__label__' 前缀
        #   scores —— 形如 (0.83,)，与 labels 一一对应
        # 默认 k=1：只返回置信度最高的 1 种语言；需要多个候选时用 model.predict(text, k=N)
        labels, scores = self.model.predict(text)
        # 去掉 '__label__' 前缀：把 ('__label__en',) 变成 ['en']，
        # 与上层（4.1 的 accept_lang_list）中使用的小写 ISO 语言代码对齐
        labels = [label.replace('__label__', '') for label in labels]
        return labels, scores


if __name__ == '__main__':
    # 加载语言识别模型（lid.176.bin 已随仓库提供）
    langidentifier = LangIdentifier(
        model_path="../models/fasttext/lid.176.bin"
    )
    # 6 条测试文本：
    #   1、2、5 号 —— 正常英文/中文；
    #   3 号 —— 被换行符、制表符污染的英文（验证 _regularize_text 删 \n 的效果）；
    #   4 号 —— 乱码/特殊字符文本（验证低置信场景）；
    #   6 号 —— 混入 HTML 标签的英文（验证对噪声的鲁棒性）
    texts = [
        "I love openai too much! It invented ChatGPT and GPT4 such tramendous inventions!!",
        "这是一段文本",
        "I lov\n\ne open\ni too much! It in  ve\tnted ChatG PT and GPT4 such tra me\tndous inv\tenti\tons!!",
        "適\n\n\t\n湜①\n葮焱\t暒 妏",
        "这是一段文本",
        "I love openai too <br> much! It invented ChatGPT and GPT4 such </br> tramendous inventions!!",
    ]
    for text in texts:
        label, score = langidentifier.evaluate_single_text(text)
        print(label, score)

    # 真实运行输出（项目根目录执行 python -m utils.evaluator.LangIdentifier）：
    #   ['en'] [0.8277927]   1 号：正常英文 → en，置信 0.83
    #   ['zh'] [1.00005996]  2 号：正常中文 → zh（注意 fastText 分数可略超 1，非严格归一化）
    #   ['en'] [0.78517467]  3 号：换行+制表符污染的英文 → 仍识别为 en（\n 已被删除）
    #   ['ba'] [0.2688787]   4 号：乱码文本 → 最高分仅 0.27 → 上层(4.1)会判为"未知语言"并丢弃
    #   ['zh'] [1.00005996]  5 号：中文 → zh
    #   ['en'] [0.75473583]  6 号：带 <br> 标签的英文 → en
