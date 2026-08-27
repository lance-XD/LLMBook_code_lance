# -*- coding: utf-8 -*-
"""
4.2 去重：基于"行级 n-gram + Jaccard 相似度"的文本去重

【作用】
互联网语料中存在大量重复/近似重复的句子（如爬虫抓取多次、内容转载）。本类逐行
比较相邻（更准确说：与上一条保留下来的行）的内容相似度，超过阈值 thre_sim 的
行视为重复，直接丢弃。

【库依赖关系】
- nltk.util.ngrams —— NLTK 自然语言工具库中的 n-gram（n 元组）生成函数：
    ngrams(序列, n) → 依次滑动窗口生成 n 元组列表，如 ngrams([a,b,c], 2) → [(a,b),(b,c)]
  安装：pip install nltk（本机已装 3.10.3；ngrams 是纯函数，无需额外下载数据）
- string / re —— Python 标准库：标点符号表 / 正则表达式
- 无其他外部依赖

【算法流程（clean_single_text）】
  1. 按行分隔符（\n）把文本拆成行列表
  2. 每行按"元组分隔符"（ASCII 标点 + 中文标点 + 空格）切成词元，再生成 n-gram 集合
  3. 依次比较相邻（与上一条保留行）n-gram 集合的 Jaccard 相似度：
       相似度 = 交集大小 / 并集大小
     相似度 < thre_sim（0.95）→ 保留该行；否则视为重复行丢弃
  4. 把所有保留行用 \n 重新拼回文本
"""
import string
import re
from nltk.util import ngrams


class CleanerDedupLineByNgram():
    def __init__(self):
        # 定义行分隔符和元组分隔符
        # 行分隔符：用什么把文本切成"行"（这里只有换行符 \n）
        self.line_delimiter = list("\n")
        # 元组分隔符：行内部用什么把文本切成"词元"（n-gram 的基本单位）——
        #   string.punctuation ：ASCII 标点，如 !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        #   中文标点            ：，。！？：；“”‘’（）《》【】、|—
        #   ' '                 ：空格
        # 含义：以"标点/空格"为界切分，让 n-gram 主要捕获"词/短语"级重复
        chinese_punctuation = "，。！？：；“”‘’（）《》【】、|—"
        self.gram_delimiter = list(string.punctuation) + list(chinese_punctuation) + [' ']

    def clean_single_text(self, text: str, n: int = 5, thre_sim: float = 0.95) -> str:
        # 参数说明：
        #   text     —— 待去重的原始文本（多行）
        #   n        —— n-gram 的 n 值（默认 5：以 5 元组衡量行的内容指纹）
        #   thre_sim —— 相似度阈值（默认 0.95：相邻行相似度达到 95% 即判为重复）
        # 返回值：去重后的文本（重复行已被删除）

        # ---- 第 1 步：依靠行分隔符分割所有行 ----
        # re.split(模式, 文本)：按模式切分。
        #   '|'.join(map(re.escape, self.line_delimiter))：
        #     把每个分隔符转义（re.escape，防止 \n 等被当正则元字符）后用 | 连接成
        #     "交替匹配"模式，等价于"按任意一个分隔符切分"；
        #   if each != '' —— 过滤掉连续分隔符产生的空串
        lines = [each for each in re.split('|'.join(map(re.escape, self.line_delimiter)), text) if each != '']

        # ---- 第 2 步：为每行计算 n-gram，临时存入 lineinfo ----
        # lineinfo：每行一个 dict；last：上一条"保留"的行（初始为空）
        lineinfo, last = list(), {}
        for idx, line in enumerate(lines):  # 计算每行的n元组
            # 依靠元组分隔符分割所有n元组，并将其暂时存储到lineinfo里
            #   按"标点/空格"切分该行 → 词元列表 grams（空串同样被过滤）
            grams = [each for each in re.split('|'.join(map(re.escape, self.gram_delimiter)), line) if each != '']
            # ngrams(grams, min(len(grams), n))：
            #   对词元序列生成 n-gram；min(len(grams), n) 处理"行词元数 < n"的情况
            #   （词元不足 n 个时退化为 len(grams) 元组，避免 ngrams 报错）
            computed_ngrams = list(ngrams(grams, min(len(grams), n)))
            lineinfo.append({
                "lineno": idx, "text": line, "n": min(len(grams), n),
                "ngrams": computed_ngrams, "keep": 0   # keep：0=待定，1=保留
            })

        # ---- 第 3 步：逐行与"上一条保留行"比较 Jaccard 相似度，决定去留 ----
        for idx, each in enumerate(lineinfo):  # 过滤掉和相邻行之间n元组的Jaccard相似度超过thre_sim的行
            if last == {}:
                # 第一行无条件保留，并作为后续比较的基准
                each["keep"], last = 1, each
            else:
                # 计算相邻行间的Jaccard相似度
                #   把两行的 n-gram 列表转成集合（去重），便于求交集/并集
                ngrams_last, ngrams_cur = set(last["ngrams"]), set(each["ngrams"])
                ngrams_intersection, ngrams_union = len(ngrams_last.intersection(ngrams_cur)), len(
                    ngrams_last.union(ngrams_cur))
                # Jaccard 相似度 = |交集| / |并集|，取值 0~1：
                #   0 = 两行毫无共同 n-gram；1 = 两行 n-gram 完全相同（完全重复）
                jaccard_sim = ngrams_intersection / ngrams_union if ngrams_union != 0 else 0
                # 相似度 < 阈值 → 视为"不同内容"，保留该行并把它设为新的比较基准；
                # 相似度 ≥ 阈值 → 视为重复行，keep 保持 0（丢弃）
                if jaccard_sim < thre_sim:
                    each["keep"], last = 1, each

        # ---- 第 4 步：将所有未被过滤掉的行重新拼接起来 ----
        # 只保留 keep==1 的行，用行分隔符（\n）重新连接成文本
        text = self.line_delimiter[0].join([each["text"] for each in lineinfo if each["keep"] == 1])
        return text


if __name__ == "__main__":
    # 测试文本：3 行 —— 第 2 行是第 1 行的近似重复（内容相同、带缩进），应被删除
    test_text_1 = """在机器学习、人工智能和数据分析领域，这是一个非常常见的术语，指在建模或分析之前对原始数据进行清洗、整理、
    在机器学习、人工智能和数据分析领域，这是一个非常常见的术语，指在建模或分析之前对原始数据进行清洗、整理、
    也具备一定的专业感和科技感，我可以进一步为你展示它们在处理 HTML 标签或 Markdown 语法时的具体清洗代码示例。"""
    n_grams = CleanerDedupLineByNgram()
    res = n_grams.clean_single_text(test_text_1)
    print(res)

    # 真实运行输出：只剩第 1、3 行（第 2 行与第 1 行 n-gram 相似度过高被丢弃）：
    #   在机器学习、人工智能和数据分析领域，这是一个非常常见的术语，指在建模或分析之前对原始数据进行清洗、整理、
    #       也具备一定的专业感和科技感，我可以进一步为你展示它们在处理 HTML 标签或 Markdown 语法时的具体清洗代码示例。
