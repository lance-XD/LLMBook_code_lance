# -*- coding: utf-8 -*-
"""
4.4 BPE分词：从零实现 Byte-Pair Encoding（字节对编码）分词算法

【作用】
BPE 是 GPT 等大模型的基础分词算法：从"单个字符"开始，反复把语料中出现频次
最高的相邻字符对合并成一个新符号，直到词表达到预设上限。得到的词表由
"单词 + 子词 + 单字符"混合构成，既能压缩词表、又能覆盖任意文本。

【库依赖关系】
- re                   —— Python 标准库正则模块（合并字符对时做字符串替换）
- collections.Counter  —— Python 标准库计数器（统计频率，most_common 取最高频项）
- 无第三方外部依赖（纯标准库实现）

【BPE 算法流程（对应下方四个函数）】
  1. extract_frequencies ：把文本按空格拆成"词"，每个词内部逐字符空格分隔，
                           末尾加 </w> 词尾标记，统计每个"字符序列"的出现次数
  2. frequency_of_pairs  ：统计所有相邻字符对的频率（按词频加权）
  3. merge_vocab         ：把最高频的字符对合并成一个新符号（正则替换）
  4. encode_with_bpe     ：循环执行 2→3，直到词表达到 limit 上限

【重要特性（__main__ 演示文字的含义）】
  遇到"未登录词"时 BPE 不会报错：词表里没有的整词会被拆成已知子词，
  最坏情况退化为单个字符（字节级 BPE 可拆到单字节 0~255），
  因此"任何文本都能被切分，绝对没有未知词"。
"""
import re
from collections import Counter

"""
对照《大语言模型》中的BPE算法的代码实现
"""


def extract_frequencies(texts):
    """
    计算字符出现的频率
    :param texts: 原始的文本
    :return: 词出现的频率
    """
    tokens = Counter()
    lst = texts.split()          # 按空白把文本切成"词"列表（中文无空格时整个句子算一个"词"）
    for text in lst:
        # " ".join(text)：把词逐字符用空格连接（如 "abc" → "a b c"），
        # 这样每个字符独立成"符号"，后续才能做字符对合并；
        # + " </w>"：在词尾追加独立的词尾标记符号 </w>，
        #   作用：标记词的边界，防止把"词尾字符"与"下一词首字符"错误合并
        text = " ".join(text) + " </w>"
        tokens[text] += 1        # 统计相同字符序列的出现次数
    return tokens


def frequency_of_pairs(frequencies):
    """
    计算成对的频率
    :param frequencies: 统计的词内容
    :return: 词对
    """
    pairs = Counter()
    for token, freq in frequencies.items():
        symbols = token.split()          # 把字符序列重新拆成符号列表（含 </w>）
        for i in range(len(symbols) - 1):
            # 统计所有相邻符号对 (symbols[i], symbols[i+1]) 的频率，
            # 出现次数按所在词的词频 freq 加权
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, vocab):
    """
    合并词汇表中的频繁出现的字符对
    :param pair: 字符对
    :param vocab: 词汇表
    :return: 新的词汇表
    """
    # 给字符加上转义字符
    #   re.escape(" ".join(pair))：把 "a b" 这类待匹配串中的正则元字符转义，
    #   防止字符本身含 . + * 等特殊含义；bigram = "a b"（带空格的形式）
    bigram = re.escape(" ".join(pair))
    merged = ''.join(pair)               # 合并后的新符号，如 ('a','b') → "ab"
    new_vocab = Counter()
    for token in vocab:
        # 用正则表达式进行替换
        #   re.sub(bigram, merged, token)：把该词内的 "a b"（带空格）替换成 "ab"，
        #   即完成"字符对 → 新符号"的合并；未命中的词原样保留
        new_token = re.sub(bigram, merged, token)
        new_vocab[new_token] = vocab[token]
    return new_vocab


def encode_with_bpe(texts, limit):
    """
    对文本进行编码
    :param texts: 源文本
    :param limit: 词元上限
    :return: 生成的词元表
    """
    # 分词表：先做字符级分词，得到"每个字符序列及其频率"（key 是空格分隔的字符串）
    vocab = extract_frequencies(texts)
    # 词汇表：tokens 是最终要返回的词元表，先放入所有【单个字符】符号
    tokens = Counter()
    for token, freq in vocab.items():
        symbols = token.split()
        # 将单个词的内容整体分割成单个字符更新
        #   tokens.update(symbols)：统计每个单字符在所有词中的总出现次数
        tokens.update(symbols)

    # 已经达到词汇表上限直接返回
    #   若单字符种类数已经超过 limit，无法再合并，直接返回
    if len(tokens) > limit:
        return tokens
    # 计算需要循环的次数
    #   每轮合并恰好新增 1 个符号，因此还需合并 (limit - 当前符号数) 次
    cnt = limit - len(tokens)
    for _ in range(cnt):
        pairs = frequency_of_pairs(vocab)   # 统计当前所有相邻符号对的频率
        # print(pairs)
        if not pairs:
            break                          # 没有可合并的字符对了（如只剩一个符号）
        # 取出频次最高的分词
        #   most_common(1) → [(字符对, 频次)]；[0][0] 取字符对，[0][1] 取频次
        most_frequent = pairs.most_common(1)
        # 更新词汇表：把合并后的新符号（如 "ab"）加入词元表，
        #   计数直接取该字符对的频次（教材简化实现；严格实现应按各词内出现次数累加）
        tokens["".join(most_frequent[0][0])] = most_frequent[0][1]
        # 把语料中所有该字符对合并成新符号，得到更新后的分词表，进入下一轮
        vocab = merge_vocab(most_frequent[0][0], vocab)
    return tokens


# data = "this is my destiny!"
data = ("当遇到“未登录词”时，BPE不会报错，它会将这个词拆分成已知的最小子词单元（甚至退化到单个字符）。"
        "因为字节级BPE最终可以拆成单个字节（0~255），所以任何文本都能被切分，绝对没有未知词。")
# 测试合并之后，不超过15个分词的情况
token_limit = 80
bpe_vocab = encode_with_bpe(data, token_limit)
print(f"bpe处理后的词汇表为：{bpe_vocab}")

# 真实运行输出（节选）：词表包含 单字符 + 合并出的子词/整词，
# 例如 'BPE'（由 B+P+E 合并）、'单个字'（由 单+个+字 合并）、
# '当遇到“未登录词”时'（整句高频片段被合并为一个词元）：
#   bpe处理后的词汇表为：Counter({'词': 4, '，': 4, '个': 3, '单': 3, ..., 'BPE': 2,
#   '单个': 2, '单个字': 2, '当遇到“未登录词”时': 1, ...})
