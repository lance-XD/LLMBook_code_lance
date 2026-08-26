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
    lst = texts.split()
    for text in lst:
        text = " ".join(text) + " </w>"
        tokens[text] += 1
    return tokens


def frequency_of_pairs(frequencies):
    """
    计算成对的频率
    :param frequencies: 统计的词内容
    :return: 词对
    """
    pairs = Counter()
    for token, freq in frequencies.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
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
    bigram = re.escape(" ".join(pair))
    merged = ''.join(pair)
    new_vocab = Counter()
    for token in vocab:
        # 用正则表达式进行替换
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
    # 分词表
    vocab = extract_frequencies(texts)
    # 词汇表
    tokens = Counter()
    for token, freq in vocab.items():
        symbols = token.split()
        # 将单个词的内容整体分割成单个字符更新
        tokens.update(symbols)

    # 已经达到词汇表上限直接返回
    if len(tokens) > limit:
        return tokens
    # 计算需要循环的次数
    cnt = limit - len(tokens)
    for _ in range(cnt):
        pairs = frequency_of_pairs(vocab)
        # print(pairs)
        if not pairs:
            break
        # 取出频次最高的分词
        most_frequent = pairs.most_common(1)
        # 更新词汇表
        tokens["".join(most_frequent[0][0])] = most_frequent[0][1]
        vocab = merge_vocab(most_frequent[0][0], vocab)
    return tokens


# data = "this is my destiny!"
data = ("当遇到“未登录词”时，BPE不会报错，它会将这个词拆分成已知的最小子词单元（甚至退化到单个字符）。"
        "因为字节级BPE最终可以拆成单个字节（0~255），所以任何文本都能被切分，绝对没有未知词。")
# 测试合并之后，不超过15个分词的情况
token_limit = 80
bpe_vocab = encode_with_bpe(data, token_limit)
print(f"bpe处理后的词汇表为：{bpe_vocab}")
