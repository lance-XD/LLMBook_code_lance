# -*- coding: utf-8 -*-
"""
4.3 隐私过滤：用正则表达式把文本中的身份证号替换成掩码

【作用】
互联网语料中常混有个人敏感信息（身份证号、手机号、邮箱等），直接用于训练会
造成隐私泄露。本类用预定义的正则 REGEX_IDCARD 匹配"中国大陆居民身份证号"，
并替换为占位符 **MASKED**IDCARD**。

【库依赖关系】
- utils.rules.regex.REGEX_IDCARD —— 本项目预定义的正则常量（utils/rules/regex.py）：
    r"([1-9]\d{5}[12]\d{3}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])\d{3}[0-9xX])"
    拆解（18 位身份证号的结构）：
      [1-9]\d{5}                —— 前 6 位：地区码（首位非 0）
      [12]\d{3}                 —— 第 7~10 位：出生年份（19xx / 20xx）
      (0[1-9]|1[012])           —— 第 11~12 位：出生月份（01~12）
      (0[1-9]|[12][0-9]|3[01])  —— 第 13~14 位：出生日（01~31）
      \d{3}                     —— 第 15~17 位：顺序码
      [0-9xX]                   —— 第 18 位：校验码（数字或 X/x）
- utils.cleaner.cleaner_base.CleanerBase —— 本项目清洗器基类：
    提供 _sub_re(text, re_text, repl_text) = re.sub(pattern, repl, string)，
    本类直接复用该实现
- re —— Python 标准库正则模块
- 无第三方外部依赖

【运行方式】
代码 import 了项目内的 utils 包，需保证项目根目录在 sys.path 上：
  在项目根目录执行：set PYTHONPATH=. && python "code/4.3 隐私过滤.py"
  （直接 python code/4.3 隐私过滤.py 会因找不到 utils 包而报错）
"""
from utils.rules.regex import REGEX_IDCARD
from utils.cleaner.cleaner_base import CleanerBase


class CleanerSubstitutePassageIDCard(CleanerBase):
    def __init__(self):
        # 调用父类 CleanerBase.__init__（空实现，仅保持接口一致）
        super().__init__()

    def clean_single_text(self, text: str, repl_text: str = "**MASKED**IDCARD**") -> str:
        # 参数说明：
        #   text     —— 待清洗的原始文本（可能包含身份证号）
        #   repl_text—— 替换占位符（默认 "**MASKED**IDCARD**"，可自定义）
        # 返回值：身份证号被替换后的文本
        # 使用正则表达式REGEX_IDCARD匹配身份证号，用repl_text代替
        #   self._sub_re(...) —— 父类 CleanerBase 的方法，等价于 re.sub：
        #     re.sub(pattern=re_text, repl=repl_text, string=text)
        #       pattern  —— REGEX_IDCARD：匹配身份证号的模式（可匹配多处）
        #       repl     —— 替换文本（所有命中的身份证号都被替换成占位符）
        #       string   —— 原文本
        return self._sub_re(text=text, re_text=REGEX_IDCARD, repl_text=repl_text)


if __name__ == "__main__":
    # 运行示例：项目根目录执行  set PYTHONPATH=. && python "code/4.3 隐私过滤.py"
    cleaner = CleanerSubstitutePassageIDCard()
    # 测试文本：含两个身份证号（校验码分别为数字 4 和字母 X），应全部被掩码
    test_text = "我的身份证号是110101199003071234，另一个是11010119900307123X，请保密！"
    print("原始文本:", test_text)
    print("清洗结果:", cleaner.clean_single_text(test_text))

    # 真实运行输出：
    #   原始文本: 我的身份证号是110101199003071234，另一个是11010119900307123X，请保密！
    #   清洗结果: 我的身份证号是**MASKED**IDCARD**，另一个是**MASKED**IDCARD**，请保密！
