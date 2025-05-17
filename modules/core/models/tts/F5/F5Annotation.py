"""
用来标注文本，以支持 f5 模型的 tokenizer
同时支持多音字标注，格式如下：
示例: "三人行(xing2)，必有我师焉"

流程:
1. 首先解析所有标注，如 "三人行(xing2)"
2. 生成初步的 CharToken 列表
3. 去除标注后自动生成拼音
4. 将手动标注覆盖到自动生成的拼音上
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import jieba
from pypinyin import Style, lazy_pinyin


@dataclass(frozen=True, repr=False)
class CharToken:
    text: str
    is_zh: bool
    pos: int
    auto_pinyin: Optional[str] = None
    anno_pinyin: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"CharToken(text={self.text}, is_zh={self.is_zh}, pos={self.pos}, "
            f"auto_pinyin={self.auto_pinyin}, anno_pinyin={self.anno_pinyin})"
        )


class F5Annotation:
    """
    用来标注文本，以支持 f5 模型的 tokenizer
    同时支持多音字标注，格式如下：
    示例: "三人行(xing2)，必有我师焉"
    """

    def __init__(self) -> None:
        self.annotation_pattern = re.compile(r"([\u4e00-\u9fff])(\(\w{1,5}\d\))")

    def remove_anno(self, text: str) -> str:
        """
        移除标注 "行(xing2)" => "行"
        """
        return re.sub(self.annotation_pattern, lambda m: m[1], text)

    def list_annotation(self, text: str) -> List[CharToken]:
        """
        找出所有手动标注的位置
        """
        char_tokens = []
        for m in re.finditer(self.annotation_pattern, text):
            char_tokens.append(
                CharToken(
                    text=m.group(),
                    pos=m.start(),
                )
            )
        return char_tokens

    def auto_pinyin(self, text: str) -> List[CharToken]:
        """
        将使用jieba分词之后自动标注
        """
        tokens = []
        pos = 0

        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))

            if seg_byte_len == len(seg):  # 纯英文和符号
                for c in seg:
                    tokens.append(
                        CharToken(text=c, is_zh=False, pos=pos, auto_pinyin=None)
                    )
                    pos += 1
            elif seg_byte_len == 3 * len(seg):  # 纯中文字符
                pinyins = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for char, py in zip(seg, pinyins):
                    tokens.append(
                        CharToken(text=char, is_zh=True, pos=pos, auto_pinyin=py)
                    )
                    pos += 1
            else:  # 混合中英文
                for c in seg:
                    if ord(c) < 256:  # ASCII字符
                        tokens.append(
                            CharToken(text=c, is_zh=False, pos=pos, auto_pinyin=None)
                        )
                    else:  # 中文字符
                        py = lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)[0]
                        tokens.append(
                            CharToken(text=c, is_zh=True, pos=pos, auto_pinyin=py)
                        )
                    pos += 1

        return tokens

    def list_annotation(self, text: str) -> List[CharToken]:
        """
        找出所有手动标注的位置
        """
        char_tokens = []
        for m in re.finditer(self.annotation_pattern, text):
            # 提取字符和拼音
            char = m.group()[0]  # 第一个字符是汉字
            pinyin = m.group()[2:-1]  # 括号中的拼音
            char_tokens.append(
                CharToken(text=char, is_zh=True, pos=m.start(), anno_pinyin=pinyin)
            )
        return char_tokens

    def pase_text(self, text: str) -> List[CharToken]:
        """
        将文本解析为 char token
        """
        clean_text = self.remove_anno(text)
        anno_tokens = self.list_annotation(text)
        auto_tokens = self.auto_pinyin(clean_text)

        # print("clean_text", clean_text)
        # print("anno_tokens", anno_tokens)
        # print("auto_tokens", auto_tokens)

        # merge anno => auto
        final_tokens: List[CharToken] = []
        anno_pos = [t.pos for t in anno_tokens]
        for tk in auto_tokens:
            if tk.pos in anno_pos:
                anno_tk = anno_tokens[anno_pos.index(tk.pos)]
                assert tk.text == anno_tk.text
                final_tokens.append(
                    CharToken(
                        text=anno_tk.text,
                        is_zh=anno_tk.is_zh,
                        pos=anno_tk.pos,
                        auto_pinyin=tk.auto_pinyin,
                        anno_pinyin=anno_tk.anno_pinyin,
                    )
                )
            else:
                final_tokens.append(tk)

        return final_tokens

    def convert_to_pinyin(self, text: str) -> List[str]:
        """
        将文本解析，并标注为拼音
        """
        tokens = self.pase_text(text)
        ret: list[str] = []

        for token in tokens:
            if token.is_zh and token.text not in "。，、；：？！《》【】—…":
                ret.append(" ")
                ret.append(token.anno_pinyin or token.auto_pinyin)
            else:
                ret.append(token.text)

        return ret


def test_f5_annotation():
    f5_annotation = F5Annotation()

    # 测试案例1: 长/chang2 短音
    text1 = "目前国债的期限长(chang2)短不一。"
    result1 = f5_annotation.convert_to_pinyin(text1)

    # fmt: off
    expect1 = [" ","mu4"," ","qian2"," ","guo2"," ","zhai4"," ","de"," ","qi1"," ","xian4"," ","chang2"," ","duan3"," ","bu4"," ","yi1","。"]
    assert result1 == expect1, f"Case 1 failed: {result1}"

    # 测试案例2: 好/hao4 不好
    text2 = "这件事情好(hao4)说。" 
    result2 = f5_annotation.convert_to_pinyin(text2)
    expect2 = [" ","zhe4"," ","jian4"," ","shi4"," ","qing2"," ","hao4"," ","shuo1","。"]
    assert result2 == expect2, f"Case 2 failed: {result2}"

    # 测试案例3: 中/zhong4 国
    text3 = "他在中(zhong4)国工作。"
    result3 = f5_annotation.convert_to_pinyin(text3)
    expect3 = [" ","ta1"," ","zai4"," ","zhong4"," ","guo2"," ","gong1"," ","zuo4","。"]
    assert result3 == expect3, f"Case 3 failed: {result3}"

    # 测试案例4: 行/xing2 人行
    text4 = "这条是人行(xing2)道。"
    result4 = f5_annotation.convert_to_pinyin(text4)
    expect4 = [" ","zhe4"," ","tiao2"," ","shi4"," ","ren2"," ","xing2"," ","dao4","。"]
    assert result4 == expect4, f"Case 4 failed: {result4}"

    # 测试案例5: 弹/tan2 器
    text5 = "他在弹(tan2)乐器。"
    result5 = f5_annotation.convert_to_pinyin(text5)
    expect5 = [" ","ta1"," ","zai4"," ","tan2"," ","yue4"," ","qi4","。"]
    assert result5 == expect5, f"Case 5 failed: {result5}"

    # 测试案例6: 地/de5 位
    text6 = "这块地(de5)很肥沃。"
    result6 = f5_annotation.convert_to_pinyin(text6)
    expect6 = [" ","zhe4"," ","kuai4"," ","de5"," ","hen3"," ","fei2"," ","wo4","。"]
    assert result6 == expect6, f"Case 6 failed: {result6}"

    # 测试案例7: 数/shu3 学
    text7 = "他正在数(shu3)钱。"
    result7 = f5_annotation.convert_to_pinyin(text7)
    expect7 = [" ","ta1"," ","zheng4"," ","zai4"," ","shu3"," ","qian2","。"]
    assert result7 == expect7, f"Case 7 failed: {result7}"
    
    # 测试案例8: 英语
    text8 = "He is a student."
    result8 = f5_annotation.convert_to_pinyin(text8)
    expect8 = ['H', 'e', ' ', 'i', 's', ' ', 'a', ' ', 's', 't', 'u', 'd', 'e', 'n', 't', '.']
    assert result8 == expect8, f"Case 8 failed: {result8}"

    print("All test cases passed!")
    # fmt: on


# 运行测试
if __name__ == "__main__":
    test_f5_annotation()
