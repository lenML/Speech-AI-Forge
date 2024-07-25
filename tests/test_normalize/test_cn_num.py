import pytest

from modules.core.tn import ChatTtsTN

ChatTtsTN.ChatTtsTN.remove_block("replace_unk_tokens")


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        # 测试分数
        ("他拿到了 2/3 的票数", "他拿到了 三分之二 的票数"),
        ("1/5 的学生通过了考试", "五分之一 的学生通过了考试"),
        # 测试百分数
        ("项目完成了 80%", "项目完成了 百分之八十"),
        ("通过率达到了 95%", "通过率达到了 百分之九十五"),
        # 测试小数
        ("价格是3.5元", "价格是三点五元"),
        ("他身高1.75米", "他身高一点七五米"),
        # 测试负数
        ("温度降到了-10度", "温度降到了零下十度"),
        ("他的存款变成了-100元", "他的存款变成了负一百元"),
        # 其他测试用例
        ("我有102块钱", "我有一百零二块钱"),
        ("他在1984年出生", "他在一九八四年出生"),
        ("这个项目耗时 2.5 年", "这个项目耗时 二点五 年"),
    ],
)
@pytest.mark.normalize
def test_text_normalize(input_text, expected_output):
    assert ChatTtsTN.ChatTtsTN.normalize(input_text) == expected_output
