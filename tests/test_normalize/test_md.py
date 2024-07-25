import pytest

from modules.core.tn import ChatTtsTN

ChatTtsTN.ChatTtsTN.remove_block("replace_unk_tokens")


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        # 测试标题
        ("# 你好，世界", "你好,世界"),
        # 测试代码块
        ("```js\nconsole.log('1')\n```", ""),
        # 测试加粗
        ("**加粗**", "加粗"),
        # 测试斜体
        ("*一条文本*", "一条文本"),
        # 测试链接
        ("[Google](https://www.google.com)", "Google"),
        # 测试无序列表
        ("* 项目一\n* 项目二", "项目一\n项目二"),
        # 测试有序列表
        ("1. 第一项\n2. 第二项", "第一项\n第二项"),
        # 测试引用
        ("> 这是一段引用", "这是一段引用"),
        # 测试分隔线
        ("---", ""),
        # 测试混合内容
        (
            """
# 你好，世界

```js
console.log('1')
```

**加粗**

*一条文本*

[Google](https://www.google.com)

* 项目一
* 项目二

1. 第一项
2. 第二项

> 这是一段引用

---
            """.strip(),
            # FIXME: 有序list现在没有序列号...
            "你好,世界\n加粗\n一条文本\nGoogle\n项目一\n项目二\n第一项\n第二项\n这是一段引用",
        ),
    ],
)
@pytest.mark.normalize
def test_text_normalize(input_text, expected_output):
    assert ChatTtsTN.ChatTtsTN.normalize(input_text) == expected_output
