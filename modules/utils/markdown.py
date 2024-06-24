import mistune


class PlainTextRenderer(mistune.HTMLRenderer):
    def text(self, text):
        return text

    def link(self, text, url, title=None):
        return text

    def image(self, alt, url, title=None):
        return alt

    def emphasis(self, text):
        return text

    def strong(self, text):
        return text

    def block_code(self, code, info=None):
        # remove code
        return ""

    def block_quote(self, text):
        return text

    def heading(self, text, level):
        return text + "\n"

    def newline(self):
        return "\n"

    def list(self, text: str, ordered: bool, **attrs) -> str:
        if ordered:
            html = ""
            return html + "\n" + text + "\n"
        return "\n" + text + "\n"

    # FIXME: 现在的 list 转换没法保留序号
    def list_item(self, text):
        return "" + text + "\n"

    def paragraph(self, text):
        return text + "\n"

    def codespan(self, text: str) -> str:
        # remove code
        return ""

    def thematic_break(self) -> str:
        # remove break
        return "\n"


def markdown_to_text(markdown_text):
    renderer = PlainTextRenderer()
    markdown = mistune.create_markdown(renderer=renderer)
    text = markdown(markdown_text)
    text = text.strip()
    return text


if __name__ == "__main__":
    markdown_text = """
# 标题

这是一个示例文本，其中包含 **加粗**、*斜体*、[链接](http://example.com) 和其他 Markdown 语法。

```ts
console.log(1)
```

- 列表项 1
- 列表项 2
- 列表项 3

1. 第一
2. 第二

> 这是一个引用。

`代码片段`
    """
    plain_text = markdown_to_text(markdown_text)
    print(plain_text)
