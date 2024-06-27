import html
import re
from html.parser import HTMLParser


# NOTE: 现在没用这个，因为不好解决转义字符的问题
#       除非分段处理，但是太麻烦了...
class HTMLTagRemover(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, data):
        self.fed.append(data)

    def get_data(self):
        return "\n".join(self.fed)


def remove_html_tags(text):
    parser = HTMLTagRemover()
    parser.feed(text)
    return parser.get_data()


def remove_html_tags_re(text):
    text = html.unescape(text)
    html_tags_pattern = re.compile(r"</?([a-zA-Z1-9]+)[^>]*>")
    return re.sub(html_tags_pattern, " ", text)


if __name__ == "__main__":
    input_text = """
<h1>一个标题</h1> 这是一段包含<code>标签</code>的文本。 <code>&amp;</code>
<设定>
一些文本
</设定>
"""
    # input_text = "我&你"
    output_text = remove_html_tags_re(input_text)
    print(output_text)  # 输出： 一个标题 这是一段包含标签的文本。
