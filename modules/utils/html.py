from html.parser import HTMLParser


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


if __name__ == "__main__":
    input_text = "<h1>一个标题</h1> 这是一段包含<code>标签</code>的文本。"
    output_text = remove_html_tags(input_text)
    print(output_text)  # 输出： 一个标题 这是一段包含标签的文本。
