import gradio as gr


def read_local_readme():
    with open("README.md", "r", encoding="utf-8") as file:
        content = file.read()
        content = content[content.index("# ") :]
        return content


def create_readme_tab():
    readme_content = read_local_readme()
    gr.Markdown(readme_content, elem_classes=["no-translate"])
