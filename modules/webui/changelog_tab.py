import gradio as gr


def read_local_changelog():
    with open("CHANGELOG.md", "r", encoding="utf-8") as file:
        content = file.read()
        content = content[content.index("# ") :]
        return content


def create_changelog_tab():
    changelog_content = read_local_changelog()
    gr.Markdown(changelog_content, elem_classes=["no-translate"])
