import gradio as gr

from modules.webui.webui_utils import get_speakers


# 显示 a b c d 四个选择框，选择一个或多个，然后可以试音，并导出
def create_speaker_panel():
    speakers = get_speakers()

    def get_speaker_show_name(spk):
        pass

    gr.Markdown("🚧 Under construction")
