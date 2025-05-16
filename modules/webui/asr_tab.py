import gradio as gr

from modules.webui import webui_config
from modules.webui.asr_tabs.whisper_asr_tab import create_whisper_asr_tab


def create_asr_tab():

    with gr.Tabs():
        with gr.TabItem("Whisper"):
            create_whisper_asr_tab()
        with gr.TabItem("SenseVoice", visible=webui_config.experimental):
            gr.Markdown("🚧 Under construction")
