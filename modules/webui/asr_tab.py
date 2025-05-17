import gradio as gr

from modules.webui import webui_config
from modules.webui.asr_tabs.transcribe_tab import create_whisper_asr_tab


def create_asr_tab():

    with gr.Tabs():
        with gr.TabItem("Transcribe"):
            create_whisper_asr_tab()
