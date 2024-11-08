import gradio as gr

from modules.webui.audio_tools.post_process import create_post_processor
from modules.webui.audio_tools.video_cut import create_audio_separator


def create_tools_tab():

    with gr.TabItem("Post Process"):
        create_post_processor()
    with gr.TabItem("Video Cut"):
        create_audio_separator()
