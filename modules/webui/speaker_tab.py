import gradio as gr

from modules.webui.speaker.speaker_creator import speaker_creator_ui
from modules.webui.speaker.speaker_editor import speaker_editor_ui
from modules.webui.speaker.speaker_merger import create_speaker_merger


def create_speaker_panel():

    with gr.Tabs():
        with gr.Tab("Editor"):
            speaker_editor_ui()
        with gr.TabItem("Creator"):
            speaker_creator_ui()
        with gr.TabItem("Merger"):
            create_speaker_merger()
