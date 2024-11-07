import gradio as gr

from modules import config
from modules.webui.speaker.speaker_creator import speaker_creator_ui
from modules.webui.speaker.speaker_editor import speaker_editor_ui
from modules.webui.speaker.speaker_editor_v2 import speaker_editor_ui_v2
from modules.webui.speaker.speaker_merger import create_speaker_merger
from modules.webui.speaker.spk_hub import create_spk_hub_ui


def create_speaker_panel():

    with gr.Tabs():

        with gr.Tab("Builder"):
            speaker_editor_ui_v2()
        with gr.Tab("Test Voice"):
            speaker_editor_ui()
        with gr.Tab("ChatTTS"):
            with gr.Tab("Creator"):
                speaker_creator_ui()
            with gr.Tab("Merger"):
                create_speaker_merger()
        with gr.Tab("Hub"):
            create_spk_hub_ui()
