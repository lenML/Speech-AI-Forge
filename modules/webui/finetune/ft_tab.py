import gradio as gr

from modules.webui.finetune.speaker_ft_tab import create_speaker_ft_tab


def create_ft_tabs(demo):
    with gr.Tabs():
        with gr.TabItem("Speaker"):
            create_speaker_ft_tab(demo)
        with gr.TabItem("GPT"):
            gr.Markdown("🚧 Under construction")
        with gr.TabItem("AE"):
            gr.Markdown("🚧 Under construction")
