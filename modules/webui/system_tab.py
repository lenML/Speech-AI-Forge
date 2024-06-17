import gradio as gr

from modules.webui import webui_config


def create_system_tab():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"info")

        with gr.Column(scale=5):
            toggle_experimental = gr.Checkbox(
                label="Enable Experimental Features",
                value=webui_config.experimental,
                interactive=False,
            )
