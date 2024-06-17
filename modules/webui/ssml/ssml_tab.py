import gradio as gr

from modules.webui import webui_config
from modules.webui.webui_utils import synthesize_ssml


def create_ssml_interface():
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("📝SSML Input")
                gr.Markdown("SSML_TEXT_GUIDE")
                ssml_input = gr.Textbox(
                    label="SSML Input",
                    lines=10,
                    value=webui_config.localization.DEFAULT_SSML_TEXT,
                    placeholder="输入 SSML 或选择示例",
                    elem_id="ssml_input",
                    show_label=False,
                )
                ssml_button = gr.Button("🔊Synthesize SSML", variant="primary")
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("🎛️Parameters")
                # batch size
                batch_size_input = gr.Slider(
                    label="Batch Size",
                    value=4,
                    minimum=1,
                    maximum=webui_config.max_batch_size,
                    step=1,
                )
            with gr.Group():
                gr.Markdown("🎛️Spliter")
                eos_input = gr.Textbox(
                    label="eos",
                    value="[uv_break]",
                )
                spliter_thr_input = gr.Slider(
                    label="Spliter Threshold",
                    value=100,
                    minimum=50,
                    maximum=1000,
                    step=1,
                )

            with gr.Group():
                gr.Markdown("💪🏼Enhance")
                enable_enhance = gr.Checkbox(value=True, label="Enable Enhance")
                enable_de_noise = gr.Checkbox(value=False, label="Enable De-noise")

            with gr.Group():
                gr.Markdown("🎄Examples")
                gr.Examples(
                    examples=webui_config.localization.ssml_examples,
                    inputs=[ssml_input],
                )

    ssml_output = gr.Audio(label="Generated Audio", format="mp3")

    ssml_button.click(
        synthesize_ssml,
        inputs=[
            ssml_input,
            batch_size_input,
            enable_enhance,
            enable_de_noise,
            eos_input,
            spliter_thr_input,
        ],
        outputs=ssml_output,
    )

    return ssml_input
