import gradio as gr
from modules.webui.webui_utils import (
    synthesize_ssml,
)
from modules.webui import webui_config
from modules.webui.examples import ssml_examples, default_ssml


def create_ssml_interface():
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("📝SSML Input")
                gr.Markdown(f"- 最长{webui_config.ssml_max:,}字符，超过会被截断")
                gr.Markdown("- 尽量保证使用相同的 seed")
                gr.Markdown(
                    "- 关于SSML可以看这个 [文档](https://github.com/lenML/ChatTTS-Forge/blob/main/docs/SSML.md)"
                )
                ssml_input = gr.Textbox(
                    label="SSML Input",
                    lines=10,
                    value=default_ssml,
                    placeholder="输入 SSML 或选择示例",
                    elem_id="ssml_input",
                    show_label=False,
                )
                ssml_button = gr.Button("🔊Synthesize SSML", variant="primary")
        with gr.Column(scale=1):
            with gr.Group():
                # 参数
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
                gr.Markdown("💪🏼Enhance")
                enable_enhance = gr.Checkbox(value=True, label="Enable Enhance")
                enable_de_noise = gr.Checkbox(value=False, label="Enable De-noise")

            with gr.Group():
                gr.Markdown("🎄Examples")
                gr.Examples(
                    examples=ssml_examples,
                    inputs=[ssml_input],
                )

    ssml_output = gr.Audio(label="Generated Audio", format="mp3")

    ssml_button.click(
        synthesize_ssml,
        inputs=[ssml_input, batch_size_input, enable_enhance, enable_de_noise],
        outputs=ssml_output,
    )

    return ssml_input
