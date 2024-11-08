import gradio as gr

from modules.core.models.zoo import ModelZoo
from modules.webui import webui_config
from modules.webui.webui_utils import synthesize_ssml


def create_ssml_interface():
    with gr.Row():
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
                models = ModelZoo.model_zoo.get_available_tts_model()
                # 选择模型
                selected_model = gr.Dropdown(
                    label="Model",
                    choices=[model.model_id for model in models],
                    value=models[0].model_id if len(models) > 0 else None,
                )

            with gr.Group():
                gr.Markdown("🎛️Spliter")
                eos_input = gr.Textbox(
                    label="eos",
                    value="。",
                )
                spliter_thr_input = gr.Slider(
                    label="Spliter Threshold",
                    value=100,
                    minimum=50,
                    maximum=1000,
                    step=1,
                )

            with gr.Group():
                gr.Markdown("🎛️Adjuster")
                # 调节 speed pitch volume
                # 可以选择开启 响度均衡

                speed_input = gr.Slider(
                    label="Speed",
                    value=1.0,
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                )
                pitch_input = gr.Slider(
                    label="Pitch",
                    value=0,
                    minimum=-12,
                    maximum=12,
                    step=0.1,
                )
                volume_up_input = gr.Slider(
                    label="Volume Gain",
                    value=0,
                    minimum=-12,
                    maximum=12,
                    step=0.1,
                )

                enable_loudness_normalization = gr.Checkbox(
                    value=True,
                    label="Enable Loudness EQ",
                )
                headroom_input = gr.Slider(
                    label="Headroom",
                    value=1,
                    minimum=0,
                    maximum=12,
                    step=0.1,
                )

            with gr.Group():
                gr.Markdown("💪🏼Enhance")
                enable_enhance = gr.Checkbox(value=True, label="Enable Enhance")
                enable_de_noise = gr.Checkbox(value=False, label="Enable De-noise")

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

            with gr.Group():
                gr.Markdown("🎄Examples")
                gr.Examples(
                    examples=webui_config.localization.ssml_examples,
                    inputs=[ssml_input],
                )

            with gr.Group():
                gr.Markdown("🎨Output")
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
            pitch_input,
            speed_input,
            volume_up_input,
            enable_loudness_normalization,
            headroom_input,
            selected_model,
        ],
        outputs=ssml_output,
    )

    return ssml_input
