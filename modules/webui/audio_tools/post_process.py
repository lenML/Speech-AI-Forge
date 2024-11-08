import gradio as gr

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from modules.utils import audio_utils


def create_post_processor():
    # 编辑音频文件
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("🎛️Adjuster")
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
                    minimum=-20,
                    maximum=6,
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
                enable_enhance = gr.Checkbox(value=False, label="Enable Enhance")
                enable_de_noise = gr.Checkbox(value=False, label="Enable De-noise")

        with gr.Column(scale=3):
            input_audio = gr.Audio(label="原始音频文件")

            gen_button = gr.Button(
                "Generate",
                variant="primary",
                elem_classes="big-button",
            )

    with gr.Row():
        output_audio = gr.Audio(label="编辑后音频文件")

    def gen_audio(
        audio,
        speed,
        pitch,
        volume_up,
        enable_loudness_normalization,
        headroom,
        enable_enhance,
        enable_de_noise,
    ):
        if audio is None:
            raise gr.Error("Audio file is required.")

        pipeline = PipelineFactory.create_postprocess_pipeline(
            audio=audio,
            ctx=TTSPipelineContext(
                adjust_config=AdjustConfig(
                    pitch=pitch,
                    speed_rate=speed,
                    volume_gain_db=volume_up,
                    normalize=enable_loudness_normalization,
                    headroom=headroom,
                ),
                enhancer_config=EnhancerConfig(
                    enabled=enable_de_noise or enable_enhance or False,
                    lambd=0.9 if enable_de_noise else 0.1,
                ),
            ),
        )
        sr, audio_data = pipeline.generate()
        audio_data = audio_utils.audio_to_int16(audio_data)
        return sr, audio_data

    gen_button.click(
        fn=gen_audio,
        inputs=[
            input_audio,
            speed_input,
            pitch_input,
            volume_up_input,
            enable_loudness_normalization,
            headroom_input,
            enable_enhance,
            enable_de_noise,
        ],
        outputs=output_audio,
    )
