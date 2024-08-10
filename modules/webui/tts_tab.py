import gradio as gr
import torch

from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.webui import webui_config
from modules.webui.webui_utils import (
    SPK_FILE_EXTS,
    get_speakers,
    get_styles,
    load_spk_info,
    refine_text,
    tts_generate,
)


def tts_generate_with_history(
    audio_history: list,
    progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
    *args,
    **kwargs,
):
    audio = tts_generate(*args, **kwargs)

    mp3_1 = audio
    mp3_2 = audio_history[-1] if len(audio_history) > 0 else None
    mp3_3 = audio_history[-2] if len(audio_history) > 1 else None

    return mp3_1, mp3_2, mp3_3, [mp3_3, mp3_2, mp3_1]


def create_tts_interface():
    speakers = get_speakers()

    def get_speaker_show_name(spk: TTSSpeaker):
        if spk.gender == "*" or spk.gender == "":
            return spk.name
        return f"{spk.gender} : {spk.name}"

    speaker_names = ["*random"] + [
        get_speaker_show_name(speaker) for speaker in speakers
    ]
    speaker_names.sort(key=lambda x: x.startswith("*") and "-1" or x)

    styles = ["*auto"] + [s.get("name") for s in get_styles()]

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Group():
                    gr.Markdown("🗣️Speaker")
                    with gr.Tabs():
                        with gr.Tab(label="Pick"):
                            spk_input_text = gr.Textbox(
                                label="Speaker (Text or Seed)",
                                value="female2",
                                show_label=False,
                            )
                            spk_input_dropdown = gr.Dropdown(
                                choices=speaker_names,
                                # label="Choose Speaker",
                                interactive=True,
                                value="female : female2",
                                show_label=False,
                            )
                            spk_rand_button = gr.Button(
                                value="🎲",
                                # tooltip="Random Seed",
                                variant="secondary",
                            )
                            spk_input_dropdown.change(
                                fn=lambda x: x.startswith("*")
                                and "-1"
                                or x.split(":")[-1].strip(),
                                inputs=[spk_input_dropdown],
                                outputs=[spk_input_text],
                            )
                            spk_rand_button.click(
                                lambda x: str(torch.randint(0, 2**32 - 1, (1,)).item()),
                                inputs=[spk_input_text],
                                outputs=[spk_input_text],
                            )

                        with gr.Tab(label="Upload"):
                            spk_file_upload = gr.File(
                                label="Speaker (Upload)",
                                file_types=SPK_FILE_EXTS,
                            )

                            gr.Markdown("📝Speaker info")
                            infos = gr.Markdown("empty", elem_classes=["no-translate"])

                            spk_file_upload.change(
                                fn=load_spk_info,
                                inputs=[spk_file_upload],
                                outputs=[infos],
                            )

            with gr.Row():
                with gr.Group():
                    gr.Markdown("🎭Style")
                    gr.Markdown("TTS_STYLE_GUIDE")
                    style_input_dropdown = gr.Dropdown(
                        choices=styles,
                        # label="Choose Style",
                        interactive=True,
                        show_label=False,
                        value="*auto",
                    )

            with gr.Group():
                gr.Markdown("🎛️Sampling")
                temperature_input = gr.Slider(
                    0.01, 2.0, value=0.3, step=0.01, label="Temperature"
                )
                top_p_input = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Top P")
                top_k_input = gr.Slider(1, 50, value=20, step=1, label="Top K")
                batch_size_input = gr.Slider(
                    1,
                    webui_config.max_batch_size,
                    value=4,
                    step=1,
                    label="Batch Size",
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
            # 感觉这个没必要设置...
            use_decoder_input = gr.Checkbox(
                value=True, label="Use Decoder", visible=False
            )
        with gr.Column(scale=4):
            with gr.Group():
                input_title = gr.Markdown(
                    "📝Text Input",
                    elem_id="input-title",
                )
                gr.Markdown(f"TTS_TEXT_GUIDE")
                text_input = gr.Textbox(
                    show_label=False,
                    label="Text to Speech",
                    lines=10,
                    placeholder="输入文本或选择示例",
                    elem_id="text-input",
                    value=webui_config.localization.DEFAULT_TTS_TEXT,
                )
                # TODO 字数统计，其实实现很好写，但是就是会触发loading...并且还要和后端交互...
                # text_input.change(
                #     fn=lambda x: (
                #         f"📝Text Input ({len(x)} char)"
                #         if x
                #         else (
                #             "📝Text Input (0 char)"
                #             if not x
                #             else "📝Text Input (0 char)"
                #         )
                #     ),
                #     inputs=[text_input],
                #     outputs=[input_title],
                # )
                with gr.Row():
                    contorl_tokens = [
                        "[laugh]",
                        "[uv_break]",
                        "[v_break]",
                        "[lbreak]",
                    ]

                    for tk in contorl_tokens:
                        t_btn = gr.Button(tk)
                        t_btn.click(
                            lambda text, tk=tk: text + " " + tk,
                            inputs=[text_input],
                            outputs=[text_input],
                        )

            with gr.Group():
                gr.Markdown("🎄Examples")
                sample_dropdown = gr.Dropdown(
                    choices=[
                        sample["text"]
                        for sample in webui_config.localization.tts_examples
                    ],
                    show_label=False,
                    value=None,
                    interactive=True,
                )
                sample_dropdown.change(
                    fn=lambda x: x,
                    inputs=[sample_dropdown],
                    outputs=[text_input],
                )

            with gr.Group():
                gr.Markdown("🎨Output")

                audio_history = gr.State([])
                tts_output1 = gr.Audio(label="Generated Audio", format="mp3")
                tts_output2 = gr.Audio(label="History -1", format="mp3")
                tts_output3 = gr.Audio(label="History -2", format="mp3")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("🎶Refiner")
                # refine_prompt_input = gr.Textbox(
                #     label="Refine Prompt",
                #     value="[oral_2][laugh_0][break_6]",
                # )
                # oral 0 - 9
                # speed 0 - 9
                # break 0 - 7
                # laugh 0 - 2
                # -1 表示不使用这个标签
                rf_oral_input = gr.Slider(
                    label="Oral",
                    value=2,
                    minimum=-1,
                    maximum=9,
                    step=1,
                )
                rf_speed_input = gr.Slider(
                    label="Speed",
                    value=2,
                    minimum=-1,
                    maximum=9,
                    step=1,
                )
                rf_break_input = gr.Slider(
                    label="Break",
                    value=2,
                    minimum=-1,
                    maximum=7,
                    step=1,
                )
                rf_laugh_input = gr.Slider(
                    label="Laugh",
                    value=0,
                    minimum=-1,
                    maximum=2,
                    step=1,
                )

                refine_button = gr.Button("✍️Refine Text")

            # 由于使用不是很方便，所以列为实验性功能
            with gr.Group(visible=webui_config.experimental):
                gr.Markdown("🔧Prompt engineering")
                prompt1_input = gr.Textbox(label="Prompt 1")
                prompt2_input = gr.Textbox(label="Prompt 2")
                prefix_input = gr.Textbox(label="Prefix")

                prompt_audio = gr.File(
                    label="prompt_audio", visible=webui_config.experimental
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
                gr.Markdown("💃Inference Seed")
                infer_seed_input = gr.Number(
                    value=42,
                    label="Inference Seed",
                    show_label=False,
                    minimum=-1,
                    maximum=2**32 - 1,
                )
                infer_seed_rand_button = gr.Button(
                    value="🎲",
                    # tooltip="Random Seed",
                    variant="secondary",
                )

            infer_seed_rand_button.click(
                lambda x: int(torch.randint(0, 2**32 - 1, (1,)).item()),
                inputs=[infer_seed_input],
                outputs=[infer_seed_input],
            )

            with gr.Group():
                gr.Markdown("🔊Generate")
                disable_normalize_input = gr.Checkbox(
                    value=False,
                    label="Disable Normalize",
                    # 不需要了
                    visible=False,
                )

                with gr.Group():
                    # gr.Markdown("💪🏼Enhance")
                    enable_enhance = gr.Checkbox(value=True, label="Enable Enhance")
                    enable_de_noise = gr.Checkbox(value=False, label="Enable De-noise")
                tts_button = gr.Button(
                    "🔊Generate Audio",
                    variant="primary",
                    elem_classes="big-button",
                )

    refine_button.click(
        refine_text,
        inputs=[
            text_input,
            rf_oral_input,
            rf_speed_input,
            rf_break_input,
            rf_laugh_input,
        ],
        outputs=[text_input],
    )

    tts_button.click(
        tts_generate_with_history,
        inputs=[
            audio_history,
            text_input,
            temperature_input,
            top_p_input,
            top_k_input,
            spk_input_text,
            infer_seed_input,
            use_decoder_input,
            prompt1_input,
            prompt2_input,
            prefix_input,
            style_input_dropdown,
            disable_normalize_input,
            batch_size_input,
            enable_enhance,
            enable_de_noise,
            spk_file_upload,
            spliter_thr_input,
            eos_input,
            pitch_input,
            speed_input,
            volume_up_input,
            enable_loudness_normalization,
            headroom_input,
        ],
        outputs=[tts_output1, tts_output2, tts_output3, audio_history],
    )
