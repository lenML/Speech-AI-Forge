import gradio as gr
import numpy as np

from modules.core.handler.STTHandler import STTHandler
from modules.core.handler.datacls.stt_model import STTConfig

from whisper.tokenizer import LANGUAGES

from modules.webui import webui_config


def stereo_to_mono(audio_input: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    sample_rate, audio_data = audio_input

    if audio_data.ndim == 2 and audio_data.shape[1] == 2:
        mono_audio = np.mean(audio_data, axis=1)
    else:
        mono_audio = audio_data

    return (sample_rate, mono_audio)


def create_asr_tab():
    """
    上传音频，然后转录

    支持转为 txt/srt/vtt/tsv/json 格式
    """

    with gr.Row():
        with gr.Column():
            with gr.Group():
                gr.Markdown("Input")
                audio_input = gr.Audio(label="Audio")
                submit_button = gr.Button("transcribe", variant="primary")

            with gr.Group():
                gr.Markdown("Params")
                with gr.Row():
                    format_select = gr.Dropdown(
                        choices=["txt", "srt", "vtt", "tsv", "json"],
                        label="Format",
                        value="srt",
                    )
                    language_select = gr.Dropdown(
                        choices=["auto"] + list(LANGUAGES.keys()),
                        label="language",
                        value="auto",
                    )
                with gr.Row():
                    prompt_input = gr.Textbox(label="Prompt")
                    prefix_input = gr.Textbox(label="Prefix")

                with gr.Row():
                    temperature_input = gr.Slider(
                        label="temperature",
                        # 0 为动态温度值
                        minimum=0,
                        maximum=1.0,
                        step=0.1,
                        value=0,
                    )
                    best_of_input = gr.Slider(
                        label="best of",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                    )
                    beams_input = gr.Slider(
                        label="Beam size",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                    )

                with gr.Row():
                    max_words_per_line_input = gr.Slider(
                        label="Max words per line",
                        minimum=-1,
                        maximum=50,
                        step=1,
                        value=-1,
                    )
                    max_line_width_input = gr.Slider(
                        label="Max line width",
                        minimum=-1,
                        maximum=50,
                        step=1,
                        value=-1,
                    )
                    max_line_count_input = gr.Slider(
                        label="Max line count",
                        minimum=-1,
                        maximum=10,
                        step=1,
                        value=-1,
                    )

        with gr.Column():
            output = gr.Textbox(label="Transcript")

    def submit(
        audio_input,
        format_select,
        prompt_input,
        prefix_input,
        language_select,
        temperature_input,
        best_of_input,
        beams_input,
        max_words_per_line_input,
        max_line_width_input,
        max_line_count_input,
        progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
    ):
        if max_words_per_line_input == -1:
            max_words_per_line_input = None
        if max_line_width_input == -1:
            max_line_width_input = None
        if max_line_count_input == -1:
            max_line_count_input = None

        audio_input = stereo_to_mono(audio_input=audio_input)
        handler = STTHandler(
            input_audio=audio_input,
            stt_config=STTConfig(
                mid="whisper",
                format=format_select,
                prompt=prompt_input or None,
                prefix=prefix_input or None,
                language=None if language_select == "auto" else language_select,
                temperature=temperature_input or 0,
                best_of=best_of_input or 5,
                beams=beams_input or 5,
                max_line_width=max_line_width_input,
                max_line_count=max_line_count_input,
                max_words_per_line=max_words_per_line_input,
            ),
        )
        audio = handler.enqueue()
        return audio.text

    submit_button.click(
        fn=submit,
        inputs=[
            audio_input,
            format_select,
            prompt_input,
            prefix_input,
            language_select,
            temperature_input,
            best_of_input,
            beams_input,
            max_words_per_line_input,
            max_line_width_input,
            max_line_count_input,
        ],
        outputs=[output],
    )
