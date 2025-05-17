from tempfile import NamedTemporaryFile

import gradio as gr
import numpy as np
from whisper.tokenizer import LANGUAGES

from modules.core.handler.datacls.stt_model import STTConfig
from modules.core.handler.STTHandler import STTHandler
from modules.core.models.zoo.ModelZoo import model_zoo
from modules.webui import webui_config


def stereo_to_mono(audio_input: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    sample_rate, audio_data = audio_input

    if audio_data.ndim == 2 and audio_data.shape[1] == 2:
        mono_audio = np.mean(audio_data, axis=1)
    else:
        mono_audio = audio_data

    return (sample_rate, mono_audio)


def create_whisper_asr_tab():
    """
    上传音频，然后转录

    支持转为 txt/srt/vtt/tsv/json 格式
    """

    with gr.Row():
        with gr.Column():
            with gr.Group():
                gr.Markdown("Params")
                with gr.Row():
                    format_select = gr.Dropdown(
                        choices=["txt", "srt", "vtt", "tsv", "lrc", "json"],
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

            with gr.Group():
                model_ids = model_zoo.get_stt_model_ids()
                gr.Markdown("Input")
                audio_input = gr.Audio(label="Audio")
                model_selected = gr.Dropdown(
                    # choices=["whisper.large", "whisper.turbo", "sensevoice"],
                    choices=model_ids,
                    label="Model",
                    value=model_ids[0] if len(model_ids) > 0 else "whisper.large",
                )
                submit_button = gr.Button("transcribe", variant="primary")

        with gr.Column():
            # 参考文本，可以留空
            input_transcript_content = gr.Textbox(
                label="Reference Transcript",
                placeholder="Reference transcript (optional)",
            )
            with gr.Group():
                gr.Markdown("Output")
                output = gr.Textbox(label="Transcript")
                output_file = gr.File(label="Download")

    async def submit(
        audio_input,
        format_select: str,
        prompt_input: str,
        prefix_input: str,
        language_select: str,
        temperature_input: float,
        best_of_input: int,
        beams_input: int,
        max_words_per_line_input: int,
        max_line_width_input: int,
        max_line_count_input: int,
        refrence_transcript: str,
        model_id: str,
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
                mid=model_id,
                # mid="whisper.turbo",
                # mid="sensevoice",
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
                refrence_transcript=refrence_transcript,
            ),
        )
        audio = await handler.enqueue()

        with NamedTemporaryFile(delete=False, suffix=f".{format_select}") as tmp_file:
            tmp_file.write(audio.text.encode("utf-8"))
            return audio.text, tmp_file.name

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
            input_transcript_content,
            model_selected,
        ],
        outputs=[output, output_file],
    )
