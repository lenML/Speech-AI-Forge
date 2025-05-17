import gradio as gr

from modules.core.models.zoo.ModelZoo import model_zoo
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.utils.hf import spaces
from modules.webui import webui_config
from modules.webui.webui_utils import SPK_FILE_EXTS, get_spk_emotions, tts_generate


@spaces.GPU(duration=120)
async def test_spk_voice(
    spk_file,
    text: str,
    model: str,
    temperature: float,
    top_p: int,
    top_k: int,
    spk_emotion: str,
    progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
):
    if spk_file == "" or spk_file is None:
        return None
    spk = TTSSpeaker.from_file(spk_file)
    return await tts_generate(
        spk=spk,
        text=text,
        model_id=model,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        progress=progress,
        eos="。",
        spk_emotion=spk_emotion,
    )


def create_test_voice_card(
    spk_file,
    model_id_select,
    temperature_input,
    top_p_input,
    top_k_input,
    spk_emotion,
):
    with gr.Group():
        gr.Markdown("🎤Test voice")
        with gr.Row():
            test_voice_btn = gr.Button("Test Voice", variant="secondary")

            with gr.Column(scale=4):
                test_text = gr.Textbox(
                    label="Test Text",
                    placeholder="Please input test text",
                    value=webui_config.localization.DEFAULT_SPEAKER_TEST_TEXT,
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        output_audio = gr.Audio(label="Output Audio", format="mp3")

    test_voice_btn.click(
        fn=test_spk_voice,
        inputs=[
            spk_file,
            test_text,
            model_id_select,
            temperature_input,
            top_p_input,
            top_k_input,
            spk_emotion,
        ],
        outputs=[output_audio],
    )

    return test_voice_btn


# 读取 spk 并 更新 info
def load_spk_file_to_info(spk_file):
    if spk_file == "" or spk_file is None:
        return "None"
    spk = TTSSpeaker.from_file(spk_file)
    name = spk.name
    gender = spk.gender
    desc = spk.desc
    version = spk.version
    author = spk.author

    rows = [
        f"💌 Speaker info",
        f"- Name: {name}",
        f"- Gender: {gender}",
        f"- Author: {author}",
        f"- Version: {version}",
        f"- Description: {desc}",
    ]
    return "\n".join(rows)


def speaker_editor_ui():

    # TODO 也许需要写个说明？
    # gr.Markdown("SPEAKER_CREATOR_GUIDE")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("💼Speaker file")
                spk_file = gr.File(label="*.json file", file_types=SPK_FILE_EXTS)
                spk_info = gr.Markdown("None")
                spk_emotion = gr.Dropdown(["default"], value="default", label="Emotion")

                spk_file.change(
                    fn=load_spk_file_to_info,
                    inputs=spk_file,
                    outputs=spk_info,
                )
                spk_file.change(
                    fn=lambda file: gr.Dropdown(choices=get_spk_emotions(file)),
                    inputs=[spk_file],
                    outputs=[spk_emotion],
                )

            with gr.Group():
                # 模型选择
                model_id_select = gr.Dropdown(
                    label="Model",
                    choices=model_zoo.get_tts_model_ids(),
                    value="chat-tts",
                )

            with gr.Group():
                # 采样率
                gr.Markdown("🎛️Sampling")
                temperature_input = gr.Slider(
                    0.01, 2.0, value=0.3, step=0.01, label="Temperature"
                )
                top_p_input = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Top P")
                top_k_input = gr.Slider(1, 50, value=20, step=1, label="Top K")

        with gr.Column(scale=5):
            kwargs = dict(
                spk_file=spk_file,
                model_id_select=model_id_select,
                temperature_input=temperature_input,
                top_p_input=top_p_input,
                top_k_input=top_k_input,
                spk_emotion=spk_emotion,
            )

            btn1 = create_test_voice_card(**kwargs)
            btn2 = create_test_voice_card(**kwargs)
            btn3 = create_test_voice_card(**kwargs)
            btn4 = create_test_voice_card(**kwargs)
