import json
import tempfile

import gradio as gr
import torch

from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.utils.hf import spaces
from modules.webui import webui_config
from modules.webui.webui_utils import SPK_FILE_EXTS, tts_generate

# TODO: 增加 png 编辑


@torch.inference_mode()
@spaces.GPU(duration=120)
def test_spk_voice(
    spk_file,
    text: str,
    progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
):
    if spk_file == "" or spk_file is None:
        return None
    spk = TTSSpeaker.from_file(spk_file)
    return tts_generate(spk=spk, text=text, progress=progress)


def speaker_editor_ui():
    def on_generate(spk_file, name: str, gender: str, desc: str):
        spk: TTSSpeaker = TTSSpeaker.from_file(spk_file)
        spk.set_name(name=name)
        spk.set_desc(desc=desc)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".spkv1.json"
        ) as tmp_file:
            json_str = spk.to_json_str()
            tmp_file.write(json_str.encode("utf-8"))
            tmp_file_path = tmp_file.name

        return tmp_file_path

    def create_test_voice_card(spk_file):
        with gr.Group():
            gr.Markdown("🎤Test voice")
            with gr.Row():
                test_voice_btn = gr.Button(
                    "Test Voice", variant="secondary", interactive=False
                )

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
            inputs=[spk_file, test_text],
            outputs=[output_audio],
        )

        return test_voice_btn

    has_file = gr.State(False)

    # TODO 也许需要写个说明？
    # gr.Markdown("SPEAKER_CREATOR_GUIDE")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("💼Speaker file")
                spk_file = gr.File(label="*.json file", file_types=SPK_FILE_EXTS)

            with gr.Group():
                gr.Markdown("ℹ️Speaker info")
                name_input = gr.Textbox(
                    label="Name",
                    placeholder="Enter speaker name",
                    value="*",
                    interactive=False,
                )
                gender_input = gr.Textbox(
                    label="Gender",
                    placeholder="Enter gender",
                    value="*",
                    interactive=False,
                )
                desc_input = gr.Textbox(
                    label="Description",
                    placeholder="Enter description",
                    value="*",
                    interactive=False,
                )
            with gr.Group():
                gr.Markdown("🔊Generate speaker.json")
                generate_button = gr.Button("Save .json file", interactive=False)
                output_file = gr.File(label="Save to File")
        with gr.Column(scale=5):
            btn1 = create_test_voice_card(spk_file=spk_file)
            btn2 = create_test_voice_card(spk_file=spk_file)
            btn3 = create_test_voice_card(spk_file=spk_file)
            btn4 = create_test_voice_card(spk_file=spk_file)

    generate_button.click(
        fn=on_generate,
        inputs=[spk_file, name_input, gender_input, desc_input],
        outputs=[output_file],
    )

    # TODO: 这些全部都需要改
    def spk_file_change(spk_file):
        empty = spk_file is None or spk_file == ""
        if empty:
            return [
                gr.Textbox(value="*", interactive=False),
                gr.Textbox(value="*", interactive=False),
                gr.Textbox(value="*", interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
            ]
        spk: TTSSpeaker = TTSSpeaker.from_file(spk_file)
        return [
            gr.Textbox(value=spk.name, interactive=True),
            gr.Textbox(value=spk.gender, interactive=True),
            gr.Textbox(value=spk.desc, interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
        ]

    spk_file.change(
        fn=spk_file_change,
        inputs=[spk_file],
        outputs=[
            name_input,
            gender_input,
            desc_input,
            generate_button,
            btn1,
            btn2,
            btn3,
            btn4,
        ],
    )
