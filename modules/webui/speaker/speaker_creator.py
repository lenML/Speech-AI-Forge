import tempfile

import gradio as gr
import torch

from modules.core.models.tts.ChatTtsModel import ChatTTSModel
from modules.utils.hf import spaces
from modules.utils.rng import np_rng
from modules.webui import webui_config
from modules.webui.webui_utils import tts_generate

names_list = [
    "Alice",
    "Bob",
    "Carol",
    "Carlos",
    "Charlie",
    "Chuck",
    "Chad",
    "Craig",
    "Dan",
    "Dave",
    "David",
    "Erin",
    "Eve",
    "Yves",
    "Faythe",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
    "Mallory",
    "Mallet",
    "Darth",
    "Michael",
    "Mike",
    "Niaj",
    "Olivia",
    "Oscar",
    "Peggy",
    "Pat",
    "Rupert",
    "Sybil",
    "Trent",
    "Ted",
    "Trudy",
    "Victor",
    "Vanna",
    "Walter",
    "Wendy",
]


@spaces.GPU(duration=120)
def create_spk_from_seed(
    seed: int, name: str, gender: str, desc: str, author: str, version: str
):
    spk = ChatTTSModel.create_speaker_from_seed(seed)
    spk.set_name(name=name)
    spk.set_desc(desc=desc)
    spk.set_gender(gender=gender)
    spk.set_author(author=author)
    spk.set_version(version=version)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".spkv1.json") as tmp_file:
        json_str = spk.to_json_str()
        tmp_file.write(json_str.encode("utf-8"))
        tmp_file_path = tmp_file.name

    return tmp_file_path


@spaces.GPU(duration=120)
async def test_spk_voice(
    seed: int,
    text: str,
    progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
):
    spk = ChatTTSModel.create_speaker_from_seed(seed)
    return await tts_generate(spk=spk, text=text, progress=progress)


def random_speaker():
    seed = np_rng()
    name = names_list[seed % len(names_list)]
    return seed, name


def speaker_creator_ui():
    def on_generate(seed, name, gender, desc, author, version):
        file_path = create_spk_from_seed(seed, name, gender, desc, author, version)
        return file_path

    def create_test_voice_card(seed_input):
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
                        current_seed = gr.Label(label="Current Seed", value=-1)
                        with gr.Column(scale=4):
                            output_audio = gr.Audio(label="Output Audio", format="mp3")

        test_voice_btn.click(
            fn=test_spk_voice,
            inputs=[seed_input, test_text],
            outputs=[output_audio],
        )
        test_voice_btn.click(
            fn=lambda x: x,
            inputs=[seed_input],
            outputs=[current_seed],
        )

    gr.Markdown("SPEAKER_CREATOR_GUIDE")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("ℹ️Speaker info")
                seed_input = gr.Number(label="Seed", value=2)
                name_input = gr.Textbox(
                    label="Name", placeholder="Enter speaker name", value="Bob"
                )
                gender_input = gr.Textbox(
                    label="Gender", placeholder="Enter gender", value="*"
                )
                author_input = gr.Textbox(
                    label="Author",
                    placeholder="Enter author",
                    value="",
                )
                version_input = gr.Textbox(
                    label="Version",
                    placeholder="Enter version",
                    value="",
                )
                desc_input = gr.Textbox(
                    label="Description",
                    placeholder="Enter description",
                )
                random_button = gr.Button("Random Speaker")
            with gr.Group():
                gr.Markdown("🔊Generate speaker.json")
                generate_button = gr.Button("Save .json file")
                output_file = gr.File(label="Save to File")
        with gr.Column(scale=5):
            create_test_voice_card(seed_input=seed_input)
            create_test_voice_card(seed_input=seed_input)
            create_test_voice_card(seed_input=seed_input)
            create_test_voice_card(seed_input=seed_input)

    random_button.click(
        random_speaker,
        outputs=[seed_input, name_input],
    )

    generate_button.click(
        fn=on_generate,
        inputs=[
            seed_input,
            name_input,
            gender_input,
            desc_input,
            author_input,
            version_input,
        ],
        outputs=[output_file],
    )
