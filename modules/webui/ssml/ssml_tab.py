import xml.dom.minidom

import gradio as gr

from modules.core.models.zoo import ModelZoo
from modules.webui import webui_config
from modules.webui.webui_utils import (
    get_speaker_names,
    get_spk_emotions_from_name,
    synthesize_ssml,
)


def apply_spk_and_emotion(xml_content: str, spk: str, emotion: str):
    if emotion == "default":
        # default 就不用设置，节约字数方便review
        emotion = ""
    if ":" in spk:
        # 如果是显示名称的话，就取后面的
        spk = spk.split(":")[-1].strip()

    # 解析 xml ，并将其中的 voice node 上的 spk 和 emotion 设定为传入值，然后再转为 xml
    dom = xml.dom.minidom.parseString(xml_content)
    root = dom.documentElement
    voices = root.getElementsByTagName("voice")
    for voice in voices:
        voice.setAttribute("spk", spk)
        if emotion:
            voice.setAttribute("emotion", emotion)

    return dom.toxml()


def create_ssml_interface():
    speakers, speaker_names = get_speaker_names()
    speaker_names = ["*random"] + speaker_names

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("🗣️Speaker")
                # 批量切换 说话人 和 emotion
                spk_input_dropdown = gr.Dropdown(
                    choices=speaker_names,
                    interactive=True,
                    value="female : female2",
                    show_label=False,
                )
                spk_emotion = gr.Dropdown(
                    ["default"], value="default", label="Emotion", visible=False
                )
                reload_button = gr.Button(value="🔄", variant="secondary")

                # 将当前选择的音色应用于所有 voice
                apply_button = gr.Button(value="Apply All", variant="secondary")

                def reload_spks():
                    names = get_speaker_names()
                    return gr.Dropdown(choices=names)

                reload_button.click(
                    fn=reload_spks,
                    inputs=[],
                    outputs=[spk_input_dropdown],
                )

                def reload_emotions(show_name: str):
                    emotions = get_spk_emotions_from_name(
                        show_name.split(":")[-1].strip()
                        if ":" in show_name
                        else show_name
                    )
                    # 如果 emotion 大于1才显示 （因为总有一个默认值 default）
                    return gr.Dropdown(choices=emotions, visible=len(emotions) > 1)

                spk_input_dropdown.change(
                    fn=reload_emotions,
                    inputs=[spk_input_dropdown],
                    outputs=[spk_emotion],
                )

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

                enable_remove_silence = gr.Checkbox(
                    value=False, label="Enable Remove Silence"
                )
                # 默认 -42
                remove_silence_threshold_input = gr.Slider(
                    label="Remove Silence Threshold",
                    value=-42,
                    minimum=-60,
                    maximum=0,
                    step=1,
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
            enable_remove_silence,
            remove_silence_threshold_input,
        ],
        outputs=[ssml_output],
    )

    apply_button.click(
        fn=apply_spk_and_emotion,
        inputs=[ssml_input, spk_input_dropdown, spk_emotion],
        outputs=[ssml_input],
    )

    return ssml_input
