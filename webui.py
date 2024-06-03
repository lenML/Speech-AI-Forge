import gradio as gr
import io

import torch

from modules.ssml import parse_ssml, synthesize_segment
from modules.generate_audio import generate_audio

from modules.speaker import speaker_mgr
from modules.data import styles_mgr

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")


def get_speakers():
    return speaker_mgr.list_speakers()


def get_styles():
    return styles_mgr.list_items()


async def synthesize_ssml(ssml: str):
    segments = parse_ssml(ssml)

    buffer = io.BytesIO()
    for segment in segments:
        audio_segment = synthesize_segment(segment=segment)
        audio_segment.export(buffer, format="wav")
    buffer.seek(0)

    return buffer.read()


def read_local_readme():
    with open("README.md", "r", encoding="utf-8") as file:
        return file.read()


# 演示示例文本
sample_texts = [
    {
        "text": "天气预报显示，今天会有小雨，请大家出门时记得带伞。降温的天气也提醒我们要适时添衣保暖。",
    },
    {
        "text": "公司的年度总结会议将在下周三举行，请各部门提前准备好相关材料，确保会议顺利进行。",
    },
    {
        "text": "今天的午餐菜单包括烤鸡、沙拉和蔬菜汤，大家可以根据自己的口味选择适合的菜品。",
    },
    {
        "text": "请注意，电梯将在下午两点进行例行维护，预计需要一个小时的时间，请大家在此期间使用楼梯。",
    },
    {
        "text": "图书馆新到了一批书籍，涵盖了文学、科学和历史等多个领域，欢迎大家前来借阅。",
    },
]

default_ssml = """
<speak version="0.1">
  <voice spk="Bob" seed="-1" style="narration-relaxed">
    这里是一个简单的 SSML 示例。 
  </voice>
</speak>
"""


def create_interface():
    speakers = get_speakers()
    speaker_names = [speaker.name for speaker in speakers]

    styles = [s.get("name") for s in get_styles()]

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Row():
                    with gr.Column(scale=1):
                        temperature_input = gr.Slider(
                            0.0, 1.0, value=0.3, label="Temperature"
                        )
                        top_p_input = gr.Slider(0.0, 1.0, value=0.7, label="Top P")
                        top_k_input = gr.Slider(1, 50, value=20, label="Top K")

                        with gr.Row():
                            spk_input_text = gr.Textbox(label="Speaker (Text or Seed)")
                            spk_input_dropdown = gr.Dropdown(
                                choices=speaker_names,
                                label="Choose Speaker",
                                interactive=True,
                            )
                            spk_input_dropdown.change(
                                fn=lambda x: x,
                                inputs=[spk_input_dropdown],
                                outputs=[spk_input_text],
                            )

                        with gr.Row():
                            style_input_text = gr.Textbox(label="Style (Text or Seed)")
                            style_input_dropdown = gr.Dropdown(
                                choices=styles, label="Choose Style", interactive=True
                            )
                            style_input_dropdown.change(
                                fn=lambda x: x,
                                inputs=[style_input_dropdown],
                                outputs=[style_input_text],
                            )
                        infer_seed_input = gr.Number(value=-1, label="Inference Seed")
                        use_decoder_input = gr.Checkbox(value=True, label="Use Decoder")
                        prompt1_input = gr.Textbox(label="Prompt 1")
                        prompt2_input = gr.Textbox(label="Prompt 2")
                        prefix_input = gr.Textbox(label="Prefix")
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="Text to Speech",
                            lines=10,
                            placeholder="输入文本或选择示例",
                        )
                        sample_dropdown = gr.Dropdown(
                            choices=[sample["text"] for sample in sample_texts],
                            label="选择示例",
                            value=None,
                            interactive=True,
                        )
                        sample_dropdown.change(
                            fn=lambda x: x,
                            inputs=[sample_dropdown],
                            outputs=[text_input],
                        )
                        tts_button = gr.Button("Generate Audio")
                        tts_output = gr.Audio(label="Generated Audio")

                def tts_generate(
                    text,
                    temperature,
                    top_p,
                    top_k,
                    spk,
                    infer_seed,
                    use_decoder,
                    prompt1,
                    prompt2,
                    prefix,
                    style,
                ):
                    try:
                        spk = int(spk)
                    except ValueError:
                        pass  # 保持原文本
                    return generate_audio(
                        text,
                        temperature,
                        top_p,
                        top_k,
                        spk,
                        infer_seed,
                        use_decoder,
                        prompt1,
                        prompt2,
                        prefix,
                    )

                tts_button.click(
                    tts_generate,
                    inputs=[
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
                        style_input_text,
                    ],
                    outputs=tts_output,
                )

            with gr.TabItem("SSML"):
                ssml_input = gr.Textbox(
                    label="SSML Input",
                    lines=10,
                    value=default_ssml,
                )
                ssml_button = gr.Button("Synthesize SSML")
                ssml_output = gr.Audio(label="Generated Audio")

                ssml_button.click(
                    synthesize_ssml,
                    inputs=[ssml_input],
                    outputs=ssml_output,
                )

            with gr.TabItem("README"):
                readme_content = read_local_readme()
                gr.Markdown(readme_content)

        gr.Markdown(
            "此项目基于 [ChatTTS-Forge](https://github.com/lenML/ChatTTS-Forge) "
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
