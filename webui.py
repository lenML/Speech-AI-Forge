import gradio as gr
import io

import torch

from modules.ssml import parse_ssml, synthesize_segments, combine_audio_segments
from modules.generate_audio import generate_audio

from modules.speaker import speaker_mgr
from modules.data import styles_mgr

from modules.api.utils import calc_spk_style

from modules.utils.normalization import text_normalize
from modules import refiner

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")


def get_speakers():
    return speaker_mgr.list_speakers()


def get_styles():
    return styles_mgr.list_items()


@torch.inference_mode()
def synthesize_ssml(ssml: str):
    segments = parse_ssml(ssml)

    audio_segments = synthesize_segments(segments)
    combined_audio = combine_audio_segments(audio_segments)

    buffer = io.BytesIO()
    combined_audio.export(buffer, format="wav")

    return buffer.read()


@torch.inference_mode()
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
    params = calc_spk_style(spk=spk, style=style)

    spk = params.get("spk", spk)
    infer_seed = infer_seed or params.get("seed", infer_seed)
    temperature = temperature or params.get("temperature", temperature)
    prefix = prefix or params.get("prefix", prefix)
    prompt1 = prompt1 or params.get("prompt1", "")
    prompt2 = prompt2 or params.get("prompt2", "")

    sample_rate, audio_data = generate_audio(
        text=text_normalize(text),
        temperature=temperature,
        top_P=top_p,
        top_K=top_k,
        spk=spk,
        infer_seed=infer_seed,
        use_decoder=use_decoder,
        prompt1=prompt1,
        prompt2=prompt2,
        prefix=prefix,
    )

    return sample_rate, audio_data


@torch.inference_mode()
def refine_text(text: str):
    return refiner.refine_text(text)


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
    {
        "text": "电影中梁朝伟扮演的陈永仁的编号27149",
    },
    {
        "text": "这块黄金重达324.75克",
    },
    {
        "text": "我们班的最高总分为583分",
    },
    {
        "text": "12~23",
    },
    {
        "text": "-1.5~2",
    },
    {
        "text": "她出生于86年8月18日，她弟弟出生于1995年3月1日",
    },
    {
        "text": "等会请在12:05请通知我",
    },
    {
        "text": "今天的最低气温达到-10°C",
    },
    {
        "text": "现场有7/12的观众投出了赞成票",
    },
    {
        "text": "明天有62％的概率降雨",
    },
    {
        "text": "随便来几个价格12块5，34.5元，20.1万",
    },
    {
        "text": "这是固话0421-33441122",
    },
    {
        "text": "这是手机+86 18544139121",
    },
]

ssml_example1 = """
<speak version="0.1">
    <voice spk="Bob" style="narration-relaxed">
        下面是一个 ChatTTS 用于合成多角色多情感的有声书示例
    </voice>
    <voice spk="Bob" style="narration-relaxed">
        黛玉冷笑道：
    </voice>
    <voice spk="female2" style="angry">
        我说呢 [uv_break] ，亏了绊住，不然，早就飞起来了。
    </voice>
    <voice spk="Bob" style="narration-relaxed">
        宝玉道：
    </voice>
    <voice spk="Alice" style="unfriendly">
        “只许和你玩 [uv_break] ，替你解闷。不过偶然到他那里，就说这些闲话。”
    </voice>
    <voice spk="female2" style="angry">
        “好没意思的话！[uv_break] 去不去，关我什么事儿？ 又没叫你替我解闷儿 [uv_break]，还许你不理我呢”
    </voice>
    <voice spk="Bob" style="narration-relaxed">
        说着，便赌气回房去了。
    </voice>
</speak>
"""
ssml_example2 = """
<speak version="0.1">
    <voice spk="Bob" style="narration-relaxed">
        使用 prosody 控制生成文本的语速语调和音量，示例如下

        <prosody>
            无任何限制将会继承父级voice配置进行生成
        </prosody>
        <prosody rate="1.5">
            设置 rate 大于1表示加速，小于1为减速
        </prosody>
        <prosody pitch="6">
            设置 pitch 调整音调，设置为6表示提高6个半音
        </prosody>
        <prosody volume="2">
            设置 volume 调整音量，设置为2表示提高2个分贝
        </prosody>

        在 voice 中无prosody包裹的文本即为默认生成状态下的语音
    </voice>
</speak>
"""
ssml_example3 = """
<speak version="0.1">
    <voice spk="Bob" style="narration-relaxed">
        使用 break 标签将会简单的
        
        <break time="500" />

        插入一段空白到生成结果中 
    </voice>
</speak>
"""

ssml_example4 = """
<speak version="0.1">
    <voice spk="Bob" style="excited">
        temperature for sampling (may be overridden by style or speaker)
        <break time="500" />
        温度值用于采样，这个值有可能被 style 或者 speaker 覆盖 
        <break time="500" />
        temperature for sampling ，这个值有可能被 style 或者 speaker 覆盖 
        <break time="500" />
        温度值用于采样，(may be overridden by style or speaker)
    </voice>
</speak>
"""

default_ssml = """
<speak version="0.1">
  <voice spk="Bob" seed="-1" style="narration-relaxed">
    这里是一个简单的 SSML 示例。 
  </voice>
</speak>
"""


def create_interface():
    speakers = get_speakers()
    speaker_names = ["*random"] + [speaker.name for speaker in speakers]

    styles = ["*auto"] + [s.get("name") for s in get_styles()]

    js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

    with gr.Blocks(js=js_func) as demo:
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
                            spk_input_text = gr.Textbox(
                                label="Speaker (Text or Seed)", value="female2"
                            )
                            spk_input_dropdown = gr.Dropdown(
                                choices=speaker_names,
                                label="Choose Speaker",
                                interactive=True,
                                value="female2",
                            )
                            spk_input_dropdown.change(
                                fn=lambda x: x.startswith("*") and "-1" or x,
                                inputs=[spk_input_dropdown],
                                outputs=[spk_input_text],
                            )

                        with gr.Row():
                            style_input_text = gr.Textbox(
                                label="Style (Text or Seed)", value="-1"
                            )
                            style_input_dropdown = gr.Dropdown(
                                choices=styles,
                                label="Choose Style",
                                interactive=True,
                                value="*auto",
                            )
                            style_input_dropdown.change(
                                fn=lambda x: x.startswith("*") and "-1" or x,
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
                        with gr.Row():
                            refine_button = gr.Button("✍️Refine Text")
                            tts_button = gr.Button("🔊Generate Audio")

                        tts_output = gr.Audio(label="Generated Audio")

                refine_button.click(
                    refine_text,
                    inputs=[text_input],
                    outputs=[text_input],
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
                ssml_button = gr.Button("🔊Synthesize SSML")
                ssml_output = gr.Audio(label="Generated Audio")

                ssml_button.click(
                    synthesize_ssml,
                    inputs=[ssml_input],
                    outputs=ssml_output,
                )

                examples = [
                    ssml_example1,
                    ssml_example2,
                    ssml_example3,
                    ssml_example4,
                ]

                gr.Examples(
                    examples=examples,
                    inputs=[ssml_input],
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
    demo.queue().launch(share=False)
