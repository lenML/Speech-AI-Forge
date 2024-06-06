try:
    import spaces
except:

    class NoneSpaces:
        def __init__(self):
            pass

        def GPU(self, fn):
            return fn

    spaces = NoneSpaces()

import os
import logging

import numpy as np

from modules.devices import devices
from modules.synthesize_audio import synthesize_audio
from modules.utils.cache import conditional_cache

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


import gradio as gr

import torch

from modules.ssml import parse_ssml
from modules.SynthesizeSegments import SynthesizeSegments, combine_audio_segments

from modules.speaker import speaker_mgr
from modules.data import styles_mgr

from modules.api.utils import calc_spk_style
import modules.generate_audio as generate

from modules.normalization import text_normalize
from modules import refiner, config

from modules.utils import env, audio
from modules.SentenceSplitter import SentenceSplitter

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")

webui_config = {
    "tts_max": 1000,
    "ssml_max": 5000,
    "spliter_threshold": 100,
    "max_batch_size": 8,
}


def get_speakers():
    return speaker_mgr.list_speakers()


def get_styles():
    return styles_mgr.list_items()


def segments_length_limit(segments, total_max: int):
    ret_segments = []
    total_len = 0
    for seg in segments:
        total_len += len(seg["text"])
        if total_len > total_max:
            break
        ret_segments.append(seg)
    return ret_segments


@torch.inference_mode()
@spaces.GPU
def synthesize_ssml(ssml: str, batch_size=4):
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 8

    ssml = ssml.strip()

    if ssml == "":
        return None

    segments = parse_ssml(ssml)
    max_len = webui_config["ssml_max"]
    segments = segments_length_limit(segments, max_len)

    if len(segments) == 0:
        return None

    synthesize = SynthesizeSegments(batch_size=batch_size)
    audio_segments = synthesize.synthesize_segments(segments)
    combined_audio = combine_audio_segments(audio_segments)

    return audio.pydub_to_np(combined_audio)


@torch.inference_mode()
@spaces.GPU
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
    disable_normalize=False,
    batch_size=4,
):
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 4

    max_len = webui_config["tts_max"]
    text = text.strip()[0:max_len]

    if text == "":
        return None

    if style == "*auto":
        style = None

    if isinstance(top_k, float):
        top_k = int(top_k)

    params = calc_spk_style(spk=spk, style=style)
    spk = params.get("spk", spk)

    infer_seed = infer_seed or params.get("seed", infer_seed)
    temperature = temperature or params.get("temperature", temperature)
    prefix = prefix or params.get("prefix", prefix)
    prompt1 = prompt1 or params.get("prompt1", "")
    prompt2 = prompt2 or params.get("prompt2", "")

    infer_seed = np.clip(infer_seed, -1, 2**32 - 1)
    infer_seed = int(infer_seed)

    if not disable_normalize:
        text = text_normalize(text)

    sample_rate, audio_data = synthesize_audio(
        text=text,
        temperature=temperature,
        top_P=top_p,
        top_K=top_k,
        spk=spk,
        infer_seed=infer_seed,
        use_decoder=use_decoder,
        prompt1=prompt1,
        prompt2=prompt2,
        prefix=prefix,
        batch_size=batch_size,
    )

    audio_data = audio.audio_to_int16(audio_data)
    return sample_rate, audio_data


@torch.inference_mode()
@spaces.GPU
def refine_text(text: str, prompt: str):
    text = text_normalize(text)
    return refiner.refine_text(text, prompt=prompt)


def read_local_readme():
    with open("README.md", "r", encoding="utf-8") as file:
        content = file.read()
        content = content[content.index("# 🗣️ ChatTTS-Forge") :]
        return content


# 演示示例文本
sample_texts = [
    {
        "text": "大🍌，一条大🍌，嘿，你的感觉真的很奇妙  [lbreak]",
    },
    {
        "text": "天气预报显示，今天会有小雨，请大家出门时记得带伞。降温的天气也提醒我们要适时添衣保暖 [lbreak]",
    },
    {
        "text": "公司的年度总结会议将在下周三举行，请各部门提前准备好相关材料，确保会议顺利进行 [lbreak]",
    },
    {
        "text": "今天的午餐菜单包括烤鸡、沙拉和蔬菜汤，大家可以根据自己的口味选择适合的菜品 [lbreak]",
    },
    {
        "text": "请注意，电梯将在下午两点进行例行维护，预计需要一个小时的时间，请大家在此期间使用楼梯 [lbreak]",
    },
    {
        "text": "图书馆新到了一批书籍，涵盖了文学、科学和历史等多个领域，欢迎大家前来借阅 [lbreak]",
    },
    {
        "text": "电影中梁朝伟扮演的陈永仁的编号27149 [lbreak]",
    },
    {
        "text": "这块黄金重达324.75克 [lbreak]",
    },
    {
        "text": "我们班的最高总分为583分 [lbreak]",
    },
    {
        "text": "12~23 [lbreak]",
    },
    {
        "text": "-1.5~2 [lbreak]",
    },
    {
        "text": "她出生于86年8月18日，她弟弟出生于1995年3月1日 [lbreak]",
    },
    {
        "text": "等会请在12:05请通知我 [lbreak]",
    },
    {
        "text": "今天的最低气温达到-10°C [lbreak]",
    },
    {
        "text": "现场有7/12的观众投出了赞成票 [lbreak]",
    },
    {
        "text": "明天有62％的概率降雨 [lbreak]",
    },
    {
        "text": "随便来几个价格12块5，34.5元，20.1万 [lbreak]",
    },
    {
        "text": "这是固话0421-33441122 [lbreak]",
    },
    {
        "text": "这是手机+86 18544139121 [lbreak]",
    },
]

ssml_example1 = """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="narration-relaxed">
        下面是一个 ChatTTS 用于合成多角色多情感的有声书示例[lbreak]
    </voice>
    <voice spk="Bob" seed="42" style="narration-relaxed">
        黛玉冷笑道：[lbreak]
    </voice>
    <voice spk="female2" seed="42" style="angry">
        我说呢 [uv_break] ，亏了绊住，不然，早就飞起来了[lbreak]
    </voice>
    <voice spk="Bob" seed="42" style="narration-relaxed">
        宝玉道：[lbreak]
    </voice>
    <voice spk="Alice" seed="42" style="unfriendly">
        “只许和你玩 [uv_break] ，替你解闷。不过偶然到他那里，就说这些闲话。”[lbreak]
    </voice>
    <voice spk="female2" seed="42" style="angry">
        “好没意思的话！[uv_break] 去不去，关我什么事儿？ 又没叫你替我解闷儿 [uv_break]，还许你不理我呢” [lbreak]
    </voice>
    <voice spk="Bob" seed="42" style="narration-relaxed">
        说着，便赌气回房去了 [lbreak]
    </voice>
</speak>
"""
ssml_example2 = """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="narration-relaxed">
        使用 prosody 控制生成文本的语速语调和音量，示例如下 [lbreak]

        <prosody>
            无任何限制将会继承父级voice配置进行生成 [lbreak]
        </prosody>
        <prosody rate="1.5">
            设置 rate 大于1表示加速，小于1为减速 [lbreak]
        </prosody>
        <prosody pitch="6">
            设置 pitch 调整音调，设置为6表示提高6个半音 [lbreak]
        </prosody>
        <prosody volume="2">
            设置 volume 调整音量，设置为2表示提高2个分贝 [lbreak]
        </prosody>

        在 voice 中无prosody包裹的文本即为默认生成状态下的语音 [lbreak]
    </voice>
</speak>
"""
ssml_example3 = """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="narration-relaxed">
        使用 break 标签将会简单的 [lbreak]
        
        <break time="500" />

        插入一段空白到生成结果中 [lbreak]
    </voice>
</speak>
"""

ssml_example4 = """
<speak version="0.1">
    <voice spk="Bob" seed="42" style="excited">
        temperature for sampling (may be overridden by style or speaker) [lbreak]
        <break time="500" />
        温度值用于采样，这个值有可能被 style 或者 speaker 覆盖  [lbreak]
        <break time="500" />
        temperature for sampling ，这个值有可能被 style 或者 speaker 覆盖  [lbreak]
        <break time="500" />
        温度值用于采样，(may be overridden by style or speaker) [lbreak]
    </voice>
</speak>
"""

default_ssml = """
<speak version="0.1">
  <voice spk="Bob" seed="42" style="narration-relaxed">
    这里是一个简单的 SSML 示例 [lbreak] 
  </voice>
</speak>
"""


def create_tts_interface():
    speakers = get_speakers()

    def get_speaker_show_name(spk):
        if spk.gender == "*" or spk.gender == "":
            return spk.name
        return f"{spk.gender} : {spk.name}"

    speaker_names = ["*random"] + [
        get_speaker_show_name(speaker) for speaker in speakers
    ]

    styles = ["*auto"] + [s.get("name") for s in get_styles()]

    history = []

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("🎛️Sampling")
                temperature_input = gr.Slider(
                    0.01, 2.0, value=0.3, step=0.01, label="Temperature"
                )
                top_p_input = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Top P")
                top_k_input = gr.Slider(1, 50, value=20, step=1, label="Top K")
                batch_size_input = gr.Slider(
                    1,
                    webui_config["max_batch_size"],
                    value=4,
                    step=1,
                    label="Batch Size",
                )

            with gr.Row():
                with gr.Group():
                    gr.Markdown("🎭Style")
                    gr.Markdown("- 后缀为 `_p` 表示带prompt，效果更强但是影响质量")
                    style_input_dropdown = gr.Dropdown(
                        choices=styles,
                        # label="Choose Style",
                        interactive=True,
                        show_label=False,
                        value="*auto",
                    )
            with gr.Row():
                with gr.Group():
                    gr.Markdown("🗣️Speaker (Name or Seed)")
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
            use_decoder_input = gr.Checkbox(
                value=True, label="Use Decoder", visible=False
            )
            with gr.Group():
                gr.Markdown("🔧Prompt engineering")
                prompt1_input = gr.Textbox(label="Prompt 1")
                prompt2_input = gr.Textbox(label="Prompt 2")
                prefix_input = gr.Textbox(label="Prefix")

            infer_seed_rand_button.click(
                lambda x: int(torch.randint(0, 2**32 - 1, (1,)).item()),
                inputs=[infer_seed_input],
                outputs=[infer_seed_input],
            )
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group():
                        input_title = gr.Markdown(
                            "📝Text Input",
                            elem_id="input-title",
                        )
                        gr.Markdown(
                            f"- 字数限制{webui_config['tts_max']:,}字，超过部分截断"
                        )
                        gr.Markdown("- 如果尾字吞字不读，可以试试结尾加上 `[lbreak]`")
                        gr.Markdown(
                            "- If the input text is all in English, it is recommended to check disable_normalize"
                        )
                        text_input = gr.Textbox(
                            show_label=False,
                            label="Text to Speech",
                            lines=10,
                            placeholder="输入文本或选择示例",
                            elem_id="text-input",
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
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("🎶Refiner")
                        refine_prompt_input = gr.Textbox(
                            label="Refine Prompt",
                            value="[oral_2][laugh_0][break_6]",
                        )
                        refine_button = gr.Button("✍️Refine Text")
                        # TODO 分割句子，使用当前配置拼接为SSML，然后发送到SSML tab
                        # send_button = gr.Button("📩Split and send to SSML")

                    with gr.Group():
                        gr.Markdown("🔊Generate")
                        disable_normalize_input = gr.Checkbox(
                            value=False, label="Disable Normalize"
                        )
                        tts_button = gr.Button(
                            "🔊Generate Audio",
                            variant="primary",
                            elem_classes="big-button",
                        )

            with gr.Group():
                gr.Markdown("🎄Examples")
                sample_dropdown = gr.Dropdown(
                    choices=[sample["text"] for sample in sample_texts],
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
                tts_output = gr.Audio(label="Generated Audio")

    refine_button.click(
        refine_text,
        inputs=[text_input, refine_prompt_input],
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
            style_input_dropdown,
            disable_normalize_input,
            batch_size_input,
        ],
        outputs=tts_output,
    )


def create_ssml_interface():
    examples = [
        ssml_example1,
        ssml_example2,
        ssml_example3,
        ssml_example4,
    ]

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("📝SSML Input")
                gr.Markdown(f"- 最长{webui_config['ssml_max']:,}字符，超过会被截断")
                gr.Markdown("- 尽量保证使用相同的 seed")
                gr.Markdown(
                    "- 关于SSML可以看这个 [文档](https://github.com/lenML/ChatTTS-Forge/blob/main/docs/SSML.md)"
                )
                ssml_input = gr.Textbox(
                    label="SSML Input",
                    lines=10,
                    value=default_ssml,
                    placeholder="输入 SSML 或选择示例",
                    elem_id="ssml_input",
                    show_label=False,
                )
                ssml_button = gr.Button("🔊Synthesize SSML", variant="primary")
        with gr.Column(scale=1):
            with gr.Group():
                # 参数
                gr.Markdown("🎛️Parameters")
                # batch size
                batch_size_input = gr.Slider(
                    label="Batch Size",
                    value=4,
                    minimum=1,
                    maximum=webui_config["max_batch_size"],
                    step=1,
                )
            with gr.Group():
                gr.Markdown("🎄Examples")
                gr.Examples(
                    examples=examples,
                    inputs=[ssml_input],
                )

    ssml_output = gr.Audio(label="Generated Audio")

    ssml_button.click(
        synthesize_ssml,
        inputs=[ssml_input, batch_size_input],
        outputs=ssml_output,
    )

    return ssml_input


def split_long_text(long_text_input):
    spliter = SentenceSplitter(webui_config["spliter_threshold"])
    sentences = spliter.parse(long_text_input)
    sentences = [text_normalize(s) for s in sentences]
    data = []
    for i, text in enumerate(sentences):
        data.append([i, text, len(text)])
    return data


def merge_dataframe_to_ssml(dataframe, spk, style, seed):
    if style == "*auto":
        style = None
    if spk == "-1" or spk == -1:
        spk = None
    if seed == -1 or seed == "-1":
        seed = None

    ssml = ""
    indent = " " * 2

    for i, row in dataframe.iterrows():
        ssml += f"{indent}<voice"
        if spk:
            ssml += f' spk="{spk}"'
        if style:
            ssml += f' style="{style}"'
        if seed:
            ssml += f' seed="{seed}"'
        ssml += ">\n"
        ssml += f"{indent}{indent}{text_normalize(row[1])}\n"
        ssml += f"{indent}</voice>\n"
    return f"<speak version='0.1'>\n{ssml}</speak>"


# 长文本处理
# 可以输入长文本，并选择切割方法，切割之后可以将拼接的SSML发送到SSML tab
# 根据 。 句号切割，切割之后显示到 data table
def create_long_content_tab(ssml_input, tabs):
    speakers = get_speakers()

    def get_speaker_show_name(spk):
        if spk.gender == "*" or spk.gender == "":
            return spk.name
        return f"{spk.gender} : {spk.name}"

    speaker_names = ["*random"] + [
        get_speaker_show_name(speaker) for speaker in speakers
    ]

    styles = ["*auto"] + [s.get("name") for s in get_styles()]

    with gr.Row():
        with gr.Column(scale=1):
            # 选择说话人 选择风格 选择seed
            with gr.Group():
                gr.Markdown("🗣️Speaker")
                spk_input_text = gr.Textbox(
                    label="Speaker (Text or Seed)",
                    value="female2",
                    show_label=False,
                )
                spk_input_dropdown = gr.Dropdown(
                    choices=speaker_names,
                    interactive=True,
                    value="female : female2",
                    show_label=False,
                )
                spk_rand_button = gr.Button(
                    value="🎲",
                    variant="secondary",
                )
            with gr.Group():
                gr.Markdown("🎭Style")
                style_input_dropdown = gr.Dropdown(
                    choices=styles,
                    interactive=True,
                    show_label=False,
                    value="*auto",
                )
            with gr.Group():
                gr.Markdown("🗣️Seed")
                infer_seed_input = gr.Number(
                    value=42,
                    label="Inference Seed",
                    show_label=False,
                    minimum=-1,
                    maximum=2**32 - 1,
                )
                infer_seed_rand_button = gr.Button(
                    value="🎲",
                    variant="secondary",
                )

            send_btn = gr.Button("📩Send to SSML", variant="primary")

        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("📝Long Text Input")
                gr.Markdown("- 此页面用于处理超长文本")
                gr.Markdown("- 切割后，可以选择说话人、风格、seed，然后发送到SSML")
                long_text_input = gr.Textbox(
                    label="Long Text Input",
                    lines=10,
                    placeholder="输入长文本",
                    elem_id="long-text-input",
                    show_label=False,
                )
                long_text_split_button = gr.Button("🔪Split Text")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("🎨Output")
                long_text_output = gr.DataFrame(
                    headers=["index", "text", "length"],
                    datatype=["number", "str", "number"],
                    elem_id="long-text-output",
                    interactive=False,
                    wrap=True,
                    value=[],
                )

    spk_input_dropdown.change(
        fn=lambda x: x.startswith("*") and "-1" or x.split(":")[-1].strip(),
        inputs=[spk_input_dropdown],
        outputs=[spk_input_text],
    )
    spk_rand_button.click(
        lambda x: int(torch.randint(0, 2**32 - 1, (1,)).item()),
        inputs=[spk_input_text],
        outputs=[spk_input_text],
    )
    infer_seed_rand_button.click(
        lambda x: int(torch.randint(0, 2**32 - 1, (1,)).item()),
        inputs=[infer_seed_input],
        outputs=[infer_seed_input],
    )
    long_text_split_button.click(
        split_long_text,
        inputs=[long_text_input],
        outputs=[long_text_output],
    )

    infer_seed_rand_button.click(
        lambda x: int(torch.randint(0, 2**32 - 1, (1,)).item()),
        inputs=[infer_seed_input],
        outputs=[infer_seed_input],
    )

    send_btn.click(
        merge_dataframe_to_ssml,
        inputs=[
            long_text_output,
            spk_input_text,
            style_input_dropdown,
            infer_seed_input,
        ],
        outputs=[ssml_input],
    )

    def change_tab():
        return gr.Tabs(selected="ssml")

    send_btn.click(change_tab, inputs=[], outputs=[tabs])


def create_readme_tab():
    readme_content = read_local_readme()
    gr.Markdown(readme_content)


def create_interface():

    js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

    head_js = """
    <script>
    </script>
    """

    with gr.Blocks(js=js_func, head=head_js, title="ChatTTS Forge WebUI") as demo:
        css = """
        <style>
        .big-button {
            height: 80px;
        }
        #input_title div.eta-bar {
            display: none !important; transform: none !important;
        }
        </style>
        """

        gr.HTML(css)
        with gr.Tabs() as tabs:
            with gr.TabItem("TTS"):
                create_tts_interface()

            with gr.TabItem("SSML", id="ssml"):
                ssml_input = create_ssml_interface()

            with gr.TabItem("Long Text"):
                create_long_content_tab(ssml_input, tabs=tabs)

            with gr.TabItem("README"):
                create_readme_tab()

        gr.Markdown(
            "此项目基于 [ChatTTS-Forge](https://github.com/lenML/ChatTTS-Forge) "
        )
    return demo


if __name__ == "__main__":
    import argparse
    import dotenv

    dotenv.load_dotenv(
        dotenv_path=os.getenv("ENV_FILE", ".env.webui"),
    )

    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument("--server_name", type=str, help="server name")
    parser.add_argument("--server_port", type=int, help="server port")
    parser.add_argument(
        "--share", action="store_true", help="share the gradio interface"
    )
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--auth", type=str, help="username:password for authentication")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable half precision for model inference",
    )
    parser.add_argument(
        "--off_tqdm",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    parser.add_argument(
        "--tts_max_len",
        type=int,
        help="Max length of text for TTS",
    )
    parser.add_argument(
        "--ssml_max_len",
        type=int,
        help="Max length of text for SSML",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        help="Max batch size for TTS",
    )
    parser.add_argument(
        "--lru_size",
        type=int,
        default=64,
        help="Set the size of the request cache pool, set it to 0 will disable lru_cache",
    )
    parser.add_argument(
        "--device_id",
        type=str,
        help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)",
        default=None,
    )
    parser.add_argument(
        "--use_cpu",
        nargs="+",
        help="use CPU as torch device for specified modules",
        default=[],
        type=str.lower,
    )
    parser.add_argument("--compile", action="store_true", help="Enable model compile")

    args = parser.parse_args()

    def get_and_update_env(*args):
        val = env.get_env_or_arg(*args)
        key = args[1]
        config.runtime_env_vars[key] = val
        return val

    server_name = get_and_update_env(args, "server_name", "0.0.0.0", str)
    server_port = get_and_update_env(args, "server_port", 7860, int)
    share = get_and_update_env(args, "share", False, bool)
    debug = get_and_update_env(args, "debug", False, bool)
    auth = get_and_update_env(args, "auth", None, str)
    half = get_and_update_env(args, "half", False, bool)
    off_tqdm = get_and_update_env(args, "off_tqdm", False, bool)
    lru_size = get_and_update_env(args, "lru_size", 64, int)
    device_id = get_and_update_env(args, "device_id", None, str)
    use_cpu = get_and_update_env(args, "use_cpu", [], list)
    compile = get_and_update_env(args, "compile", False, bool)

    webui_config["tts_max"] = get_and_update_env(args, "tts_max_len", 1000, int)
    webui_config["ssml_max"] = get_and_update_env(args, "ssml_max_len", 5000, int)
    webui_config["max_batch_size"] = get_and_update_env(args, "max_batch_size", 8, int)

    demo = create_interface()

    if auth:
        auth = tuple(auth.split(":"))

    generate.setup_lru_cache()
    devices.reset_device()
    devices.first_time_calculation()

    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug,
        auth=auth,
        show_api=False,
    )
