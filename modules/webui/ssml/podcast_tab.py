import gradio as gr
import pandas as pd
import torch

from modules.normalization import text_normalize
from modules.webui import webui_utils
from modules.hf import spaces

podcast_default_case = [
    [
        1,
        "female2",
        "你好，欢迎收听今天的播客内容。今天我们要聊的是中华料理。 [lbreak]",
        "podcast_p",
    ],
    [
        2,
        "Alice",
        "嗨，我特别期待这个话题！中华料理真的是博大精深。 [lbreak]",
        "podcast_p",
    ],
    [
        3,
        "Bob",
        "没错，中华料理有着几千年的历史，而且每个地区都有自己的特色菜。 [lbreak]",
        "podcast_p",
    ],
    [
        4,
        "female2",
        "那我们先从最有名的川菜开始吧。川菜以其麻辣著称，是很多人的最爱。 [lbreak]",
        "podcast_p",
    ],
    [
        5,
        "Alice",
        "对，我特别喜欢吃麻婆豆腐和辣子鸡。那种麻辣的感觉真是让人难以忘怀。 [lbreak]",
        "podcast_p",
    ],
    [
        6,
        "Bob",
        "除了川菜，粤菜也是很受欢迎的。粤菜讲究鲜美，像是白切鸡和蒸鱼都是经典。 [lbreak]",
        "podcast_p",
    ],
    [
        7,
        "female2",
        "对啊，粤菜的烹饪方式比较清淡，更注重食材本身的味道。 [lbreak]",
        "podcast_p",
    ],
    [
        8,
        "Alice",
        "还有北京的京菜，像北京烤鸭，那可是来北京必吃的美食。 [lbreak]",
        "podcast_p",
    ],
    [
        9,
        "Bob",
        "不仅如此，还有淮扬菜、湘菜、鲁菜等等，每个菜系都有其独特的风味。 [lbreak]",
        "podcast_p",
    ],
    [
        10,
        "female2",
        "对对对，像淮扬菜的狮子头，湘菜的剁椒鱼头，都是让人垂涎三尺的美味。 [lbreak]",
        "podcast_p",
    ],
]


# NOTE: 因为 text_normalize 需要使用 tokenizer
@torch.inference_mode()
@spaces.GPU
def merge_dataframe_to_ssml(msg, spk, style, df: pd.DataFrame):
    ssml = ""
    indent = " " * 2

    for i, row in df.iterrows():
        text = row.get("text")
        spk = row.get("speaker")
        style = row.get("style")

        ssml += f"{indent}<voice"
        if spk:
            ssml += f' spk="{spk}"'
        if style:
            ssml += f' style="{style}"'
        ssml += ">\n"
        ssml += f"{indent}{indent}{text_normalize(text)}\n"
        ssml += f"{indent}</voice>\n"
    # 原封不动输出回去是为了触发 loadding 效果
    return msg, spk, style, f"<speak version='0.1'>\n{ssml}</speak>"


def create_ssml_podcast_tab(ssml_input: gr.Textbox, tabs1: gr.Tabs, tabs2: gr.Tabs):
    def get_spk_choices():
        speakers, speaker_names = webui_utils.get_speaker_names()
        speaker_names = ["-1"] + speaker_names
        return speaker_names

    styles = ["*auto"] + [s.get("name") for s in webui_utils.get_styles()]

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                spk_input_dropdown = gr.Dropdown(
                    choices=get_spk_choices(),
                    interactive=True,
                    value="female : female2",
                    show_label=False,
                )
                style_input_dropdown = gr.Dropdown(
                    choices=styles,
                    # label="Choose Style",
                    interactive=True,
                    show_label=False,
                    value="*auto",
                )
            with gr.Group():
                msg = gr.Textbox(
                    lines=5, label="Message", placeholder="Type speaker message here"
                )
                add = gr.Button("Add")
                undo = gr.Button("Undo")
                clear = gr.Button("Clear")
        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("📔Script")
                script_table = gr.DataFrame(
                    headers=["index", "speaker", "text", "style"],
                    datatype=["number", "str", "str", "str"],
                    interactive=True,
                    wrap=True,
                    value=podcast_default_case,
                    row_count=(0, "dynamic"),
                    col_count=(4, "fixed"),
                )

    send_to_ssml_btn = gr.Button("📩Send to SSML", variant="primary")

    def add_message(msg, spk, style, sheet: pd.DataFrame):
        if not msg:
            return "", sheet

        data = pd.DataFrame(
            {
                "index": [sheet.shape[0]],
                "speaker": [spk.split(" : ")[1].strip()],
                "text": [msg],
                "style": [style],
            },
        )

        # 如果只有一行 并且是空的
        is_empty = sheet.empty or (sheet.shape[0] == 1 and "text" not in sheet.iloc[0])

        if is_empty:
            sheet = data
        else:
            sheet = pd.concat(
                [
                    sheet,
                    data,
                ],
                ignore_index=True,
            )
        return "", sheet

    def undo_message(msg, spk, style, sheet: pd.DataFrame):
        if sheet.empty:
            return msg, spk, style, sheet
        data = sheet.iloc[-1]
        sheet = sheet.iloc[:-1]
        spk = ""
        for choice in get_spk_choices():
            if choice.endswith(data["speaker"]) and " : " in choice:
                spk = choice
                break
        return data["text"], spk, data["style"], sheet

    def clear_message():
        return "", pd.DataFrame(
            columns=["index", "speaker", "text", "style"],
        )

    def send_to_ssml(msg, spk, style, sheet: pd.DataFrame):
        if sheet.empty:
            return gr.Error("Please add some text to the script table.")
        msg, spk, style, ssml = merge_dataframe_to_ssml(msg, spk, style, sheet)
        return [
            msg,
            spk,
            style,
            gr.Textbox(value=ssml),
            gr.Tabs(selected="ssml"),
            gr.Tabs(selected="ssml.editor"),
        ]

    msg.submit(
        add_message,
        inputs=[msg, spk_input_dropdown, style_input_dropdown, script_table],
        outputs=[msg, script_table],
    )
    add.click(
        add_message,
        inputs=[msg, spk_input_dropdown, style_input_dropdown, script_table],
        outputs=[msg, script_table],
    )
    undo.click(
        undo_message,
        inputs=[msg, spk_input_dropdown, style_input_dropdown, script_table],
        outputs=[msg, spk_input_dropdown, style_input_dropdown, script_table],
    )
    clear.click(
        clear_message,
        outputs=[msg, script_table],
    )
    send_to_ssml_btn.click(
        send_to_ssml,
        inputs=[msg, spk_input_dropdown, style_input_dropdown, script_table],
        outputs=[
            msg,
            spk_input_dropdown,
            style_input_dropdown,
            ssml_input,
            tabs1,
            tabs2,
        ],
    )
