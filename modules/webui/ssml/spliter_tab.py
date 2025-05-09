import gradio as gr
import pandas as pd
import torch

from modules.utils.hf import spaces
from modules.webui import webui_utils
from modules.webui.webui_utils import get_styles, split_long_text, text_normalize


def merge_dataframe_to_ssml(dataframe, spk, style):
    if style == "*auto":
        style = None
    if spk == "-1" or spk == -1:
        spk = None
    if seed == -1 or seed == "-1":
        seed = None

    ssml = ""
    indent = " " * 2

    for i, row in dataframe.iterrows():
        text = row.iloc[1]

        # NOTE: 不用 normalize 了，因为调用的时候会走tn，不需要在这里预处理，还慢
        # text = text_normalize(text)

        if text.strip() == "":
            continue

        ssml += f"{indent}<voice"
        if spk:
            ssml += f' spk="{spk}"'
        if style:
            ssml += f' style="{style}"'
        if seed:
            ssml += f' seed="{seed}"'
        ssml += ">\n"
        ssml += f"{indent}{indent}{text}\n"
        ssml += f"{indent}</voice>\n"
    # 原封不动输出回去是为了触发 loadding 效果
    return dataframe, spk, style, f"<speak version='0.1'>\n{ssml}</speak>"


# 转换为 script tab 下面的 data 列表格式
def transfer_to_script_data(dataframe: pd.DataFrame, spk, style):
    script_data = []
    # table_headers = [
    #     "index",
    #     "type",
    #     "duration",
    #     "speed",
    #     "speaker",
    #     "text",
    #     "style",
    # ]
    for i, row in dataframe.iterrows():
        script_data.append(
            # {
            #     "index": i,
            #     "type": "voice",
            #     "text": row.iloc[1],
            #     "speaker": spk or "",
            #     "style": style or "",
            #     "duration": "",
            #     "speed": "",
            # }
            [
                i,
                "voice",
                "",
                "",
                spk or "",
                row.iloc[1],
                style or "",
            ]
        )
    return script_data, gr.Tabs(selected="ssml"), gr.Tabs(selected="ssml.script")


# 长文本处理
# 可以输入长文本，并选择切割方法，切割之后可以将拼接的SSML发送到SSML tab
# 根据 。 句号切割，切割之后显示到 data table
def create_spliter_tab(ssml_input, tabs1, tabs2, script_table_out):
    speakers, speaker_names = webui_utils.get_speaker_names()
    speaker_names = ["*random"] + speaker_names

    styles = ["*auto"] + [s.get("name") for s in get_styles()]

    with gr.Row():
        with gr.Column(scale=1):
            # 选择说话人 选择风格 选择seed
            with gr.Group():
                gr.Markdown("🗣️Speaker")
                spk_input_text = gr.Textbox(
                    label="Speaker",
                    value="female2",
                    show_label=False,
                    # NOTE: 由于 rand 功能不显示，所以这个也不用显示只作为一个值传递...
                    visible=False,
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
                    # NOTE: 不想支持这个功能了，容易产生歧义也没什么用
                    visible=False,
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
                gr.Markdown("🎛️Spliter")
                eos_input = gr.Textbox(
                    label="eos",
                    value="[uv_break]",
                )
                spliter_thr_input = gr.Slider(
                    label="Spliter Threshold",
                    value=100,
                    minimum=50,
                    maximum=1000,
                    step=1,
                )

        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("📝Long Text Input")
                gr.Markdown("SSML_SPLITER_GUIDE")
                long_text_input = gr.Textbox(
                    label="Long Text Input",
                    lines=10,
                    placeholder="输入长文本",
                    elem_id="long-text-input",
                    show_label=False,
                )
                long_text_split_button = gr.Button("🔪Split Text")

            with gr.Group():
                gr.Markdown("🎨Output")
                long_text_output = gr.DataFrame(
                    headers=["index", "text", "length"],
                    datatype=["number", "str", "number"],
                    elem_id="long-text-output",
                    interactive=True,
                    wrap=True,
                    value=[],
                    row_count=(0, "dynamic"),
                    col_count=(3, "fixed"),
                )

                send_btn = gr.Button("📩Send to SSML", variant="primary")
                send_script_btn = gr.Button("📩Send to Script")

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
    long_text_split_button.click(
        split_long_text,
        inputs=[
            long_text_input,
            spliter_thr_input,
            eos_input,
        ],
        outputs=[
            long_text_output,
        ],
    )

    send_btn.click(
        merge_dataframe_to_ssml,
        inputs=[
            long_text_output,
            spk_input_text,
            style_input_dropdown,
        ],
        outputs=[
            long_text_output,
            spk_input_text,
            style_input_dropdown,
            ssml_input,
        ],
    )

    def change_tab():
        return gr.Tabs(selected="ssml"), gr.Tabs(selected="ssml.editor")

    send_btn.click(change_tab, inputs=[], outputs=[tabs1, tabs2])

    send_script_btn.click(
        transfer_to_script_data,
        inputs=[long_text_output, spk_input_text, style_input_dropdown],
        outputs=[script_table_out, tabs1, tabs2],
    )
