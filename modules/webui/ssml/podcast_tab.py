import gradio as gr
import pandas as pd
import torch

from modules.utils.hf import spaces
from modules.webui import webui_config, webui_utils


# NOTE: 因为 text_normalize 需要使用 tokenizer
@torch.inference_mode()
@spaces.GPU(duration=120)
def merge_dataframe_to_ssml(msg, spk, style, df: pd.DataFrame):
    ssml = ""
    indent = " " * 2

    for i, row in df.iterrows():
        text = row.get("text")
        spk = row.get("speaker")
        style = row.get("style")

        text = webui_utils.text_normalize(text)

        if text.strip() == "":
            continue

        ssml += f"{indent}<voice"
        if spk:
            ssml += f' spk="{spk}"'
        if style:
            ssml += f' style="{style}"'
        ssml += ">\n"
        ssml += f"{indent}{indent}{text}\n"
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
                gr.Markdown("🗣️Speaker")
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
                gr.Markdown("📝Text Input")
                msg = gr.Textbox(
                    lines=5,
                    label="Message",
                    show_label=False,
                    placeholder="Type speaker message here",
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
                    value=webui_config.localization.podcast_default,
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
            raise gr.Error("Please add some text to the script table.")
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
