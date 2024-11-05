import xml.dom.minidom

import gradio as gr
import pandas as pd

from modules.webui import webui_config, webui_utils


# UPDATE NOTE: webui_utils.text_normalize 里面屏蔽了 gpu 要求，所以这个函数不需要 gpu 资源了
def merge_dataframe_to_ssml(msg, spk, style, df: pd.DataFrame):
    document = xml.dom.minidom.Document()

    root = document.createElement("speak")
    root.setAttribute("version", "0.1")

    document.appendChild(root)

    for _, row in df.iterrows():
        text = row.get("text")
        spk = row.get("speaker")
        style = row.get("style")

        text = webui_utils.text_normalize(text)

        if text.strip() == "":
            continue

        voice_node = document.createElement("voice")
        if spk:
            voice_node.setAttribute("spk", spk)
        if style and style != "*auto":
            voice_node.setAttribute("style", style)

        voice_node.appendChild(document.createTextNode(text))
        root.appendChild(voice_node)

    xml_content = document.toprettyxml(indent="  ")
    return xml_content


# 转换为 script tab 下面的 data 列表格式
def transfer_to_script_data(dataframe: pd.DataFrame):
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
            #     "type": "voice",
            #     "index": row.iloc[0],
            #     "speaker": row.iloc[1],
            #     "text": row.iloc[2],
            #     "style": row.iloc[3],
            #     "duration": "",
            #     "speed": "",
            # }
            [row.iloc[0], "voice", "", "", row.iloc[1], row.iloc[2], row.iloc[3]]
        )
    return script_data, gr.Tabs(selected="ssml"), gr.Tabs(selected="ssml.script")


def create_ssml_podcast_tab(
    ssml_input: gr.Textbox,
    tabs1: gr.Tabs,
    tabs2: gr.Tabs,
    script_table_out: gr.DataFrame,
):
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
                add = gr.Button("➕ Add")
                undo = gr.Button("🔙 Undo")
                clear = gr.Button("🧹 Clear")

            with gr.Group():
                reload = gr.Button("🔄 Reload demo script")

            with gr.Group():
                gr.Markdown("🎶Refiner (ChatTTS)")
                rf_oral_input = gr.Slider(
                    label="Oral", value=2, minimum=-1, maximum=9, step=1
                )
                rf_speed_input = gr.Slider(
                    label="Speed", value=2, minimum=-1, maximum=9, step=1
                )
                rf_break_input = gr.Slider(
                    label="Break", value=2, minimum=-1, maximum=7, step=1
                )
                rf_laugh_input = gr.Slider(
                    label="Laugh", value=0, minimum=-1, maximum=2, step=1
                )
                refine_button = gr.Button("✍️Refine Text")

        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("📔Script")
                script_table = gr.DataFrame(
                    headers=["index", "speaker", "text", "style"],
                    datatype=["number", "str", "str", "str"],
                    interactive=True,
                    wrap=True,
                    value=webui_config.localization.podcast_default.copy(),
                    row_count=(0, "dynamic"),
                    col_count=(4, "fixed"),
                )

            send_to_ssml_btn = gr.Button("📩Send to SSML", variant="primary")
            send_to_script_btn = gr.Button("📩Send to Script")

    def add_message(msg, spk, style, sheet: pd.DataFrame):
        if not msg:
            return "", sheet

        speaker = spk
        if ":" in spk:
            speaker = spk.split(" : ")[1].strip()
        else:
            speaker = ""

        data = pd.DataFrame(
            {
                "index": [sheet.shape[0]],
                "speaker": [speaker],
                "text": [msg],
                "style": [style],
            },
        )

        # 如果只有一行 并且是空的
        is_empty = (
            sheet.empty
            or (sheet.shape[0] == 1 and "text" not in sheet.iloc[0])
            or (sheet.shape[0] == 1 and sheet.iloc[0]["text"] == "")
        )

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
        ssml = merge_dataframe_to_ssml(msg, spk, style, sheet)
        return (
            msg,
            spk,
            style,
            gr.Textbox(value=ssml),
            gr.Tabs(selected="ssml"),
            gr.Tabs(selected="ssml.editor"),
        )

    def reload_default_data():
        data = webui_config.localization.podcast_default.copy()
        return "", pd.DataFrame(
            data,
            columns=["index", "speaker", "text", "style"],
        )

    def refine_texts(
        sheet: pd.DataFrame,
        oral,
        speed,
        break_,
        laugh,
        progress: gr.Progress = gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
    ):
        """
        将每行的 text 调用 refine
        """

        def need_refine(text: str):
            if text.strip() == "":
                return False
            if "[uv_break]" in text or "[laugh]" in text or "[v_break]" in text:
                return False
            return True

        for i, row in sheet.iterrows():
            text = row["text"]
            if not need_refine(text):
                continue
            text = webui_utils.refine_text(
                text=text,
                oral=oral,
                speed=speed,
                rf_break=break_,
                laugh=laugh,
            )
            text = text.replace("\n", " ")
            sheet.at[i, "text"] = text
        return sheet

    refine_button.click(
        refine_texts,
        inputs=[
            script_table,
            rf_oral_input,
            rf_speed_input,
            rf_break_input,
            rf_laugh_input,
        ],
        outputs=script_table,
    )
    reload.click(
        reload_default_data,
        outputs=[msg, script_table],
    )
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
    send_to_script_btn.click(
        transfer_to_script_data,
        inputs=[script_table],
        outputs=[
            script_table_out,
            tabs1,
            tabs2,
        ],
    )
