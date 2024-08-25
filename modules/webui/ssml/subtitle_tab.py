import gradio as gr

import pysubs2

from modules.webui import webui_utils
import xml.dom.minidom


def read_subtitle_file(file: str, spk: str):
    if not file:
        raise gr.Error("Please upload a subtitle file")
    if spk != "*random" and ":" in spk:
        spk = spk.split(":")[-1].strip()

    subs = pysubs2.load(file, encoding="utf-8")

    ts = 0
    document = xml.dom.minidom.Document()

    root = document.createElement("speak")
    root.setAttribute("version", "0.1")

    document.appendChild(root)

    for line in subs:
        start = line.start
        end = line.end
        if ts != start:
            break_node = document.createElement("break")
            break_duration = start - ts
            break_duration = round(break_duration)
            break_node.setAttribute("time", str(break_duration) + "ms")
            root.appendChild(break_node)
        duration = end - start
        voice_node = document.createElement("voice")
        voice_node.setAttribute("duration", str(duration) + "ms")
        if spk != "*random":
            voice_node.setAttribute("spk", spk)
        voice_node.appendChild(document.createTextNode(line.text))
        root.appendChild(voice_node)
        ts = end

    xml_content = document.toprettyxml(indent=" " * 2)
    return xml_content


# 输入字幕文件，转为 ssml 格式发送到 ssml
def create_subtitle_tab(ssml_input: gr.Textbox, tabs1: gr.Tabs, tabs2: gr.Tabs):
    speakers, speaker_names = webui_utils.get_speaker_names()
    speaker_names = ["*random"] + speaker_names

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("🗣️Speaker")
                spk_input_dropdown = gr.Dropdown(
                    choices=speaker_names,
                    interactive=True,
                    value="female : female2",
                    show_label=False,
                )

        with gr.Column(scale=2):
            file_upload = gr.File(label="Upload file", file_types=[".srt", ".ass", ""])

            send_file_btn = gr.Button(value="Send to SSML")

        # with gr.Column(scale=3):
        #     text_input = gr.Textbox(label="Content")
        #     text_format = gr.Dropdown(
        #         choices=["txt", "srt", "vtt", "tsv", "json"],
        #         label="Format",
        #         value="srt",
        #     )

        #     send_text_btn = gr.Button(value="Send to SSML")

    send_file_btn.click(
        fn=read_subtitle_file,
        inputs=[file_upload, spk_input_dropdown],
        outputs=[ssml_input],
    )

    def change_tab():
        return gr.Tabs(selected="ssml"), gr.Tabs(selected="ssml.editor")

    send_file_btn.click(change_tab, inputs=[], outputs=[tabs1, tabs2])
