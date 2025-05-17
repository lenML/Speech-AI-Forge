import tempfile
import xml.dom.minidom

import gradio as gr
import pysubs2

from modules.core.ssml.SSMLParser import SSMLBreak, SSMLSegment
from modules.webui import webui_utils


def read_subtitle_file_to_segs(file: str, spk: str):
    if spk != "*random" and ":" in spk:
        spk = spk.split(":")[-1].strip()

    subs = pysubs2.load(file, encoding="utf-8")
    # subs 根据 start 排序 从小到大
    subs_list = sorted(subs, key=lambda x: x.start)

    cursor = 0

    segs = []

    for line in subs_list:
        start = line.start
        end = line.end

        # 如果当前字幕的起始时间早于前一条的结束时间（即有重叠）
        if start < cursor:
            # 裁剪掉前面的重叠部分
            start = cursor
            if start >= end:
                continue  # 完全被前一个覆盖，跳过

        # 插入 break
        if start > cursor:
            segs.append(SSMLBreak(round(start - cursor)))

        text = line.text.strip()
        if not text:
            cursor = end
            continue

        seg = SSMLSegment(text=text)
        # NOTE: 这种随机 spk 应该要废弃...
        if spk != "*random":
            seg.attrs.spk = spk
        seg.attrs.duration = f"{end - start}ms"
        segs.append(seg)

        cursor = end

    return segs


def read_subtitle_file_to_ssml(file: str, spk: str):
    if not file:
        raise gr.Error("Please upload a subtitle file")
    if spk != "*random" and ":" in spk:
        spk = spk.split(":")[-1].strip()

    subs = pysubs2.load(file, encoding="utf-8")
    # subs 根据 start 排序 从小到大
    subs_list = sorted(subs, key=lambda x: x.start)

    cursor = 0
    document = xml.dom.minidom.Document()

    root = document.createElement("speak")
    root.setAttribute("version", "0.1")

    document.appendChild(root)

    for line in subs_list:
        start = line.start
        end = line.end

        # 如果当前字幕的起始时间早于前一条的结束时间（即有重叠）
        if start < cursor:
            # 裁剪掉前面的重叠部分
            start = cursor
            if start >= end:
                continue  # 完全被前一个覆盖，跳过

        # 插入 break
        if start > cursor:
            break_node = document.createElement("break")
            break_duration = start - cursor
            break_duration = round(break_duration)
            if break_duration > 0:
                break_node.setAttribute("time", str(break_duration) + "ms")
                root.appendChild(break_node)
            else:
                # TODO: logging
                pass

        text = line.text.strip()
        if not text:
            cursor = end
            continue

        duration = end - start
        if duration < 0:
            # TODO: logging
            continue
        voice_node = document.createElement("voice")
        voice_node.setAttribute("duration", str(duration) + "ms")
        # NOTE: 这种随机 spk 应该废弃
        if spk != "*random":
            voice_node.setAttribute("spk", spk)
        voice_node.appendChild(document.createTextNode(text))
        root.appendChild(voice_node)

        cursor = end

    xml_content = document.toprettyxml(indent=" " * 2)
    return xml_content


# 读取字幕为表格，用于传递给 脚本 tab
def read_subtitle_to_table(file: str, spk: str):
    if not file:
        raise gr.Error("Please upload a subtitle file")

    segs = read_subtitle_file_to_segs(file, spk)
    data = []
    # table_headers = [
    #     "index",
    #     "type",
    #     "duration",
    #     "speed",
    #     "speaker",
    #     "text",
    #     "style",
    # ]
    index = 1
    for seg in segs:
        if isinstance(seg, SSMLSegment):
            data.append(
                [
                    index,
                    "voice",
                    seg.attrs.duration,
                    "",
                    seg.attrs.spk,
                    seg.text,
                    "",
                ]
            )
        elif isinstance(seg, SSMLBreak):
            duration = seg.attrs.duration
            data.append(
                [
                    index,
                    "break",
                    f"{duration}ms" if isinstance(duration, int) else duration,
                    "",
                    "",
                    "",
                    "",
                ]
            )
        else:
            raise ValueError(f"Unknown segment type: {type(seg)}")
        index += 1

    return data


def read_subtitle_text_to_table(text: str, format: str, spk: str):
    # 存为临时文件，然后调用之前的函数
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp_file:
        tmp_file.write(text.encode("utf-8"))
        tmp_file_path = tmp_file.name

    return read_subtitle_to_table(tmp_file_path, spk)


def read_subtitle_text_to_ssml(text: str, format: str, spk: str):
    # 存为临时文件，然后调用之前的函数
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp_file:
        tmp_file.write(text.encode("utf-8"))
        tmp_file_path = tmp_file.name

    return read_subtitle_file_to_ssml(tmp_file_path, spk)


# 输入字幕文件，转为 ssml 格式发送到 ssml
def create_subtitle_tab(
    ssml_input: gr.Textbox,
    tabs1: gr.Tabs,
    tabs2: gr.Tabs,
    script_table_out: gr.DataFrame,
):
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
            # 从文件上传
            with gr.TabItem("From File"):
                file_upload = gr.File(
                    label="Upload file", file_types=[".srt", ".ass", ""]
                )

                send_ssml_btn = gr.Button("📩Send to SSML", variant="primary")
                send_script_btn = gr.Button("📩Send to Script")
            # 从文本上传
            with gr.TabItem("From Text"):
                text_input = gr.Textbox(
                    label="Content", lines=10, placeholder="Type here..."
                )
                text_format = gr.Dropdown(
                    choices=["srt", "vtt", "lrc", "ass"],
                    label="Format",
                    value="srt",
                )

                send_text_btn = gr.Button(value="Send to SSML", variant="primary")
                send_text_script_btn = gr.Button("📩Send to Script")

    send_ssml_btn.click(
        fn=read_subtitle_file_to_ssml,
        inputs=[file_upload, spk_input_dropdown],
        outputs=[ssml_input],
    )
    send_script_btn.click(
        fn=read_subtitle_to_table,
        inputs=[file_upload, spk_input_dropdown],
        outputs=[script_table_out],
    )
    send_text_btn.click(
        fn=read_subtitle_text_to_ssml,
        inputs=[text_input, text_format, spk_input_dropdown],
        outputs=[ssml_input],
    )
    send_text_script_btn.click(
        fn=read_subtitle_text_to_table,
        inputs=[text_input, text_format, spk_input_dropdown],
        outputs=[script_table_out],
    )

    def to_ssml_tab():
        return gr.Tabs(selected="ssml"), gr.Tabs(selected="ssml.editor")

    def to_script_tab():
        return gr.Tabs(selected="ssml"), gr.Tabs(selected="ssml.script")

    send_ssml_btn.click(to_ssml_tab, inputs=[], outputs=[tabs1, tabs2])
    send_script_btn.click(to_script_tab, inputs=[], outputs=[tabs1, tabs2])
    send_text_btn.click(to_ssml_tab, inputs=[], outputs=[tabs1, tabs2])
    send_text_script_btn.click(to_script_tab, inputs=[], outputs=[tabs1, tabs2])
