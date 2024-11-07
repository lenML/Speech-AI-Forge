import html
import os
import gradio as gr
import requests

from modules.core.spk import TTSSpeaker, spk_mgr

# 远程 JSON 文件的 URL，您可以使用环境变量设置 SPKS_INDEX
DEFAULT_SPKS_INDEX_URL = os.getenv(
    "SPKS_INDEX",
    "https://github.com/lenML/Speech-AI-Forge-spks/raw/refs/heads/main/index.json",
)


def fetch_speakers_data(url):
    """
    从指定 URL 下载音色数据。
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def filter_speakers(files, hide_tags=None, search_query=""):
    """
    根据 hide_tags 和搜索关键词过滤音色数据。
    """

    if hide_tags:
        # 过滤 tags
        files = [
            file
            for file in files
            if not any(tag in file.get("tags", []) for tag in hide_tags)
        ]
        # 过滤 gender
        files = [file for file in files if not file.get("gender") in hide_tags]

    if search_query:
        files = [
            file
            for file in files
            if search_query.lower() in file.get("name", "").lower()
        ]

    return files


def sort_speakers(files, sort_option="newest first"):
    """
    根据排序选项排序音色数据。
    """
    if sort_option == "newest first":
        files = sorted(files, key=lambda x: x.get("created_date", ""), reverse=True)
    elif sort_option == "oldest first":
        files = sorted(files, key=lambda x: x.get("created_date", ""))
    elif sort_option == "a-z":
        files = sorted(files, key=lambda x: x.get("name", ""))
    elif sort_option == "z-a":
        files = sorted(files, key=lambda x: x.get("name", ""), reverse=True)

    return files


def render_speakers_html(files):
    """
    将音色数据渲染为 HTML 表格。

    file 例子:
    {
      "id": "9609c6a2d27b485fb0c8aec05a684579",
      "name": "mona",
      "desc": "mona",
      "gender": "female",
      "author": "",
      "version": "",
      "filename": "yuanshen/mona.spkv1.json",
      "url": "https://github.com/lenML/Speech-AI-Forge-spks/raw/refs/heads/main/spks/yuanshen/mona.spkv1.json"
    }
    """
    html_content = "<table>"
    headers = [
        "ID(0:5)",
        "Name",
        "Description",
        "Gender",
        "Author",
        "Version",
        "URL",
        "Actions",
    ]
    html_content += "<tr>"
    for header in headers:
        html_content += f"<th>{header}</th>"

    for file in files:
        id = file["id"]
        name = file["name"]
        desc = file["desc"]
        gender = file["gender"]
        author = file["author"]
        version = file["version"]
        filename = file["filename"]
        url = file["url"]

        html_content += f"<tr>"
        datas = [id[0:5], name, desc, gender, author, version]
        for data in datas:
            html_content += f"<td>{data}</td>"

        html_content += f'<td><a href="{url}" target="_blank">{filename}</a></td>'

        downloaded = spk_mgr.get_speaker_by_id(id) is not None
        if downloaded:
            html_content += f"<td>✅downloaded</td>"
        else:
            html_content += f"<td><button onclick='download_speaker(this, \"{html.escape(url)}\")'>🟡Download</button></td>"
        html_content += f"</tr>"
    html_content += "</table>"

    return html_content


def load_and_process_speakers(url, hide_tags, sort_option, search_query, cached_data):
    """
    加载音色数据，并进行过滤、排序和渲染。优先使用缓存数据。
    """
    # 如果有缓存数据，则直接使用缓存
    data = cached_data if cached_data else fetch_speakers_data(url)
    if not data:
        return "<p style='color:red;'>无法加载数据</p>", None

    files = data.get("files", [])
    files = filter_speakers(files, hide_tags, search_query)
    files = sort_speakers(files, sort_option)
    html_content = render_speakers_html(files)

    return html_content, data  # 返回 HTML 内容和下载的数据（缓存用）


def install_speaker(
    spk_url, hub_url, hide_tags, sort_option, search_query, cached_data
):
    """
    下载 speaker 文件到 ./data/speakers 目录下面
    """
    response = requests.get(spk_url)
    response.raise_for_status()

    filename = os.path.basename(spk_url)
    with open(f"./data/speakers/{filename}", "wb") as f:
        f.write(response.content)
    spk_mgr.refresh()

    return load_and_process_speakers(
        hub_url, hide_tags, sort_option, search_query, cached_data
    )


def create_spk_hub_ui():
    """
    加载远程的 spk hub 中的数据，并可以直接下载到本地
    """
    with gr.TabItem("Available", id="available"):
        with gr.Row():
            with gr.Column(scale=1):
                refresh_button = gr.Button(value="Load from:", variant="primary")
            with gr.Column(scale=5):
                spk_index_url = gr.Text(
                    value=DEFAULT_SPKS_INDEX_URL,
                    label="Directory Index URL",
                    container=False,
                    lines=1,
                )

        with gr.Row():
            search_query = gr.Text(label="Search", show_label=True)
            hide_tags = gr.CheckboxGroup(
                value=[],
                label="Hide voices with tags",
                choices=["female", "male"],
            )
            sort_option = gr.Radio(
                value="newest first",
                label="Order",
                choices=["newest first", "oldest first", "a-z", "z-a"],
                type="index",
            )

        load_result = gr.HTML(label="Speaker List")
        cached_data = gr.State()  # 用于保存下载下来的数据

        # 这两个组件用来和js脚本配合 传递html中的参数
        speaker_to_install = gr.Text(elem_id="speaker_to_install", visible=False)
        install_speaker_button = gr.Button(
            elem_id="install_speaker_button", visible=False
        )

        # 按钮点击事件：加载数据并渲染，使用缓存
        refresh_button.click(
            fn=load_and_process_speakers,
            inputs=[spk_index_url, hide_tags, sort_option, search_query, cached_data],
            outputs=[load_result, cached_data],
        )

        # 筛选和排序的变化自动刷新结果
        hide_tags.change(
            fn=load_and_process_speakers,
            inputs=[spk_index_url, hide_tags, sort_option, search_query, cached_data],
            outputs=[load_result, cached_data],
        )
        sort_option.change(
            fn=load_and_process_speakers,
            inputs=[spk_index_url, hide_tags, sort_option, search_query, cached_data],
            outputs=[load_result, cached_data],
        )
        search_query.change(
            fn=load_and_process_speakers,
            inputs=[spk_index_url, hide_tags, sort_option, search_query, cached_data],
            outputs=[load_result, cached_data],
        )

        # 下载逻辑
        install_speaker_button.click(
            fn=install_speaker,
            inputs=[
                speaker_to_install,
                spk_index_url,
                hide_tags,
                sort_option,
                search_query,
                cached_data,
            ],
            outputs=[load_result, cached_data],
        )
