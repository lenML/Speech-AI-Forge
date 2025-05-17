import html
import logging
import os
from datetime import datetime

import gradio as gr
import requests

from modules.core.spk import TTSSpeaker, spk_mgr

logger = logging.getLogger(__name__)


def github_fallback_download(url: str):
    """
    如果url是github.com/githubusercontent.com下的地址，那么在第一次请求失败的情况，采用镜像地址再次请求
    镜像地址： https://ghfast.top/ + (github资源url)
    """
    is_github_asset = "githubusercontent.com" in url or "github.com" in url
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        if is_github_asset:
            mirror_url = f"https://ghfast.top/{url}"
            logger.error(f"Error fetching data: {e}, try mirror url: {mirror_url}")
            logger.warning(
                f"Warning: Using mirror site can be slow. Consider setting up a proxy (set HTTPS_PROXY env) for GitHub requests."
            )
            response = requests.get(mirror_url)
            response.raise_for_status()
            logger.info(f"Success mirror url: {mirror_url}")
            # NOTE: 建议设置代理请求 github 而不是使用镜像站，因为很慢
            return response
        else:
            print(f"Error fetching data: {e}")
            return None


def fetch_speakers_data(url: str):
    """
    从指定 URL 下载音色数据。
    """
    try:
        response = github_fallback_download(url)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None


def filter_speakers(files: list[dict], hide_tags=None, search_query=""):
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


def sort_speakers(files: list[dict], sort_option="newest first"):
    """
    根据排序选项排序音色数据。
    """
    if sort_option == "newest first":
        files = sorted(files, key=lambda x: x.get("created_date", 0), reverse=True)
    elif sort_option == "oldest first":
        files = sorted(files, key=lambda x: x.get("created_date", 0))
    elif sort_option == "a-z":
        files = sorted(files, key=lambda x: x.get("name", ""))
    elif sort_option == "z-a":
        files = sorted(files, key=lambda x: x.get("name", ""), reverse=True)

    return files


def render_speakers_html(files: list[dict]):
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
      "created_date": 1730976254888,
      "avatar": "",
      "tags": [
        "原神"
      ],
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
        "Tags",
        "Created Date",
        "URL",
        "Actions",
    ]
    html_content += "<tr>"
    for header in headers:
        html_content += f"<th>{header}</th>"

    for file in files:
        id = file.get("id", "")
        name = file.get("name", "")
        desc = file.get("desc", "")
        gender = file.get("gender", "")
        author = file.get("author", "")
        version = file.get("version", "")
        filename = file.get("filename", "")
        tags = file.get("tags", [])
        # 时间是毫秒
        created_date = file.get("created_date", 0)
        avatar = file.get("avatar", "")
        url = file.get("url", "")

        html_content += f"<tr>"
        datas = [
            id[0:5],
            name,
            desc,
            gender,
            author,
            version,
            ",".join([f"#{tag}" for tag in tags]),
            datetime.fromtimestamp(created_date / 1000).strftime("%Y-%m-%d %H:%M:%S"),
        ]
        for data in datas:
            html_content += f"<td>{data}</td>"

        html_content += f'<td><a href="{url}" target="_blank">{filename}</a></td>'
        def dl_btn_html(label: str):
            return f"<td><button onclick='download_speaker(this, \"{html.escape(url)}\")'>{label}</button></td>"

        downloaded_spk = spk_mgr.get_speaker_by_id(id)
        if downloaded_spk is not None:
            if downloaded_spk.version != version:
                ver_str = version or "new"
                html_content += dl_btn_html(f"🚀Upgrade to [{ver_str}]")
            else:
                html_content += dl_btn_html("🔁ReDownload")
        else:
            html_content += f"<td><button onclick='download_speaker(this, \"{html.escape(url)}\")'>🟡Download it</button></td>"
        html_content += f"</tr>"
    html_content += "</table>"

    return html_content


def rerender_table(
    hide_tags: list[str],
    sort_option: str,
    search_query: str,
    cached_data: dict,
):
    """
    加载音色数据，并进行过滤、排序和渲染
    """
    data: dict = cached_data
    if not data:
        return "<p style='color:red;'>无数据</p>"

    files = data.get("files", [])
    files = filter_speakers(files, hide_tags, search_query)
    files = sort_speakers(files, sort_option)
    html_content = render_speakers_html(files)

    return html_content


def refresh_speakers(
    hub_url: str,
    hide_tags: list[str],
    sort_option: str,
    search_query: str,
):
    """
    无视缓存
    刷新 speakers, 也会更新 tags 列表
    """
    data: dict = fetch_speakers_data(hub_url)
    if not data:
        return "<p style='color:red;'>无法加载数据</p>", None, None

    files: list[dict] = data.get("files", [])
    files = filter_speakers(files, hide_tags, search_query)
    files = sort_speakers(files, sort_option)
    html_content = render_speakers_html(files)
    all_tags = [tag for file in files for tag in file.get("tags", [])]
    # all_tags 去重
    all_tags = list(set(all_tags))

    return html_content, data, gr.CheckboxGroup(choices=["female", "male", *all_tags])

def install_speaker(spk_url, hide_tags, sort_option, search_query, cached_data):
    """
    下载 speaker 文件到 ./data/speakers 目录下面
    """
    spk_bytes = github_fallback_download(spk_url).content

    filename = os.path.basename(spk_url)
    with open(f"./data/speakers/{filename}", "wb") as f:
        f.write(spk_bytes)
    spk_mgr.refresh()

    return rerender_table(hide_tags, sort_option, search_query, cached_data)


def create_spk_hub_ui():
    """
    加载远程的 spk hub 中的数据，并可以直接下载到本地
    """
    # 远程 JSON 文件的 URL，您可以使用环境变量设置 SPKS_INDEX
    DEFAULT_SPKS_INDEX_URL = os.getenv(
        "SPKS_INDEX",
        "https://github.com/lenML/Speech-AI-Forge-spks/raw/refs/heads/main/index.json",
    )

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
                type="value",
            )

        speakers_table = gr.HTML(label="Speaker List")
        cached_data = gr.State()  # 用于保存下载下来的数据

        # 这两个组件用来和js脚本配合 传递html中的参数
        speaker_to_install = gr.Text(elem_id="speaker_to_install", visible=False)
        install_speaker_button = gr.Button(
            elem_id="install_speaker_button", visible=False
        )

        # 按钮点击事件：加载数据并渲染，使用缓存
        refresh_button.click(
            fn=refresh_speakers,
            inputs=[spk_index_url, hide_tags, sort_option, search_query],
            outputs=[speakers_table, cached_data, hide_tags],
        )

        # 筛选和排序的变化自动刷新结果
        hide_tags.change(
            fn=rerender_table,
            inputs=[hide_tags, sort_option, search_query, cached_data],
            outputs=[speakers_table],
        )
        sort_option.change(
            fn=rerender_table,
            inputs=[hide_tags, sort_option, search_query, cached_data],
            outputs=[speakers_table],
        )
        search_query.change(
            fn=rerender_table,
            inputs=[hide_tags, sort_option, search_query, cached_data],
            outputs=[speakers_table],
        )

        # 下载逻辑
        install_speaker_button.click(
            fn=install_speaker,
            inputs=[
                speaker_to_install,
                hide_tags,
                sort_option,
                search_query,
                cached_data,
            ],
            outputs=[speakers_table],
        )
