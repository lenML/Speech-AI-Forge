import logging
import os

import torch
import gradio as gr

from modules import config
from modules.webui import webui_config

from modules.webui.system_tab import create_system_tab
from modules.webui.tts_tab import create_tts_interface
from modules.webui.ssml_tab import create_ssml_interface
from modules.webui.spliter_tab import create_spliter_tab
from modules.webui.speaker_tab import create_speaker_panel
from modules.webui.readme_tab import create_readme_tab

logger = logging.getLogger(__name__)


def webui_init():
    # fix: If the system proxy is enabled in the Windows system, you need to skip these
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision("high")

    logger.info("WebUI module initialized")


def create_app_footer():
    gradio_version = gr.__version__
    git_tag = config.versions.git_tag
    git_commit = config.versions.git_commit
    git_branch = config.versions.git_branch
    python_version = config.versions.python_version
    torch_version = config.versions.torch_version

    config.versions.gradio_version = gradio_version

    gr.Markdown(
        f"""
🍦 [ChatTTS-Forge](https://github.com/lenML/ChatTTS-Forge)
version: [{git_tag}](https://github.com/lenML/ChatTTS-Forge/commit/{git_commit}) | branch: `{git_branch}` | python: `{python_version}` | torch: `{torch_version}`
        """
    )


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
        footer {
            display: none !important;
        }
        </style>
        """

        gr.HTML(css)
        with gr.Tabs() as tabs:
            with gr.TabItem("TTS"):
                create_tts_interface()

            with gr.TabItem("SSML", id="ssml"):
                ssml_input = create_ssml_interface()

            with gr.TabItem("Spilter"):
                create_spliter_tab(ssml_input, tabs=tabs)

            with gr.TabItem("Speaker"):
                create_speaker_panel()
            with gr.TabItem("Inpainting", visible=webui_config.experimental):
                gr.Markdown("🚧 Under construction")
            with gr.TabItem("ASR", visible=webui_config.experimental):
                gr.Markdown("🚧 Under construction")

            with gr.TabItem("System"):
                create_system_tab()

            with gr.TabItem("README"):
                create_readme_tab()

        create_app_footer()
    return demo
