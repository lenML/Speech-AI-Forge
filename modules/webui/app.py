import logging
import os

import gradio as gr

from modules import config
from modules.webui import gradio_extensions, webui_config
from modules.webui.asr_tab import create_asr_tab
from modules.webui.audio_tools.tools_tab import create_tools_tab
from modules.webui.changelog_tab import create_changelog_tab
from modules.webui.finetune.ft_tab import create_ft_tabs
from modules.webui.localization_runtime import ENLocalizationVars, ZHLocalizationVars
from modules.webui.readme_tab import create_readme_tab
from modules.webui.speaker_tab import create_speaker_panel
from modules.webui.ssml.podcast_tab import create_ssml_podcast_tab
from modules.webui.ssml.script_tab import create_script_tab
from modules.webui.ssml.spliter_tab import create_spliter_tab
from modules.webui.ssml.ssml_tab import create_ssml_interface
from modules.webui.ssml.subtitle_tab import create_subtitle_tab
from modules.webui.system_tab import create_system_tab
from modules.webui.tts_tab import create_tts_interface

logger = logging.getLogger(__name__)


def webui_init():
    # fix: If the system proxy is enabled in the Windows system, you need to skip these
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

    if config.runtime_env_vars.language == "en":
        webui_config.localization = ENLocalizationVars()
    else:
        webui_config.localization = ZHLocalizationVars()

    logger.info("WebUI module initialized")


def create_app_footer():
    gradio_version = gr.__version__
    git_tag = os.environ.get("V_GIT_TAG") or config.versions.git_tag
    git_commit = os.environ.get("V_GIT_COMMIT") or config.versions.git_commit
    git_branch = os.environ.get("V_GIT_BRANCH") or config.versions.git_branch
    python_version = config.versions.python_version
    torch_version = config.versions.torch_version
    ffmpeg_version = config.versions.ffmpeg_version

    config.versions.gradio_version = gradio_version

    footer_items = ["🍦 [Speech-AI-Forge](https://github.com/lenML/Speech-AI-Forge)"]
    footer_items.append(
        f"version: [{git_tag}](https://github.com/lenML/Speech-AI-Forge/commit/{git_commit})"
    )
    footer_items.append(f"branch: `{git_branch}`")
    footer_items.append(f"python: `{python_version}`")
    footer_items.append(f"torch: `{torch_version}`")
    footer_items.append(f"ffmpeg: `{ffmpeg_version}`")

    if config.runtime_env_vars.api and not config.runtime_env_vars.no_docs:
        footer_items.append(f"[api](/docs)")

    gr.Markdown(
        " | ".join(footer_items),
        elem_classes=["no-translate"],
    )


def create_interface():
    gradio_extensions.reload_javascript()

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

    with gr.Blocks(js=js_func, head=head_js, title="Speech-AI Forge WebUI") as demo:
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
                with gr.Tabs() as ssml_tabs:
                    with gr.TabItem("Editor", id="ssml.editor"):
                        ssml_input = create_ssml_interface()
                    with gr.TabItem("Script", id="ssml.script"):
                        script_table = create_script_tab(
                            ssml_input=ssml_input, tabs1=tabs, tabs2=ssml_tabs
                        )
                    with gr.TabItem("Spilter"):
                        create_spliter_tab(
                            ssml_input=ssml_input,
                            tabs1=tabs,
                            tabs2=ssml_tabs,
                            script_table_out=script_table,
                        )
                    with gr.TabItem("Podcast"):
                        create_ssml_podcast_tab(
                            ssml_input=ssml_input,
                            tabs1=tabs,
                            tabs2=ssml_tabs,
                            script_table_out=script_table,
                        )
                    with gr.TabItem("From subtitle"):
                        create_subtitle_tab(
                            ssml_input=ssml_input,
                            tabs1=tabs,
                            tabs2=ssml_tabs,
                            script_table_out=script_table,
                        )

            with gr.TabItem("Speaker"):
                create_speaker_panel()
            with gr.TabItem("Inpainting", visible=webui_config.experimental):
                gr.Markdown("🚧 Under construction")
            with gr.TabItem("ASR"):
                create_asr_tab()
            with gr.TabItem("Finetune", visible=webui_config.experimental):
                create_ft_tabs(demo)

            with gr.TabItem("Tools"):
                create_tools_tab()

            with gr.TabItem("System"):
                create_system_tab(demo)

            with gr.TabItem("README"):
                with gr.Tabs():
                    with gr.TabItem("readme"):
                        create_readme_tab()
                    # TODO: changlog 现在删了，还没搞新的
                    # with gr.TabItem("changelog"):
                    #     create_changelog_tab()

        create_app_footer()

    # Dump the English config for the localization
    # ** JUST for developer
    # localization.dump_english_config(gradio_hijack.all_components)
    return demo
