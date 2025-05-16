import logging
import os
import sys

from modules.ffmpeg_env import setup_ffmpeg_path
from modules.repos_static.sys_paths import setup_repos_paths
from modules.webui import webui_config

try:
    # 由于 gradio 的设计，要关闭 track 需要在所有模块之前
    if "--off_track_tqdm" in sys.argv:
        webui_config.off_track_tqdm = True

    setup_repos_paths()
    setup_ffmpeg_path()
    # NOTE: 因为 logger 都是在模块中初始化，所以这个 config 必须在最前面
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
except BaseException:
    pass

import argparse

from modules import config
from modules.api.api_setup import process_api_args, setup_api_args
from modules.api.app_config import app_description, app_title, app_version
from modules.fixs.gradio_dcls_fix import dcls_patch
from modules.models_setup import process_model_args, setup_model_args
from modules.utils.env import get_and_update_env
from modules.utils.ignore_warn import ignore_useless_warnings
from modules.utils.torch_opt import configure_torch_optimizations
from modules.webui.app import create_interface, webui_init

dcls_patch()
ignore_useless_warnings()

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def setup_webui_args(parser: argparse.ArgumentParser):
    parser.add_argument("--server_name", type=str, help="server name")
    parser.add_argument("--server_port", type=int, help="server port")
    parser.add_argument(
        "--share", action="store_true", help="share the gradio interface"
    )
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--auth", type=str, help="username:password for authentication")
    parser.add_argument(
        "--tts_max_len",
        type=int,
        help="Max length of text for TTS",
    )
    parser.add_argument(
        "--ssml_max_len",
        type=int,
        help="Max length of text for SSML",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        help="Max batch size for TTS",
    )
    # webui_Experimental
    parser.add_argument(
        "--webui_experimental",
        action="store_true",
        help="Enable webui_experimental features",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Set the default language for the webui",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="use api=True to launch the API together with the webui (run launch.py for only API server)",
    )
    parser.add_argument(
        "--off_track_tqdm",
        action="store_true",
        help="turn off track_tqdm",
    )


def process_webui_args(args):
    server_name = get_and_update_env(args, "server_name", "0.0.0.0", str)
    server_port = get_and_update_env(args, "server_port", 7860, int)
    share = get_and_update_env(args, "share", False, bool)
    debug = get_and_update_env(args, "debug", False, bool)
    auth = get_and_update_env(args, "auth", None, str)
    language = get_and_update_env(args, "language", "zh-CN", str)
    api = get_and_update_env(args, "api", False, bool)

    webui_config.off_track_tqdm = get_and_update_env(
        args, "off_track_tqdm", False, bool
    )
    webui_config.experimental = get_and_update_env(
        args, "webui_experimental", False, bool
    )
    webui_config.tts_max = get_and_update_env(args, "tts_max_len", 1000, int)
    webui_config.ssml_max = get_and_update_env(args, "ssml_max_len", 5000, int)
    webui_config.max_batch_size = get_and_update_env(args, "max_batch_size", 8, int)

    webui_config.experimental = get_and_update_env(
        args, "webui_experimental", False, bool
    )
    webui_config.tts_max = get_and_update_env(args, "tts_max_len", 1000, int)
    webui_config.ssml_max = get_and_update_env(args, "ssml_max_len", 5000, int)
    webui_config.max_batch_size = get_and_update_env(args, "max_batch_size", 8, int)

    configure_torch_optimizations()
    webui_init()
    demo = create_interface()

    if auth:
        auth = tuple(auth.split(":"))

    app, local_url, share_url = demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug,
        auth=auth,
        show_api=False,
        prevent_thread_lock=True,
        inbrowser=sys.platform == "win32",
        app_kwargs={
            "title": app_title,
            "description": app_description,
            "version": app_version,
            "redoc_url": (
                None
                if api is False
                else None if config.runtime_env_vars.no_docs else "/redoc"
            ),
            "docs_url": (
                None
                if api is False
                else None if config.runtime_env_vars.no_docs else "/docs"
            ),
        },
    )
    # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
    # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
    # running web ui and do whatever the attacker wants, including installing an extension and
    # running its code. We disable this here. Suggested by RyotaK.
    app.user_middleware = [
        x for x in app.user_middleware if x.cls.__name__ != "CustomCORSMiddleware"
    ]

    if api:
        process_api_args(args, app)

    demo.block_thread()


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv(
        dotenv_path=os.getenv("ENV_FILE", ".env.webui"),
    )

    parser = argparse.ArgumentParser(description="Gradio App")
    config.runtime_env_vars.is_webui = True

    setup_webui_args(parser)
    setup_model_args(parser)
    setup_api_args(parser)

    args = parser.parse_args()

    process_model_args(args)
    process_webui_args(args)
