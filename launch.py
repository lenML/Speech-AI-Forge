import os
import logging

from modules.ffmpeg_env import setup_ffmpeg_path

setup_ffmpeg_path()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from modules.devices import devices
import argparse
import uvicorn

import torch
from modules import config
from modules.utils import env
from modules import generate_audio
from modules.api.Api import APIManager

from modules.api.impl import (
    style_api,
    tts_api,
    ssml_api,
    google_api,
    openai_api,
    refiner_api,
    speaker_api,
    ping_api,
    models_api,
)

logger = logging.getLogger(__name__)

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")


def create_api(app, no_docs=False, exclude=[]):
    app_mgr = APIManager(app=app, no_docs=no_docs, exclude_patterns=exclude)

    ping_api.setup(app_mgr)
    models_api.setup(app_mgr)
    style_api.setup(app_mgr)
    speaker_api.setup(app_mgr)
    tts_api.setup(app_mgr)
    ssml_api.setup(app_mgr)
    google_api.setup(app_mgr)
    openai_api.setup(app_mgr)
    refiner_api.setup(app_mgr)

    return app_mgr


def get_and_update_env(*args):
    val = env.get_env_or_arg(*args)
    key = args[1]
    config.runtime_env_vars[key] = val
    return val


def setup_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--compile", action="store_true", help="Enable model compile")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable half precision for model inference",
    )
    parser.add_argument(
        "--off_tqdm",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    parser.add_argument(
        "--device_id",
        type=str,
        help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)",
        default=None,
    )
    parser.add_argument(
        "--use_cpu",
        nargs="+",
        help="use CPU as torch device for specified modules",
        default=[],
        type=str.lower,
    )
    parser.add_argument(
        "--lru_size",
        type=int,
        default=64,
        help="Set the size of the request cache pool, set it to 0 will disable lru_cache",
    )
    parser.add_argument(
        "--debug_generate",
        action="store_true",
        help="Enable debug mode for audio generation",
    )


def setup_api_args(parser: argparse.ArgumentParser):
    parser.add_argument("--api_host", type=str, help="Host to run the server on")
    parser.add_argument("--api_port", type=int, help="Port to run the server on")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--cors_origin",
        type=str,
        help="Allowed CORS origins. Use '*' to allow all origins.",
    )
    parser.add_argument(
        "--no_playground",
        action="store_true",
        help="Disable the playground entry",
    )
    parser.add_argument(
        "--no_docs",
        action="store_true",
        help="Disable the documentation entry",
    )
    # 配置哪些api要跳过 比如 exclude="/v1/speakers/*,/v1/tts/*"
    parser.add_argument(
        "--exclude",
        type=str,
        help="Exclude the specified API from the server",
    )


def process_model_args(args):
    lru_size = get_and_update_env(args, "lru_size", 64, int)
    compile = get_and_update_env(args, "compile", False, bool)
    device_id = get_and_update_env(args, "device_id", None, str)
    use_cpu = get_and_update_env(args, "use_cpu", [], list)
    half = get_and_update_env(args, "half", False, bool)
    off_tqdm = get_and_update_env(args, "off_tqdm", False, bool)
    debug_generate = get_and_update_env(args, "debug_generate", False, bool)

    generate_audio.setup_lru_cache()
    devices.reset_device()
    devices.first_time_calculation()

    if debug_generate:
        generate_audio.logger.setLevel(logging.DEBUG)


def process_api_args(args, app):
    cors_origin = get_and_update_env(args, "cors_origin", "*", str)
    no_playground = get_and_update_env(args, "no_playground", False, bool)
    no_docs = get_and_update_env(args, "no_docs", False, bool)
    exclude = get_and_update_env(args, "exclude", "", str)

    api = create_api(app=app, no_docs=no_docs, exclude=exclude.split(","))
    config.api = api

    if cors_origin:
        api.set_cors(allow_origins=[cors_origin])

    if not no_playground:
        api.setup_playground()

    if compile:
        logger.info("Model compile is enabled")


app_description = """
ChatTTS-Forge 是一个功能强大的文本转语音生成工具，支持通过类 SSML 语法生成丰富的音频长文本，并提供全面的 API 服务，适用于各种场景。<br/>
ChatTTS-Forge is a powerful text-to-speech generation tool that supports generating rich audio long texts through class SSML syntax

项目地址: [https://github.com/lenML/ChatTTS-Forge](https://github.com/lenML/ChatTTS-Forge)

> 所有生成音频的 POST api都无法在此页面调试，调试建议使用 playground <br/>
> All audio generation POST APIs cannot be debugged on this page, it is recommended to use playground for debugging

> 如果你不熟悉本系统，建议从这个一键脚本开始，在colab中尝试一下：<br/>
> [https://colab.research.google.com/github/lenML/ChatTTS-Forge/blob/main/colab.ipynb](https://colab.research.google.com/github/lenML/ChatTTS-Forge/blob/main/colab.ipynb)
            """
app_title = "ChatTTS Forge API"
app_version = "0.1.0"

if __name__ == "__main__":
    import dotenv
    from fastapi import FastAPI

    dotenv.load_dotenv(
        dotenv_path=os.getenv("ENV_FILE", ".env.api"),
    )

    parser = argparse.ArgumentParser(
        description="Start the FastAPI server with command line arguments"
    )
    setup_api_args(parser)
    setup_model_args(parser)

    args = parser.parse_args()

    app = FastAPI(
        title=app_title,
        description=app_description,
        version=app_version,
        redoc_url=None if config.runtime_env_vars.no_docs else "/redoc",
        docs_url=None if config.runtime_env_vars.no_docs else "/docs",
    )

    process_model_args(args)
    process_api_args(args, app)

    host = get_and_update_env(args, "api_host", "0.0.0.0", str)
    port = get_and_update_env(args, "api_port", 7870, int)
    reload = get_and_update_env(args, "reload", False, bool)

    uvicorn.run(app, host=host, port=port, reload=reload)
