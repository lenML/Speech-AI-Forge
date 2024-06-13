import logging
from modules.devices import devices
import argparse

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
    xtts_v2_api,
)

logger = logging.getLogger(__name__)


def create_api(app, exclude=[]):
    app_mgr = APIManager(app=app, exclude_patterns=exclude)

    ping_api.setup(app_mgr)
    models_api.setup(app_mgr)
    style_api.setup(app_mgr)
    speaker_api.setup(app_mgr)
    tts_api.setup(app_mgr)
    ssml_api.setup(app_mgr)
    google_api.setup(app_mgr)
    openai_api.setup(app_mgr)
    refiner_api.setup(app_mgr)
    xtts_v2_api.setup(app_mgr)

    return app_mgr


def setup_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--compile", action="store_true", help="Enable model compile")
    parser.add_argument(
        "--no_half",
        action="store_true",
        help="Disalbe half precision for model inference",
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


def process_model_args(args):
    lru_size = env.get_and_update_env(args, "lru_size", 64, int)
    compile = env.get_and_update_env(args, "compile", False, bool)
    device_id = env.get_and_update_env(args, "device_id", None, str)
    use_cpu = env.get_and_update_env(args, "use_cpu", [], list)
    no_half = env.get_and_update_env(args, "no_half", False, bool)
    off_tqdm = env.get_and_update_env(args, "off_tqdm", False, bool)
    debug_generate = env.get_and_update_env(args, "debug_generate", False, bool)

    generate_audio.setup_lru_cache()
    devices.reset_device()
    devices.first_time_calculation()

    if debug_generate:
        generate_audio.logger.setLevel(logging.DEBUG)


def setup_uvicon_args(parser: argparse.ArgumentParser):
    parser.add_argument("--host", type=str, help="Host to run the server on")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--log_level", type=str, help="Log level")
    parser.add_argument("--access_log", action="store_true", help="Enable access log")
    parser.add_argument(
        "--proxy_headers", action="store_true", help="Enable proxy headers"
    )
    parser.add_argument(
        "--timeout_keep_alive", type=int, help="Keep-alive timeout duration"
    )
    parser.add_argument(
        "--timeout_graceful_shutdown",
        type=int,
        help="Graceful shutdown timeout duration",
    )
    parser.add_argument("--ssl_keyfile", type=str, help="SSL key file path")
    parser.add_argument("--ssl_certfile", type=str, help="SSL certificate file path")
    parser.add_argument(
        "--ssl_keyfile_password", type=str, help="SSL key file password"
    )


def setup_api_args(parser: argparse.ArgumentParser):
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


def process_api_args(args, app):
    cors_origin = env.get_and_update_env(args, "cors_origin", "*", str)
    no_playground = env.get_and_update_env(args, "no_playground", False, bool)
    no_docs = env.get_and_update_env(args, "no_docs", False, bool)
    exclude = env.get_and_update_env(args, "exclude", "", str)

    api = create_api(app=app, exclude=exclude.split(","))
    config.api = api

    if cors_origin:
        api.set_cors(allow_origins=[cors_origin])

    if not no_playground:
        api.setup_playground()

    if compile:
        logger.info("Model compile is enabled")
