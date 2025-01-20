import argparse
import logging
import threading

from fastapi import FastAPI

from modules import config
from modules.api.Api import APIManager
from modules.api.impl import (
    google_api,
    models_api,
    openai_api,
    refiner_api,
    speaker_api,
    ssml_api,
    stt_api,
    style_api,
    sys_api,
    tts_api,
    vc_api,
    xtts_v2_api,
)
from modules.api.v2 import (
    tts_api as tts_api_v2,
    #
)
from modules.utils import env

logger = logging.getLogger(__name__)


def create_api(app: FastAPI, exclude=[]):
    app_mgr = APIManager(app=app, exclude_patterns=exclude)

    sys_api.setup(app_mgr)
    models_api.setup(app_mgr)
    style_api.setup(app_mgr)
    speaker_api.setup(app_mgr)
    tts_api.setup(app_mgr)
    ssml_api.setup(app_mgr)
    google_api.setup(app_mgr)
    openai_api.setup(app_mgr)
    refiner_api.setup(app_mgr)
    xtts_v2_api.setup(app_mgr)
    stt_api.setup(app_mgr)
    vc_api.setup(app_mgr)

    # v2 apis
    tts_api_v2.setup(app_mgr)

    return app_mgr


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


def process_api_args(args: argparse.Namespace, app: FastAPI):
    cors_origin = env.get_and_update_env(args, "cors_origin", "*", str)
    no_playground = env.get_and_update_env(args, "no_playground", False, bool)
    no_docs = env.get_and_update_env(args, "no_docs", False, bool)
    exclude = env.get_and_update_env(args, "exclude", "", str)

    api = create_api(app=app, exclude=exclude.split(","))
    config.api = api

    if cors_origin:
        api.set_cors(allow_origins=[cors_origin])
        logger.info(f"allow CORS origin: {cors_origin}")

    if not no_playground:
        api.setup_playground()

    threading.excepthook = lambda exctype: logger.exception(
        "Uncaught exception", exc_info=(exctype)
    )
