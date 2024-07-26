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
    ping_api,
    refiner_api,
    speaker_api,
    ssml_api,
    style_api,
    tts_api,
    xtts_v2_api,
)
from modules.utils import env

logger = logging.getLogger(__name__)


def create_api(app: FastAPI, exclude=[]):
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

    if not no_playground:
        api.setup_playground()

    if compile:
        logger.info("Model compile is enabled")

    threading.excepthook = lambda exctype, value, tb: logger.exception(
        "Uncaught exception", exc_info=(exctype, value, tb)
    )
