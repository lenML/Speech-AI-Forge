import os
import logging

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

import torch
from modules import config
from modules.utils import env
from modules import generate_audio as generate

from functools import lru_cache
from typing import Callable

from modules.api.Api import APIManager

from modules.api.impl import (
    base_api,
    tts_api,
    ssml_api,
    google_api,
    openai_api,
    refiner_api,
)

logger = logging.getLogger(__name__)

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")


def create_api(no_docs=False):
    api = APIManager(no_docs=no_docs)

    base_api.setup(api)
    tts_api.setup(api)
    ssml_api.setup(api)
    google_api.setup(api)
    openai_api.setup(api)
    refiner_api.setup(api)

    return api


def conditional_cache(condition: Callable):
    def decorator(func):
        @lru_cache(None)
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)

        def wrapper(*args, **kwargs):
            if condition(*args, **kwargs):
                return cached_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    import argparse
    import uvicorn
    import dotenv

    dotenv.load_dotenv(
        dotenv_path=os.getenv("ENV_FILE", ".webui.env"),
    )

    parser = argparse.ArgumentParser(
        description="Start the FastAPI server with command line arguments"
    )
    parser.add_argument("--host", type=str, help="Host to run the server on")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument("--compile", action="store_true", help="Enable model compile")
    parser.add_argument(
        "--lru_size",
        type=int,
        default=64,
        help="Set the size of the request cache pool, set it to 0 will disable lru_cache",
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

    args = parser.parse_args()

    config.args = args

    host = env.get_env_or_arg(args, "host", "0.0.0.0", str)
    port = env.get_env_or_arg(args, "port", 8000, int)
    reload = env.get_env_or_arg(args, "reload", False, bool)
    compile = env.get_env_or_arg(args, "compile", False, bool)
    lru_size = env.get_env_or_arg(args, "lru_size", 64, int)
    cors_origin = env.get_env_or_arg(args, "cors_origin", "*", str)
    no_playground = env.get_env_or_arg(args, "no_playground", False, bool)
    no_docs = env.get_env_or_arg(args, "no_docs", False, bool)
    half = env.get_env_or_arg(args, "half", False, bool)
    off_tqdm = env.get_env_or_arg(args, "off_tqdm", False, bool)

    if compile:
        print("Model compile is enabled")
        config.enable_model_compile = True

    def should_cache(*args, **kwargs):
        spk_seed = kwargs.get("spk_seed", -1)
        infer_seed = kwargs.get("infer_seed", -1)
        return spk_seed != -1 and infer_seed != -1

    if lru_size > 0:
        config.lru_size = lru_size
        generate.generate_audio = conditional_cache(should_cache)(
            generate.generate_audio
        )

    api = create_api(no_docs=no_docs)
    config.api = api

    if cors_origin:
        api.set_cors(allow_origins=[cors_origin])

    if not no_playground:
        api.setup_playground()

    if half:
        config.model_config["half"] = True

    if off_tqdm:
        config.disable_tqdm = True

    if args.compile:
        logger.info("Model compile is enabled")
        config.enable_model_compile = True

    def should_cache(*args, **kwargs):
        spk_seed = kwargs.get("spk_seed", -1)
        infer_seed = kwargs.get("infer_seed", -1)
        return spk_seed != -1 and infer_seed != -1

    if args.lru_size > 0:
        config.lru_size = args.lru_size
        generate.generate_audio = conditional_cache(should_cache)(
            generate.generate_audio
        )

    api = create_api(no_docs=args.no_docs)
    config.api = api

    if args.cors_origin:
        api.set_cors(allow_origins=[args.cors_origin])

    if not args.no_playground:
        api.setup_playground()

    if args.half:
        config.model_config["half"] = True

    if args.off_tqdm:
        config.disable_tqdm = True

    uvicorn.run(api.app, host=args.host, port=args.port, reload=args.reload)
