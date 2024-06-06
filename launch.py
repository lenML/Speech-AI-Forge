import os
import logging

from modules.devices import devices
from modules.utils.cache import conditional_cache

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

import torch
from modules import config
from modules.utils import env
from modules import generate_audio as generate


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


def create_api(no_docs=False, exclude=[]):
    app = APIManager(no_docs=no_docs, exclude_patterns=exclude)

    ping_api.setup(app)
    models_api.setup(app)
    style_api.setup(app)
    speaker_api.setup(app)
    tts_api.setup(app)
    ssml_api.setup(app)
    google_api.setup(app)
    openai_api.setup(app)
    refiner_api.setup(app)

    return app


if __name__ == "__main__":
    import argparse
    import uvicorn
    import dotenv

    dotenv.load_dotenv(
        dotenv_path=os.getenv("ENV_FILE", ".env.api"),
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
    # 配置哪些api要跳过 比如 exclude="/v1/speakers/*,/v1/tts/*"
    parser.add_argument(
        "--exclude",
        type=str,
        help="Exclude the specified API from the server",
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

    args = parser.parse_args()

    def get_and_update_env(*args):
        val = env.get_env_or_arg(*args)
        key = args[1]
        config.runtime_env_vars[key] = val
        return val

    host = get_and_update_env(args, "host", "0.0.0.0", str)
    port = get_and_update_env(args, "port", 8000, int)
    reload = get_and_update_env(args, "reload", False, bool)
    compile = get_and_update_env(args, "compile", False, bool)
    lru_size = get_and_update_env(args, "lru_size", 64, int)
    cors_origin = get_and_update_env(args, "cors_origin", "*", str)
    no_playground = get_and_update_env(args, "no_playground", False, bool)
    no_docs = get_and_update_env(args, "no_docs", False, bool)
    half = get_and_update_env(args, "half", False, bool)
    off_tqdm = get_and_update_env(args, "off_tqdm", False, bool)
    exclude = get_and_update_env(args, "exclude", "", str)
    device_id = get_and_update_env(args, "device_id", None, str)
    use_cpu = get_and_update_env(args, "use_cpu", [], list)

    api = create_api(no_docs=no_docs, exclude=exclude.split(","))
    config.api = api

    if cors_origin:
        api.set_cors(allow_origins=[cors_origin])

    if not no_playground:
        api.setup_playground()

    if compile:
        logger.info("Model compile is enabled")

    def should_cache(*args, **kwargs):
        spk_seed = kwargs.get("spk_seed", -1)
        infer_seed = kwargs.get("infer_seed", -1)
        return spk_seed != -1 and infer_seed != -1

    generate.setup_lru_cache()
    devices.reset_device()
    devices.first_time_calculation()

    uvicorn.run(api.app, host=host, port=port, reload=reload)
