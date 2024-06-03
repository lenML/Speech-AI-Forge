import torch
from modules import config
from modules import generate_audio as generate
from modules.api import api

from functools import lru_cache
from typing import Callable

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")


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

    parser = argparse.ArgumentParser(
        description="Start the FastAPI server with command line arguments"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
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
        default="*",
        help="Allowed CORS origins. Use '*' to allow all origins.",
    )

    args = parser.parse_args()

    config.args = args

    if args.compile:
        print("Model compile is enabled")
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

    api.set_cors()

    uvicorn.run(api.app, host=args.host, port=args.port, reload=args.reload)
