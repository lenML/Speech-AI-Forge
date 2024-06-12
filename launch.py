import os
import logging

from modules.api.api_setup import setup_api_args, setup_model_args, setup_uvicon_args
from modules.ffmpeg_env import setup_ffmpeg_path

setup_ffmpeg_path()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

import argparse
import uvicorn

from modules import config
from modules.utils import env

from fastapi import FastAPI

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv(
        dotenv_path=os.getenv("ENV_FILE", ".env.api"),
    )
    parser = argparse.ArgumentParser(
        description="Start the FastAPI server with command line arguments"
    )
    setup_api_args(parser)
    setup_model_args(parser)
    setup_uvicon_args(parser=parser)

    args = parser.parse_args()

    host = env.get_and_update_env(args, "host", "0.0.0.0", str)
    port = env.get_and_update_env(args, "port", 7870, int)
    reload = env.get_and_update_env(args, "reload", False, bool)
    workers = env.get_and_update_env(args, "workers", 1, int)
    log_level = env.get_and_update_env(args, "log_level", "info", str)
    access_log = env.get_and_update_env(args, "access_log", True, bool)
    proxy_headers = env.get_and_update_env(args, "proxy_headers", True, bool)
    timeout_keep_alive = env.get_and_update_env(args, "timeout_keep_alive", 5, int)
    timeout_graceful_shutdown = env.get_and_update_env(
        args, "timeout_graceful_shutdown", 0, int
    )
    ssl_keyfile = env.get_and_update_env(args, "ssl_keyfile", None, str)
    ssl_certfile = env.get_and_update_env(args, "ssl_certfile", None, str)
    ssl_keyfile_password = env.get_and_update_env(
        args, "ssl_keyfile_password", None, str
    )

    uvicorn.run(
        "modules.api.worker:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        access_log=access_log,
        proxy_headers=proxy_headers,
        timeout_keep_alive=timeout_keep_alive,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_keyfile_password=ssl_keyfile_password,
    )
