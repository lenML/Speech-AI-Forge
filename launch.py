import logging
import os

from modules.ffmpeg_env import setup_ffmpeg_path
from modules.repos_static.sys_paths import setup_repos_paths

try:
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

import uvicorn

from modules.api.api_setup import setup_api_args
from modules.models_setup import setup_model_args
from modules.utils import env
from modules.utils.ignore_warn import ignore_useless_warnings

ignore_useless_warnings()

logger = logging.getLogger(__name__)


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


def process_uvicon_args(args):
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


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv(
        dotenv_path=os.getenv("ENV_FILE", ".env.api"),
    )
    parser = argparse.ArgumentParser(
        description="Start the FastAPI server with command line arguments"
    )
    # NOTE: 主进程中不需要处理 model args / api args，但是要接收这些参数, 具体处理在 worker.py 中
    setup_api_args(parser=parser)
    setup_model_args(parser=parser)
    setup_uvicon_args(parser=parser)

    args = parser.parse_args()

    process_uvicon_args(args)
