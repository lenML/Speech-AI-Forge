import os
from modules.devices import devices
from modules.utils import env
from modules.webui import webui_config
from modules.webui.app import webui_init, create_interface
from modules import generate_audio
from modules import config

if __name__ == "__main__":
    import argparse
    import dotenv

    dotenv.load_dotenv(
        dotenv_path=os.getenv("ENV_FILE", ".env.webui"),
    )

    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument("--server_name", type=str, help="server name")
    parser.add_argument("--server_port", type=int, help="server port")
    parser.add_argument(
        "--share", action="store_true", help="share the gradio interface"
    )
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--auth", type=str, help="username:password for authentication")
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
    parser.add_argument(
        "--lru_size",
        type=int,
        default=64,
        help="Set the size of the request cache pool, set it to 0 will disable lru_cache",
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
    parser.add_argument("--compile", action="store_true", help="Enable model compile")
    # webui_Experimental
    parser.add_argument(
        "--webui_experimental",
        action="store_true",
        help="Enable webui_experimental features",
    )

    args = parser.parse_args()

    def get_and_update_env(*args):
        val = env.get_env_or_arg(*args)
        key = args[1]
        config.runtime_env_vars[key] = val
        return val

    server_name = get_and_update_env(args, "server_name", "0.0.0.0", str)
    server_port = get_and_update_env(args, "server_port", 7860, int)
    share = get_and_update_env(args, "share", False, bool)
    debug = get_and_update_env(args, "debug", False, bool)
    auth = get_and_update_env(args, "auth", None, str)
    half = get_and_update_env(args, "half", False, bool)
    off_tqdm = get_and_update_env(args, "off_tqdm", False, bool)
    lru_size = get_and_update_env(args, "lru_size", 64, int)
    device_id = get_and_update_env(args, "device_id", None, str)
    use_cpu = get_and_update_env(args, "use_cpu", [], list)
    compile = get_and_update_env(args, "compile", False, bool)
    webui_experimental = get_and_update_env(args, "webui_experimental", False, bool)

    webui_config.tts_max = get_and_update_env(args, "tts_max_len", 1000, int)
    webui_config.ssml_max = get_and_update_env(args, "ssml_max_len", 5000, int)
    webui_config.max_batch_size = get_and_update_env(args, "max_batch_size", 8, int)

    demo = create_interface()

    if auth:
        auth = tuple(auth.split(":"))

    generate_audio.setup_lru_cache()
    devices.reset_device()
    devices.first_time_calculation()

    webui_init()

    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug,
        auth=auth,
        show_api=False,
    )
