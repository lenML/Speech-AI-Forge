import argparse
import logging

from modules import generate_audio
from modules.devices import devices
from modules.Enhancer.ResembleEnhance import load_enhancer
from modules.models import load_chat_tts
from modules.utils import env


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
        choices=["all", "chattts", "enhancer", "trainer"],
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
    parser.add_argument(
        "--preload_models",
        action="store_true",
        help="Preload all models at startup",
    )


def process_model_args(args: argparse.Namespace):
    lru_size = env.get_and_update_env(args, "lru_size", 64, int)
    compile = env.get_and_update_env(args, "compile", False, bool)
    device_id = env.get_and_update_env(args, "device_id", None, str)
    use_cpu = env.get_and_update_env(args, "use_cpu", [], list)
    no_half = env.get_and_update_env(args, "no_half", False, bool)
    off_tqdm = env.get_and_update_env(args, "off_tqdm", False, bool)
    debug_generate = env.get_and_update_env(args, "debug_generate", False, bool)
    preload_models = env.get_and_update_env(args, "preload_models", False, bool)

    generate_audio.setup_lru_cache()
    devices.reset_device()
    devices.first_time_calculation()

    if debug_generate:
        generate_audio.logger.setLevel(logging.DEBUG)

    if preload_models:
        load_chat_tts()
        load_enhancer()
