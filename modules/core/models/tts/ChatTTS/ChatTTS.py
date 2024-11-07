import gc
import logging
import threading

import torch

from modules import config
from modules.devices import devices
from modules.repos_static.ChatTTS import ChatTTS

logger = logging.getLogger(__name__)

chat_tts = None
lock = threading.Lock()


def do_load_chat_tts():
    global chat_tts
    if chat_tts:
        return

    logger.info("Loading ChatTTS models")
    chat_tts = ChatTTS.Chat()

    device = devices.get_device_for("chattts")
    dtype = devices.dtype

    has_loaded = chat_tts.load(
        compile=config.runtime_env_vars.compile,
        use_flash_attn=config.runtime_env_vars.flash_attn,
        use_vllm=config.runtime_env_vars.vllm,
        source="custom",
        custom_path="./models/ChatTTS",
        device=device,
    )

    if not has_loaded:
        chat_tts = None
        raise Exception("Failed to load ChatTTS models, please check the log")

    all_modules: list[torch.nn.Module] = [
        chat_tts.gpt,
        chat_tts.dvae,
        chat_tts.decoder,
        chat_tts.vocos,
    ]
    for md in all_modules:
        md.to(device=device, dtype=dtype)

    # 如果 device 为 cpu 同时，又是 dtype == float16 那么报 warn
    # 提示可能无法正常运行，建议使用 float32 即开启 `--no_half` 参数
    if device == devices.cpu and dtype == torch.float16:
        logger.warning(
            "The device is CPU and dtype is float16, which may not work properly. It is recommended to use float32 by enabling the `--no_half` parameter."
        )

    devices.torch_gc()
    logger.info("ChatTTS models loaded")


@devices.after_gc()
def load_chat_tts():
    with lock:
        do_load_chat_tts()
    if chat_tts is None:
        raise Exception("Failed to load ChatTTS models")
    return chat_tts


@devices.after_gc()
def unload_chat_tts():
    logging.info("Unloading ChatTTS models")
    global chat_tts

    if chat_tts:
        chat_tts.unload()
    chat_tts = None
    logger.info("ChatTTS models unloaded")


@devices.after_gc()
def reload_chat_tts():
    logging.info("Reloading ChatTTS models")
    unload_chat_tts()
    instance = load_chat_tts()
    logger.info("ChatTTS models reloaded")
    return instance


def get_tokenizer():
    chat_tts = load_chat_tts()
    tokenizer = chat_tts.tokenizer._tokenizer
    if not tokenizer:
        raise Exception("Failed to load tokenizer")
    return tokenizer
