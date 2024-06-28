import gc
import logging
import threading

import torch
from transformers import LlamaTokenizer

from modules import config
from modules.ChatTTS import ChatTTS
from modules.devices import devices

logger = logging.getLogger(__name__)

chat_tts = None
lock = threading.Lock()


def load_chat_tts_in_thread():
    global chat_tts
    if chat_tts:
        return

    logger.info("Loading ChatTTS models")
    chat_tts = ChatTTS.Chat()
    device = devices.get_device_for("chattts")
    dtype = devices.dtype
    chat_tts.load(
        compile=config.runtime_env_vars.compile,
        source="custom",
        custom_path="./models/ChatTTS",
        device=device,
        dtype=dtype,
        # dtype_vocos=devices.dtype_vocos,
        # dtype_dvae=devices.dtype_dvae,
        # dtype_gpt=devices.dtype_gpt,
        # dtype_decoder=devices.dtype_decoder,
    )

    # 如果 device 为 cpu 同时，又是 dtype == float16 那么报 warn
    # 提示可能无法正常运行，建议使用 float32 即开启 `--no_half` 参数
    if device == devices.cpu and dtype == torch.float16:
        logger.warning(
            "The device is CPU and dtype is float16, which may not work properly. It is recommended to use float32 by enabling the `--no_half` parameter."
        )

    devices.torch_gc()
    logger.info("ChatTTS models loaded")


def load_chat_tts():
    with lock:
        if chat_tts is None:
            load_chat_tts_in_thread()
    if chat_tts is None:
        raise Exception("Failed to load ChatTTS models")
    return chat_tts


def unload_chat_tts():
    logging.info("Unloading ChatTTS models")
    global chat_tts

    if chat_tts:
        for model_name, model in chat_tts.pretrain_models.items():
            if isinstance(model, torch.nn.Module):
                model.cpu()
                del model
    chat_tts = None
    devices.torch_gc()
    gc.collect()
    logger.info("ChatTTS models unloaded")


def reload_chat_tts():
    logging.info("Reloading ChatTTS models")
    unload_chat_tts()
    instance = load_chat_tts()
    logger.info("ChatTTS models reloaded")
    return instance


def get_tokenizer() -> LlamaTokenizer:
    chat_tts = load_chat_tts()
    tokenizer = chat_tts.pretrain_models["tokenizer"]
    return tokenizer
