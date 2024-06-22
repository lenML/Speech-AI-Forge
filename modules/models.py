import gc
import logging
import threading

import torch

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
    chat_tts.load_models(
        compile=config.runtime_env_vars.compile,
        source="local",
        local_path="./models/ChatTTS",
        device=devices.get_device_for("chattts"),
        dtype=devices.dtype,
        dtype_vocos=devices.dtype_vocos,
        dtype_dvae=devices.dtype_dvae,
        dtype_gpt=devices.dtype_gpt,
        dtype_decoder=devices.dtype_decoder,
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
