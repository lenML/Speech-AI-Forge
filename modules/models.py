import threading
import torch
from modules.ChatTTS import ChatTTS
from modules import config
from modules.devices import devices

import logging
import gc

logger = logging.getLogger(__name__)

chat_tts = None
# 某些平台上，不让在主线程中加载模型，否则会出现错误
# huggingface Error:
# RuntimeError: CUDA must not be initialized in the main process on Spaces with Stateless GPU environment.
# You can look at this Stacktrace to find out which part of your code triggered a CUDA init
load_event = threading.Event()


def load_chat_tts_in_thread():
    global chat_tts
    if chat_tts:
        load_event.set()
        return

    logger.info("Loading ChatTTS models")
    chat_tts = ChatTTS.Chat()
    chat_tts.load_models(
        compile=config.runtime_env_vars.compile,
        source="local",
        local_path="./models/ChatTTS",
        device=devices.device,
        dtype=devices.dtype,
        dtype_vocos=devices.dtype_vocos,
        dtype_dvae=devices.dtype_dvae,
        dtype_gpt=devices.dtype_gpt,
        dtype_decoder=devices.dtype_decoder,
    )

    devices.torch_gc()
    load_event.set()
    logger.info("ChatTTS models loaded")


def initialize_chat_tts():
    model_thread = threading.Thread(target=load_chat_tts_in_thread)
    model_thread.start()
    return model_thread


def load_chat_tts():
    if chat_tts is None:
        initialize_chat_tts().join()
    load_event.wait()
    return chat_tts


def unload_chat_tts():
    logging.info("Unloading ChatTTS models")
    global chat_tts

    if chat_tts:
        for model_name, model in chat_tts.pretrain_models.items():
            if isinstance(model, torch.nn.Module):
                model.cpu()
                del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gc.collect()
    chat_tts = None
    logger.info("ChatTTS models unloaded")


def reload_chat_tts():
    logging.info("Reloading ChatTTS models")
    unload_chat_tts()
    instance = load_chat_tts()
    logger.info("ChatTTS models reloaded")
    return instance
