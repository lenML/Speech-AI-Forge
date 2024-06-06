import torch
from modules.ChatTTS import ChatTTS
from modules import config
from modules.devices import devices

import logging

logger = logging.getLogger(__name__)
chat_tts = None


def load_chat_tts():
    global chat_tts
    if chat_tts:
        return chat_tts

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

    return chat_tts


def reload_chat_tts():
    logging.info("Reloading ChatTTS models")
    global chat_tts
    if chat_tts:
        if torch.cuda.is_available():
            for model_name, model in chat_tts.pretrain_models.items():
                if isinstance(model, torch.nn.Module):
                    model.cpu()
            torch.cuda.empty_cache()
    chat_tts = None
    return load_chat_tts()
