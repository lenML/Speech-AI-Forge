from modules.ChatTTS import ChatTTS
import torch

from modules import config

import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device use {device}")

chat_tts = None


def load_chat_tts():
    global chat_tts
    if chat_tts:
        return chat_tts
    chat_tts = ChatTTS.Chat()
    chat_tts.load_models(
        compile=config.enable_model_compile,
        source="local",
        local_path="./models/ChatTTS",
        device=device,
    )

    if config.model_config.get("half", False):
        logging.info("half precision enabled")
        for model_name, model in chat_tts.pretrain_models.items():
            if isinstance(model, torch.nn.Module):
                model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model.half()
                if torch.cuda.is_available():
                    model.cuda()
                model.eval()
                logger.log(logging.INFO, f"{model_name} converted to half precision.")

    return chat_tts
