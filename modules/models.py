from modules.ChatTTS import ChatTTS
import torch

from modules import config

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
    return chat_tts
