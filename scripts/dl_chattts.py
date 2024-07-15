import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class ChatTTSDownloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "asset/DVAE.pt",
            "asset/Decoder.pt",
            "asset/GPT.pt",
            "asset/Vocos.pt",
            "asset/spk_stat.pt",
            "asset/tokenizer.pt",
            "config/decoder.yaml",
            "config/dvae.yaml",
            "config/gpt.yaml",
            "config/path.yaml",
            "config/vocos.yaml",
        ]
        super().__init__(
            model_name="ChatTTS",
            modelscope_repo="pzc163/chatTTS",
            huggingface_repo="2Noise/ChatTTS",
            required_files=required_files,
        )

        self.logger = logger
