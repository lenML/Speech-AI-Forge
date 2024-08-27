import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class ChatTTSDownloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "asset/DVAE.pt",
            "asset/DVAE_full.pt",
            "asset/Decoder.pt",
            "asset/GPT.pt",
            "asset/Vocos.pt",
            "asset/spk_stat.pt",
            "asset/tokenizer.pt",
            "asset/tokenizer/special_tokens_map.json",
            "asset/tokenizer/tokenizer.json",
            "asset/tokenizer/tokenizer_config.json",
            "config/decoder.yaml",
            "config/dvae.yaml",
            "config/gpt.yaml",
            "config/path.yaml",
            "config/vocos.yaml",
        ]
        super().__init__(
            model_name="ChatTTS",
            modelscope_repo="AI-ModelScope/ChatTTS",
            huggingface_repo="2Noise/ChatTTS",
            required_files=required_files,
        )

        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    ChatTTSDownloader()(source=args.source)
