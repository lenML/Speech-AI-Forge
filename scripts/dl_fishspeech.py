import logging
from scripts.dl_base import BaseModelDownloader
from scripts.download_models import parser_args

logger = logging.getLogger(__name__)


class FishSpeechDownloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "config.json",
            "firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
            "model.pth",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        super().__init__(
            model_name="FishSpeech",
            modelscope_repo="AI-ModelScope/fish-speech-1.2",
            huggingface_repo="fishaudio/fish-speech-1.2",
            required_files=required_files,
        )
        self.logger = logger


if __name__ == "__main__":
    args = parser_args()
    FishSpeechDownloader()(source=args.source)
