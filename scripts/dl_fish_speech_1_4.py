import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class FishSpeech14Downloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "config.json",
            "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            "model.pth",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        super().__init__(
            model_name="fish-speech-1_4",
            modelscope_repo="AI-ModelScope/fish-speech-1.4",
            huggingface_repo="fishaudio/fish-speech-1.4",
            required_files=required_files,
        )

        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    FishSpeech14Downloader()(source=args.source)
