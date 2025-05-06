import logging

from scripts.dl_base import BaseModelDownloader

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
            model_name="fish-speech-1.2-sft",
            modelscope_repo="fishaudio/fish-speech-1.2-sft",
            huggingface_repo="fishaudio/fish-speech-1.2-sft",
            required_files=required_files,
        )

        self.logger = logger


if __name__ == "__main__":
    print(
        """
        此模型已不支持，请使用 1.4 版本
        """
    )

    from scripts.dl_args import parser_args

    args = parser_args()
    FishSpeechDownloader()(source=args.source)
