import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class CosyVoice2Downloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "bigvgan_discriminator.pth",
            "bigvgan_generator.pth",
            "bpe.model",
            "config.yaml",
            "dvae.pth",
            "gpt.pth",
            "unigram_12000.vocab",
        ]
        super().__init__(
            model_name="Index-TTS",
            modelscope_repo="IndexTeam/Index-TTS",
            huggingface_repo="IndexTeam/Index-TTS",
            required_files=required_files,
            just_download_required_files=True,
        )
        self.logger = logger

        self.blank_dir = self.model_dir / "Index-TTS"
        self.blank_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    CosyVoice2Downloader()(source=args.source)
