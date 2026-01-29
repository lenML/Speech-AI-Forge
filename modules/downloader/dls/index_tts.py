import logging

from modules.downloader.dl_base import RemoteModelDownloader

logger = logging.getLogger(__name__)


class IndexTTSDownloader(RemoteModelDownloader):
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


if __name__ == "__main__":
    from modules.downloader.dl_args import parser_args

    args = parser_args()
    IndexTTSDownloader()(source=args.source)
