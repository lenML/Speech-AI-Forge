import logging

from modules.downloader.dl_base import RemoteModelDownloader

logger = logging.getLogger(__name__)


class IndexTTS15Downloader(RemoteModelDownloader):
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
            model_name="Index-TTS-1.5",
            modelscope_repo="IndexTeam/IndexTTS-1.5",
            huggingface_repo="IndexTeam/IndexTTS-1.5",
            required_files=required_files,
            just_download_required_files=True,
        )
        self.logger = logger


if __name__ == "__main__":
    from modules.downloader.dl_args import parser_args

    args = parser_args()
    IndexTTS15Downloader()(source=args.source)
