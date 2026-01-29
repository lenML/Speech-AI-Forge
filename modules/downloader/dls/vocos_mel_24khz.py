import logging

from modules.downloader.dl_base import RemoteModelDownloader

logger = logging.getLogger(__name__)


class VocosMel24khzDownloader(RemoteModelDownloader):
    def __init__(self):
        required_files = [
            "config.yaml",
            "pytorch_model.bin",
        ]
        super().__init__(
            model_name="vocos-mel-24khz",
            modelscope_repo="pengzhendong/vocos-mel-24khz",
            huggingface_repo="charactr/vocos-mel-24khz",
            required_files=required_files,
        )

        self.logger = logger


if __name__ == "__main__":
    from modules.downloader.dl_args import parser_args

    args = parser_args()
    VocosMel24khzDownloader()(source=args.source)
