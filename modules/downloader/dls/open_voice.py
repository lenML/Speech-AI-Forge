import logging

from modules.downloader.dl_base import RemoteModelDownloader

logger = logging.getLogger(__name__)


class OpenVoiceDownloader(RemoteModelDownloader):
    def __init__(self):
        required_files = [
            "converter/checkpoint.pth",
            "converter/config.json",
        ]
        super().__init__(
            model_name="OpenVoiceV2",
            modelscope_repo="myshell-ai/OpenVoiceV2",
            huggingface_repo="myshell-ai/OpenVoiceV2",
            required_files=required_files,
        )

        self.logger = logger


if __name__ == "__main__":
    from modules.downloader.dl_args import parser_args

    args = parser_args()
    OpenVoiceDownloader()(source=args.source)
