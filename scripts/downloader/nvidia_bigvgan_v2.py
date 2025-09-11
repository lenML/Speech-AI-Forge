import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class NvidiaBigvganV2Downloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "configuration.json",
            "config.json",
            "bigvgan_generator.pt",
        ]
        super().__init__(
            model_name="nvidia/bigvgan_v2_22khz_80band_256x",
            modelscope_repo="nv-community/bigvgan_v2_22khz_80band_256x",
            huggingface_repo="nvidia/bigvgan_v2_22khz_80band_256x",
            required_files=required_files,
            just_download_required_files=True,
        )
        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    NvidiaBigvganV2Downloader()(source=args.source)
