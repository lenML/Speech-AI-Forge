import logging

from modules.downloader.dl_base import RemoteModelDownloader

logger = logging.getLogger(__name__)


class MaskGCTSemanticDownloader(RemoteModelDownloader):
    def __init__(self):
        required_files = [
            "semantic_codec/model.safetensors",
        ]
        super().__init__(
            model_name="amphion/MaskGCT",
            modelscope_repo="amphion/MaskGCT",
            huggingface_repo="amphion/MaskGCT",
            required_files=required_files,
            just_download_required_files=True,
        )
        self.logger = logger


if __name__ == "__main__":
    from modules.downloader.dl_args import parser_args

    args = parser_args()
    MaskGCTSemanticDownloader()(source=args.source)
