import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class MaskGCTSemanticDownloader(BaseModelDownloader):
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
    from scripts.dl_args import parser_args

    args = parser_args()
    MaskGCTSemanticDownloader()(source=args.source)
