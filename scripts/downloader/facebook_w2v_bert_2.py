import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class W2vBert2Downloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "preprocessor_config.json",
            "config.json",
            "model.safetensors",
        ]
        super().__init__(
            model_name="facebook/w2v-bert-2.0",
            modelscope_repo="facebook/w2v-bert-2.0",
            huggingface_repo="facebook/w2v-bert-2.0",
            required_files=required_files,
            just_download_required_files=True,
        )
        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    W2vBert2Downloader()(source=args.source)
