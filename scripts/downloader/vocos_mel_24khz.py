import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class VocosMel24khzDownloader(BaseModelDownloader):
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
    from scripts.dl_args import parser_args

    args = parser_args()
    VocosMel24khzDownloader()(source=args.source)
