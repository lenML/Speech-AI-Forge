import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class FasterWhisperDownloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "model.bin",
            "tokenizer.json",
            "vocabulary.json",
            "preprocessor_config.json",
            "config.json",
        ]
        super().__init__(
            model_name="faster-whisper-large-v3-turbo-ct2",
            modelscope_repo="pengzhendong/faster-whisper-large-v3-turbo",
            huggingface_repo="deepdml/faster-whisper-large-v3-turbo-ct2",
            required_files=required_files,
        )

        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    FasterWhisperDownloader()(source=args.source)
