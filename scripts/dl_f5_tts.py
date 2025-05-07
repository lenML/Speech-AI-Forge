import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class F5TTSDownloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "F5TTS_Base/model_1200000.safetensors",
        ]
        super().__init__(
            model_name="F5-TTS",
            modelscope_repo="AI-ModelScope/F5-TTS",
            huggingface_repo="SWivid/F5-TTS",
            required_files=required_files,
        )

        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    F5TTSDownloader()(source=args.source)
