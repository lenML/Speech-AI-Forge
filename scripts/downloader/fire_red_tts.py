import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class FireRedTTSDownloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "fireredtts_gpt.pt",
            "fireredtts_speaker.bin",
            "fireredtts_token2wav.pt",
        ]
        super().__init__(
            model_name="FireRedTTS",
            modelscope_repo="pengzhendong/FireRedTTS",
            huggingface_repo="fireredteam/FireRedTTS",
            required_files=required_files,
        )

        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    FireRedTTSDownloader()(source=args.source)
