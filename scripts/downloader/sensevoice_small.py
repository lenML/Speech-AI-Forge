import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class SenseVoiceSmallDownloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "model.pt",
            "config.yaml",
            "configuration.json",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
        ]
        super().__init__(
            model_name="SenseVoiceSmall",
            modelscope_repo="iic/SenseVoiceSmall",
            huggingface_repo="FunAudioLLM/SenseVoiceSmall",
            required_files=required_files,
            just_download_required_files=False,
        )
        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    SenseVoiceSmallDownloader()(source=args.source)
