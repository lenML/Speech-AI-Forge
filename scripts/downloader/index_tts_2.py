import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class IndexTTSV2Downloader(BaseModelDownloader):
    def __init__(self):
        # 只对几个大文件进行验证，因为所有文件都需要下载
        required_files = [
            "gpt.pth",
            "s2mel.pth",
            "qwen0.6bemo4-merge/model.safetensors",
        ]
        super().__init__(
            model_name="Index-TTS-2",
            modelscope_repo="IndexTeam/IndexTTS-2",
            huggingface_repo="IndexTeam/IndexTTS-2",
            required_files=required_files,
            just_download_required_files=False,
        )
        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    IndexTTSV2Downloader()(source=args.source)
