import logging

from modules.downloader.dl_base import RemoteModelDownloader

logger = logging.getLogger(__name__)


class FunasrCampplusDownloader(RemoteModelDownloader):
    def __init__(self):
        required_files = [
            "campplus_cn_common.bin",
        ]
        super().__init__(
            model_name="funasr/campplus",
            modelscope_repo="iic/speech_campplus_sv_zh-cn_16k-common",
            huggingface_repo="funasr/campplus",
            required_files=required_files,
            just_download_required_files=True,
        )
        self.logger = logger


if __name__ == "__main__":
    from modules.downloader.dl_args import parser_args

    args = parser_args()
    FunasrCampplusDownloader()(source=args.source)
