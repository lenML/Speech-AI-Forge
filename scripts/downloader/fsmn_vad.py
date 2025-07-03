import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class FsmnVADDownloader(BaseModelDownloader):
    """
    这个是 sensevoice 使用的 vad
    只有 2mb+ 大小，所以，其实自动下载也可以

    但是你预先下载可以跳过funasr内部的很拉跨的下载逻辑
    （funasr居然不会先检查是否有缓存，而是先下载...）
    """

    def __init__(self):
        required_files = [
            "model.pt",
            "config.yaml",
            "configuration.json",
            "am.mvn",
        ]
        super().__init__(
            model_name="fsmn-vad",
            modelscope_repo="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            huggingface_repo="funasr/fsmn-vad",
            required_files=required_files,
            just_download_required_files=True,
        )

        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    FsmnVADDownloader()(source=args.source)
