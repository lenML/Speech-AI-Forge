import logging
import os
import urllib.request
from pathlib import Path

from tqdm import tqdm

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class F5TTSDownloader(BaseModelDownloader):
    # TODO: 验证文件完整性
    MODEL_SHA256 = "4180310f91d592cee4bc14998cd37c781f779cf105e8ca8744d9bd48ca7046ae"

    def __init__(self):
        # NOTE: 只需要下载一个文件
        required_files = [
            "F5TTS_Base/model_1200000.safetensors",
        ]
        super().__init__(
            model_name="F5-TTS",
            modelscope_repo="AI-ModelScope/F5-TTS",
            huggingface_repo="SWivid/F5-TTS",
            required_files=required_files,
            just_download_required_files=True,
        )

        self.logger = logger

    def ensure_dir(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.model_dir / "F5TTS_Base"):
            os.makedirs(self.model_dir / "F5TTS_Base")

    def from_huggingface(self):
        from huggingface_hub import hf_hub_download

        self.ensure_dir()

        for file in self.required_files:
            file_path = hf_hub_download(
                repo_id=self.huggingface_repo,
                filename=file,
                cache_dir=self.model_dir / "F5TTS_Base",
            )
            logger.info(f"Downloaded {file} from Huggingface Hub to {file_path}.")

    def from_modelscope(self):
        """从 ModelScope 下载指定文件，并显示进度条"""
        self.ensure_dir()

        # 这个是 2024 年发布的模型 版本号是 0.6
        url = "https://modelscope.cn/models/AI-ModelScope/F5-TTS/resolve/master/F5TTS_Base/model_1200000.safetensors"
        dest_path = self.model_dir / "F5TTS_Base" / "model_1200000.safetensors"

        self._download_with_progress(url, dest_path)

    def _download_with_progress(self, url: str, dest_path: Path):
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=dest_path.name
        ) as pbar:
            urllib.request.urlretrieve(url, dest_path, pbar.update_to)

        print(f"Downloaded {dest_path.name} to {dest_path}")


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    F5TTSDownloader()(source=args.source)
