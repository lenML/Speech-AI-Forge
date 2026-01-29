# 下载器的注册和管理
import logging
from pathlib import Path
import time
from typing import Literal
from modules.downloader.dl_base import RemoteModelDownloader
from modules.downloader.net import can_net_access


HF_TEST_FILE_URL = (
    "https://huggingface.co/openai-community/gpt2/raw/main/tokenizer_config.json"
)

logger = logging.getLogger(__name__)


class DownloadRegistry:
    def __init__(self):
        self.registry: "list[RemoteModelDownloader]" = []
        # 自动下载
        self.auto_download = False

    def match_model_name(self, model_name: str) -> tuple[bool, str]:
        """
        大小写不敏感，并且移除 -_ 等字符
        返回匹配到的原始 model_name 字符串
        """
        preprocess = lambda x: x.lower().replace("-", "").replace("_", "")
        model_name = preprocess(model_name)
        for downloader in self.registry:
            if preprocess(downloader.model_name) == model_name:
                return True, downloader.model_name
        return False, ""

    def register(self, downloader: RemoteModelDownloader):
        if self.is_registered(downloader):
            raise ValueError(
                f"Downloader for model {downloader.model_name} already registered."
            )
        self.registry.append(downloader)

    def is_registered(self, downloader: RemoteModelDownloader) -> bool:
        name = downloader.model_name
        return any(d.model_name == name for d in self.registry)

    def get_downloader(self, model_name: str) -> RemoteModelDownloader:
        for downloader in self.registry:
            if downloader.model_name == model_name:
                return downloader
        raise ValueError(f"No downloader registered for model: {model_name}")

    def get_model_dir_path(self, model_name: str) -> Path:
        downloader = self.get_downloader(model_name)
        return downloader.dir_path

    def download_model(
        self,
        model_name: str,
        down_source: Literal["huggingface", "modelscope", "auto"] = "auto",
        request_type: Literal["script", "webui", "api"] = "script",
    ):
        downloader = self.get_downloader(model_name)
        if downloader.check_exist():
            logger.info(f"🟢 Model [{downloader.model_name}] already exists.")
            return
        # 来自 webui 和 api 的下载需要开启自动下载
        do_download = self.auto_download or request_type == "script"
        if not do_download:
            raise ValueError(
                f"Model {model_name} not downloaded, auto_download is False."
            )

        if down_source == "auto":
            can_access_hf = can_net_access(HF_TEST_FILE_URL)
            if can_access_hf:
                down_source = "huggingface"
            else:
                print(f"Cannot access Hugging Face, will download from ModelScope.")
                down_source = "modelscope"

        downloader.download(down_source=down_source)

        logger.info(f"✅ Model [{downloader.model_name}] downloaded.")
