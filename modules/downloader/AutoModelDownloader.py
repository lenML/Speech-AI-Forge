from fnmatch import fnmatch
import inspect
from typing import Literal
from modules.downloader.dl_base import RemoteModelDownloader, TModelDownloader
from modules.downloader.dl_registry import DownloadRegistry

import modules.downloader.dls as dls

from modules import config

def get_default_down_source():
    # TODO: 根据配置文件选择默认下载源
    return "auto"


class AutoModelDownloader:

    def __init__(
        self,
        down_source: Literal[
            "huggingface", "modelscope", "auto"
        ] = get_default_down_source(),
    ):
        self.registry = DownloadRegistry()
        self.down_source = down_source

        self.__init_all_dls__()

    def __init_all_dls__(self):
        for [name, obj] in inspect.getmembers(dls):
            if (
                obj is TModelDownloader
                or obj is TModelDownloader
                or name.startswith("_")
            ):
                continue
            if inspect.isclass(obj) and issubclass(obj, TModelDownloader):
                dl = obj()
                self.registry.register(dl)
                # print(f"Registered downloader: {dl.model_name}")

    def can_auto_download(self, model_name: str):
        """
        根据 auto_download 配置，判断是否可以自动下载

        auto_download=False 关闭自动下载
        auto_download=* 通配符，表示所有模型都可以自动下载
        auto_download=model_name 指定模型名称，表示只有该模型可以自动下载
        auto_download=model_* 模式匹配，表示所有以 model_ 开头的模型都可以自动下载
        """
        auto_download = config.runtime_env_vars["auto_download"]
        if auto_download is False or auto_download is None or auto_download == "False":
            return False
        if auto_download == "*":
            return True
        if not isinstance(auto_download, str):
            return False
        # 如果包含逗号，表示多个模型名称
        for name in auto_download.split(","):
            if fnmatch(model_name, name):
                return True
        return False

    def is_downloaded(self, model_name: str):
        downloader = self.registry.get_downloader(model_name=model_name)
        if downloader is None:
            return False
        return downloader.check_exist()

    def download(self, model_name: str, force=False):
        downloader = self.registry.get_downloader(model_name=model_name)
        if downloader is None:
            raise ValueError(f"Model name not found: {model_name}")
        model_exist = downloader.check_exist()
        if (
            not force
            and not model_exist
            and not self.can_auto_download(model_name=model_name)
        ):
            raise ValueError(
                f"Model not exist and auto_download is False: {model_name}"
            )

        self.registry.download_model(
            model_name=model_name,
            down_source=self.down_source,
        )
        return self.registry.get_model_dir_path(model_name=model_name)

    def download_models(self, model_names: list[str], force=False):
        names = []
        for name in model_names:
            matched, model_name = self.registry.match_model_name(name)
            if matched:
                names.append(model_name)
            else:
                raise ValueError(f"Model name not found: {name}")
        return [
            self.download(model_name=model_name, force=force) for model_name in names
        ]


if __name__ == "__main__":
    downloader = AutoModelDownloader()
    # 列出所有 model_name
    print(downloader.registry.list_model_names())
