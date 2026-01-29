import inspect
from typing import Literal
from modules.downloader.dl_base import RemoteModelDownloader, TModelDownloader
from modules.downloader.dl_registry import DownloadRegistry

import modules.downloader.dls as dls


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

    def download(
        self,
        model_name: str,
        request_type: Literal["script", "webui", "api"] = "script",
    ):
        self.registry.download_model(
            model_name=model_name,
            down_source=self.down_source,
            request_type=request_type,
        )
        return self.registry.get_model_dir_path(model_name=model_name)

    def download_models(
        self,
        model_names: list[str],
        request_type: Literal["script", "webui", "api"] = "script",
    ):
        names = []
        for name in model_names:
            matched, model_name = self.registry.match_model_name(name)
            if matched:
                names.append(model_name)
            else:
                raise ValueError(f"Model name not found: {name}")
        return [
            self.download(model_name=model_name, request_type=request_type)
            for model_name in names
        ]
