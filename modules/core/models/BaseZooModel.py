from modules.devices import devices
from modules.downloader.AutoModelDownloader import AutoModelDownloader


# NOTE: model_id 和 model_name 可能相同，比如一个模型只有一种变体的时候

class BaseZooModel:

    def __init__(self, model_id: str, model_name: str = None) -> None:
        # NOTE: model_id 指的是模型架构
        self.model_id = model_id
        # NOTE: model_name 指的是权重名
        self.model_name = model_name or model_id
        self.hash = ""

    def get_device(self):
        return devices.get_device_for(self.model_id)

    def get_downloader(self):
        adl = AutoModelDownloader()
        return adl.registry.get_downloader(model_name=self.model_name)

    def get_dtype(self):
        return devices.dtype

    def reset(self) -> None:
        """
        重置推理上下文
        """
        pass

    def is_loaded(self) -> bool:
        return False

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def download(self) -> None:
        pass

    def interrupt(self) -> None:
        pass

    def is_downloaded(self) -> bool:
        """
        检查模型是否已经安装 比如在 webui 页面是否显示
        """
        dl = self.get_downloader()
        return dl.check_exist()
