from pathlib import Path
import time

MODEL_DIR = Path("models")


class ModelDownloader:
    def __init__(self):
        self.model_name = "<no-name>"
        self.dir_path = MODEL_DIR

    def from_modelscope(self):
        raise NotImplementedError()

    def from_huggingface(self):
        raise NotImplementedError()

    def check_exist(self) -> bool:
        return NotImplementedError()

    def gc(self):
        raise NotImplementedError()

    def __call__(self, source: str):
        self.execate(downloader=self, source=source)

    @staticmethod
    def execate(*, downloader: "ModelDownloader", source: str):
        if downloader.check_exist():
            print(f"Model {downloader.model_name} already exists.")
            return

        if source == "modelscope":
            downloader.from_modelscope()
        elif source == "huggingface":
            downloader.from_huggingface()
        else:
            raise ValueError("Invalid source")

        # after check
        times = 5
        for i in range(times):
            if downloader.check_exist():
                break
            time.sleep(5)
            if i == times - 1:
                raise TimeoutError("Download timeout")

        downloader.gc()
