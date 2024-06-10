from pathlib import Path

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
