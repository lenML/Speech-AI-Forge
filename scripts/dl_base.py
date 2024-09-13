import os
import shutil
import sys

from scripts.ModelDownloader import ModelDownloader


class BaseModelDownloader(ModelDownloader):
    def __init__(
        self, model_name, modelscope_repo=None, huggingface_repo=None, required_files=[]
    ):
        super().__init__()
        self.model_name = model_name
        self.modelscope_repo = modelscope_repo
        self.huggingface_repo = huggingface_repo
        self.required_files = required_files
        self.model_dir = self.dir_path / model_name
        self.cache_dir = self.dir_path / "cache"

        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True)
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    def from_modelscope(self):
        if self.modelscope_repo is None:
            raise Exception(
                "This downloader does not support downloading from ModelScope."
            )

        from modelscope import snapshot_download

        snapshot_download(self.modelscope_repo, cache_dir=self.cache_dir)

        # Move files to the model directory
        from_path = self.cache_dir / self.modelscope_repo
        to_path = self.model_dir
        if sys.platform == "win32":
            for item in from_path.glob("*"):
                try:
                    shutil.move(str(item), str(to_path))
                except Exception as e:
                    print(f"Failed to move {item}: {e}")
        else:
            os.system(f"mv {from_path}/* {to_path}/")
        self.logger.info(
            f"Model downloaded from ModelScope successfully, saved at: {self.model_dir}"
        )

    def from_huggingface(self):
        if self.huggingface_repo is None:
            raise Exception(
                "This downloader does not support downloading from HuggingFace."
            )

        from huggingface_hub import snapshot_download

        snapshot_download(
            self.huggingface_repo,
            cache_dir=self.cache_dir,
            local_dir=self.model_dir,
            local_dir_use_symlinks=False,
        )
        self.logger.info(
            f"Model downloaded from HuggingFace successfully, saved at: {self.model_dir}"
        )

    def check_exist(self) -> bool:
        if not self.model_dir.exists():
            return False
        for file in self.required_files:
            if not (self.model_dir / file).exists():
                self.logger.info(f"Missing file: {file}")
                return False
        return True

    def gc(self):
        try:
            shutil.rmtree(self.cache_dir)
            self.logger.info(f"Cache directory {self.cache_dir} deleted successfully.")
        except Exception as e:
            self.logger.error(f"Failed to delete cache directory {self.cache_dir}: {e}")
