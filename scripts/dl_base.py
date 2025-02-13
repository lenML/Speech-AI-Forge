import os
import shutil
import sys
from typing import Optional

from scripts.ModelDownloader import ModelDownloader


class BaseModelDownloader(ModelDownloader):

    def __init__(
        self,
        model_name,
        modelscope_repo=None,
        huggingface_repo=None,
        required_files=[],
        modelscope_revision="master",
        huggingface_revision="main",
        just_download_required_files=False,
        ignore_patterns=[".gitattributes"],
    ):
        super().__init__()
        self.model_name = model_name
        self.modelscope_repo: Optional[str] = modelscope_repo
        self.huggingface_repo: Optional[str] = huggingface_repo
        self.modelscope_revision = modelscope_revision
        self.huggingface_revision = huggingface_revision
        self.required_files = required_files
        self.model_dir = self.dir_path / model_name
        self.cache_dir = self.dir_path / "cache"
        self.just_download_required_files = just_download_required_files
        self.ignore_patterns = ignore_patterns

        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True)
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    def from_modelscope_just_requires(self):
        from modelscope import model_file_download

        for file in self.required_files:
            try:
                downloaded_file = model_file_download(
                    model_id=self.modelscope_repo,
                    file_path=file,
                    revision=self.modelscope_revision,
                    cache_dir=str(self.cache_dir),
                )
                target_path = self.model_dir / file
                shutil.copy2(downloaded_file, target_path)
                os.remove(downloaded_file)
                self.logger.info(f"Downloaded {file} from ModelScope.")
            except Exception as e:
                self.logger.error(f"Failed to download {file} from ModelScope: {e}")

    def from_modelscope_repo(self):
        from modelscope import snapshot_download

        snapshot_download(self.modelscope_repo, cache_dir=self.cache_dir)
        # Move files to the model directory
        from_path = self.cache_dir / self.modelscope_repo.replace(".", "___")
        to_path = self.model_dir
        if sys.platform == "win32":
            for item in from_path.glob("*"):
                try:
                    shutil.move(str(item), str(to_path))
                except Exception as e:
                    print(f"Failed to move {item}: {e}")
        else:
            os.system(f"mv {from_path}/* {to_path}/")

    def from_modelscope(self):
        if self.modelscope_repo is None:
            raise Exception(
                "This downloader does not support downloading from ModelScope."
            )

        if self.just_download_required_files:
            self.from_modelscope_just_requires()
        else:
            self.from_modelscope_repo()

        self.logger.info(
            f"Model downloaded from ModelScope successfully, saved at: {self.model_dir}"
        )

    def from_huggingface_repo(self):
        from huggingface_hub import snapshot_download

        snapshot_download(
            self.huggingface_repo,
            cache_dir=self.cache_dir,
            local_dir=self.model_dir,
            local_dir_use_symlinks=False,
            force_download=True,
            ignore_patterns=self.ignore_patterns,
        )

    def from_huggingface_just_requires(self):
        from huggingface_hub import hf_hub_download

        for file in self.required_files:
            try:
                downloaded_file = hf_hub_download(
                    repo_id=self.huggingface_repo,
                    filename=file,
                    revision=self.huggingface_revision,
                    cache_dir=str(self.cache_dir),
                    force_download=True,
                    local_dir_use_symlinks=False,
                )
                target_path = self.model_dir / file
                shutil.copy2(downloaded_file, target_path)
                os.remove(downloaded_file)
                self.logger.info(f"Downloaded {file} from HuggingFace.")
            except Exception as e:
                self.logger.error(f"Failed to download {file} from HuggingFace: {e}")

    def from_huggingface(self):
        if self.huggingface_repo is None:
            raise Exception(
                "This downloader does not support downloading from HuggingFace."
            )
        if self.just_download_required_files:
            self.from_huggingface_just_requires()
        else:
            self.from_huggingface_repo()

        self.logger.info(
            f"Model downloaded from HuggingFace successfully, saved at: {self.model_dir}"
        )

    # 移除文件重新下载，huggingface会创建软连接导致检测的时候以为已经下载了...
    def remove_files(self):
        try:
            shutil.rmtree(self.model_dir)
            shutil.rmtree(self.cache_dir)
        except Exception as e:
            self.logger.error(f"Failed to delete cache directory {self.cache_dir}: {e}")

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
