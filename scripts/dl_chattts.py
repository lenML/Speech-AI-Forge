import os
from scripts.ModelDownloader import ModelDownloader


class ChatTTSDownloader(ModelDownloader):
    def __init__(self):
        super().__init__()
        self.model_name = "ChatTTS"
        self.model_dir = self.dir_path / "ChatTTS"
        self.cache_dir = self.dir_path / "cache"

        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True)
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    def from_modelscope(self):
        from modelscope import snapshot_download

        repo_id = "pzc163/chatTTS"
        snapshot_download(repo_id, cache_dir=self.cache_dir)

        # Move files to the model directory
        os.system(f"mv {self.cache_dir}/{repo_id}/* {self.model_dir}/")
        self.gc()
        print(
            f"Model downloaded from ModelScope successfully, saved at: {self.model_dir}"
        )

    def from_huggingface(self):
        from huggingface_hub import snapshot_download

        repo_id = "2Noise/ChatTTS"
        snapshot_download(
            repo_id,
            cache_dir=self.cache_dir,
            local_dir=self.model_dir,
            local_dir_use_symlinks=False,
        )
        self.gc()
        print(
            f"Model downloaded from HuggingFace successfully, saved at: {self.model_dir}"
        )

    def check_exist(self) -> bool:
        return self.model_dir.exists() and any(self.model_dir.iterdir())

    def gc(self):
        os.system(f"rm -rf {self.cache_dir}")
