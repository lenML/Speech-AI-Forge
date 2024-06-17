import os
import shutil
import sys

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
        from_path = self.cache_dir / repo_id
        to_path = self.model_dir
        if sys.platform == "win32":
            for item in from_path.glob("*"):
                try:
                    shutil.move(str(item), str(to_path))
                except Exception as e:
                    print(f"Failed to move {item}: {e}")
        else:
            os.system(f"mv {from_path}/* {to_path}/")
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
        print(
            f"Model downloaded from HuggingFace successfully, saved at: {self.model_dir}"
        )

    def check_exist(self) -> bool:
        if not self.model_dir.exists():
            return False
        asset_dir_files = [
            "DVAE.pt",
            "Decoder.pt",
            "GPT.pt",
            "Vocos.pt",
            "spk_stat.pt",
            "tokenizer.pt",
        ]
        config_dir_files = [
            "decoder.yaml",
            "dvae.yaml",
            "gpt.yaml",
            "path.yaml",
            "vocos.yaml",
        ]
        for file in asset_dir_files:
            if not (self.model_dir / "asset" / file).exists():
                print(f"Missing file: {file}")
                return False
        for file in config_dir_files:
            if not (self.model_dir / "config" / file).exists():
                print(f"Missing file: {file}")
                return False

        return True

    def gc(self):
        os.system(f"rm -rf {self.cache_dir}")
