import requests

from scripts.ModelDownloader import ModelDownloader


class ResembleEnhanceDownloader(ModelDownloader):
    def __init__(self):
        super().__init__()
        self.model_name = "resemble-enhance"
        self.model_dir = self.dir_path / "resemble-enhance"

        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True)
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True)

    def from_modelscope(self):
        url = "https://modelscope.cn/api/v1/studio/insummer/ResembleEnhance/repo?Revision=master&FilePath=resemble_enhance%2Fmodel_repo%2Fenhancer_stage2%2Fds%2FG%2Fdefault%2Fmp_rank_00_model_states.pt"
        self._download_file(url, self.model_dir / "mp_rank_00_model_states.pt")
        print(
            f"Model downloaded from ModelScope successfully, saved at: {self.model_dir}"
        )

    def from_huggingface(self):
        url = "https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt?download=true"
        self._download_file(url, self.model_dir / "mp_rank_00_model_states.pt")
        print(
            f"Model downloaded from HuggingFace successfully, saved at: {self.model_dir}"
        )

    def _download_file(self, url, dest_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    def check_exist(self) -> bool:
        return self.model_dir.exists() and any(self.model_dir.iterdir())

    def gc(self):
        pass  # No cache to clear for direct download
