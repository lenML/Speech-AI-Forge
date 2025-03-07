import logging

import requests
from tqdm import tqdm

from scripts.ModelDownloader import ModelDownloader

logger = logging.getLogger(__name__)


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
        url = "https://modelscope.cn/models/dragonlittle/resemble-enhance/resolve/master/mp_rank_00_model_states.pt"
        self._download_file(url, self.model_dir / "mp_rank_00_model_states.pt")
        logger.info(
            f"Model downloaded from ModelScope successfully, saved at: {self.model_dir}"
        )

    def from_huggingface(self):
        url = "https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt?download=true"
        self._download_file(url, self.model_dir / "mp_rank_00_model_states.pt")
        logger.info(
            f"Model downloaded from HuggingFace successfully, saved at: {self.model_dir}"
        )

    def _download_file(self, url, dest_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192  # 8 Kibibytes

        with open(dest_path, "wb") as file, tqdm(
            total=total_size, unit="iB", unit_scale=True, desc=dest_path.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                file.write(chunk)
                bar.update(len(chunk))

    def check_exist(self) -> bool:
        model_file = self.model_dir / "mp_rank_00_model_states.pt"
        return self.model_dir.exists() and model_file.exists()

    def gc(self):
        pass  # No cache to clear for direct download


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    ResembleEnhanceDownloader()(source=args.source)
