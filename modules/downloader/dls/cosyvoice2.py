import logging

from modules.downloader.dl_base import RemoteModelDownloader

logger = logging.getLogger(__name__)


class CosyVoice2Downloader(RemoteModelDownloader):
    def __init__(self):
        required_files = [
            "campplus.onnx",
            "configuration.json",
            "cosyvoice2.yaml",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "speech_tokenizer_v2.onnx",
            "CosyVoice-BlankEN/model.safetensors",
            "CosyVoice-BlankEN/config.json",
            "CosyVoice-BlankEN/generation_config.json",
            "CosyVoice-BlankEN/merges.txt",
            "CosyVoice-BlankEN/tokenizer_config.json",
            "CosyVoice-BlankEN/vocab.json",
        ]
        super().__init__(
            model_name="CosyVoice2-0.5B",
            modelscope_repo="iic/CosyVoice2-0.5B",
            huggingface_repo="FunAudioLLM/CosyVoice2-0.5B",
            required_files=required_files,
            just_download_required_files=True,
        )
        self.logger = logger

        self.blank_dir = self.model_dir / "CosyVoice-BlankEN"
        self.blank_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    from modules.downloader.dl_args import parser_args

    args = parser_args()
    CosyVoice2Downloader()(source=args.source)
