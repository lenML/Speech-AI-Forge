import logging

from modules.downloader.dl_base import RemoteModelDownloader

logger = logging.getLogger(__name__)


class Qwen3TTS12hz06BBaseDownloader(RemoteModelDownloader):
    def __init__(self):
        super().__init__(
            model_name="Qwen3-TTS-12Hz-0.6B-Base",
            modelscope_repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            huggingface_repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            required_files=["model.safetensors", "speech_tokenizer/model.safetensors"],
        )
        self.logger = logger


if __name__ == "__main__":
    from modules.downloader.dl_args import parser_args

    args = parser_args()
    Qwen3TTS12hz06BBaseDownloader()(source=args.source)
