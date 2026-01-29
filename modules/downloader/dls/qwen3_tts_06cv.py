import logging

from modules.downloader.dl_base import RemoteModelDownloader

logger = logging.getLogger(__name__)


class Qwen3TTS12hz06BCustomVoiceDownloader(RemoteModelDownloader):
    def __init__(self):
        super().__init__(
            model_name="Qwen3-TTS-12Hz-0.6B-CustomVoice",
            modelscope_repo="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            huggingface_repo="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            required_files=["model.safetensors", "speech_tokenizer/model.safetensors"],
        )
        self.logger = logger


if __name__ == "__main__":
    from modules.downloader.dl_args import parser_args

    args = parser_args()
    Qwen3TTS12hz06BCustomVoiceDownloader()(source=args.source)
