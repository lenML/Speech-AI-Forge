import logging
from scripts.dl_base import BaseModelDownloader
from scripts.download_models import parser_args

logger = logging.getLogger(__name__)


class CosyVoiceInstructDownloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "campplus.onnx",
            "configuration.json",
            "cosyvoice.yaml",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "speech_tokenizer_v1.onnx",
            "spk2info.pt",
        ]
        super().__init__(
            model_name="CosyVoice_300M",
            modelscope_repo="iic/CosyVoice-300M",
            huggingface_repo="model-scope/CosyVoice-300M",
            required_files=required_files,
        )
        self.logger = logger


if __name__ == "__main__":
    args = parser_args()
    CosyVoiceInstructDownloader()(source=args.source)
