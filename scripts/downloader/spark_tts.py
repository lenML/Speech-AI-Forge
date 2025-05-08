import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class SparkTTSDownloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "config.yaml",
            "BiCodec/model.safetensors",
            "BiCodec/config.yaml",
            "LLM/model.safetensors",
            "LLM/config.yaml",
            "LLM/tokenizer.json",
            "LLM/tokenizer_config.json",
            "LLM/vocab.json",
            "wav2vec2-large-xlsr-53/config.json",
            "wav2vec2-large-xlsr-53/pytorch_model.bin",
        ]
        super().__init__(
            model_name="Spark-TTS-0.5B",
            modelscope_repo="SparkAudio/Spark-TTS-0.5B",
            huggingface_repo="SparkAudio/Spark-TTS-0.5B",
            required_files=required_files,
            # 全部下载 required_files 用于验证几个关键文件
            just_download_required_files=False,
        )
        self.logger = logger


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    SparkTTSDownloader()(source=args.source)
