import logging

from scripts.dl_base import BaseModelDownloader

logger = logging.getLogger(__name__)


class GptSoVitsV4Downloader(BaseModelDownloader):
    def __init__(self):
        required_files = [
            "chinese-hubert-base/config.json",
            "chinese-hubert-base/preprocessor_config.json",
            "chinese-hubert-base/pytorch_model.bin",
            "chinese-roberta-wwm-ext-large/config.json",
            "chinese-roberta-wwm-ext-large/tokenizer.json",
            "chinese-roberta-wwm-ext-large/pytorch_model.bin",
            "gsv-v4-pretrained/s2Gv4.pth",
            "gsv-v4-pretrained/vocoder.pth",
            "s1v3.ckpt",
        ]
        super().__init__(
            model_name="gpt_sovits_v4",
            modelscope_repo="AI-ModelScope/GPT-SoVITS",
            huggingface_repo="lj1995/GPT-SoVITS",
            required_files=required_files,
            just_download_required_files=True,
        )

        self.logger = logger

    def extra_data_prepare(self):
        import nltk

        nltk.download(
            ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"]
        )


if __name__ == "__main__":
    from scripts.dl_args import parser_args

    args = parser_args()
    GptSoVitsV4Downloader()(source=args.source)
