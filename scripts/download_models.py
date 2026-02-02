import argparse
import logging

from modules.downloader.AutoModelDownloader import AutoModelDownloader
from modules.downloader.dl_registry import HF_TEST_FILE_URL
from modules.downloader.net import can_net_access

try:
    logging.basicConfig(level=logging.INFO)
except:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download model to the specified folder"
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "ms", "huggingface", "hf", "auto"],
        # 默认 auto 自动根据网络环境选择，首选 huggingface
        default="auto",
        help="Choose the source to download the model from",
    )
    # models 可以列出模型名字 逗号分割
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="The models to download, separated by commas",
    )
    args = parser.parse_args()
    if args.source == "ms":
        args.source = "modelscope"
    if args.source == "hf":
        args.source = "huggingface"
    if args.source == "auto":
        can_access_hf = can_net_access(HF_TEST_FILE_URL)
        args.source = "huggingface" if can_access_hf else "modelscope"
    md = AutoModelDownloader(down_source=args.source)
    md.download_models(model_names=args.models.split(","), force=True)
