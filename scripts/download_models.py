import argparse
import logging

from modules.downloader.AutoModelDownloader import AutoModelDownloader

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
        choices=["modelscope", "ms", "huggingface", "hf"],
        required=True,
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
    md = AutoModelDownloader(down_source=args.source)
    md.download_models(model_names=args.models.split(","), request_type="script")
