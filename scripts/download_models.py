import argparse
import time
import logging

try:
    logging.basicConfig(level=logging.INFO)
except:
    pass

from scripts.dl_chattts import ChatTTSDownloader
from scripts.dl_enhance import ResembleEnhanceDownloader
from scripts.ModelDownloader import ModelDownloader
from scripts.dl_fishspeech import FishSpeechDownloader


def parser_args():
    parser = argparse.ArgumentParser(
        description="Download model to the specified folder"
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        required=True,
        help="Choose the source to download the model from",
    )
    args = parser.parse_args()
    return args


def main():
    args = parser_args()

    downloaders: list[ModelDownloader] = []
    downloaders.append(ChatTTSDownloader())
    downloaders.append(ResembleEnhanceDownloader())
    # downloaders.append(FishSpeechDownloader())

    for downloader in downloaders:
        downloader(source=args.source)


if __name__ == "__main__":
    main()
