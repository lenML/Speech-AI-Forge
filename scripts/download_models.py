import logging

try:
    logging.basicConfig(level=logging.INFO)
except:
    pass

from scripts.dl_args import parser_args
from scripts.dl_chattts import ChatTTSDownloader
from scripts.dl_enhance import ResembleEnhanceDownloader
from scripts.ModelDownloader import ModelDownloader


def main():
    args = parser_args()

    downloaders: list[ModelDownloader] = []
    downloaders.append(ChatTTSDownloader())
    downloaders.append(ResembleEnhanceDownloader())

    for downloader in downloaders:
        downloader(source=args.source)


if __name__ == "__main__":
    main()
