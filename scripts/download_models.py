import argparse

from scripts.ModelDownloader import ModelDownloader
from scripts.dl_chattts import ChatTTSDownloader
from scripts.dl_enhance import ResembleEnhanceDownloader


def main():
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

    downloaders: list[ModelDownloader] = []
    downloaders.append(ChatTTSDownloader())
    downloaders.append(ResembleEnhanceDownloader())

    for downloader in downloaders:
        if downloader.check_exist():
            print(f"Model {downloader.model_name} already exists.")
            continue

        if args.source == "modelscope":
            downloader.from_modelscope()
        elif args.source == "huggingface":
            downloader.from_huggingface()
        else:
            raise ValueError("Invalid source")

        downloader.gc()


if __name__ == "__main__":
    main()
