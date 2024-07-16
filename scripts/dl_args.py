import argparse


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
