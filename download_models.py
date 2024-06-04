import os
import argparse

MODEL_DIR = "./models/ChatTTS"
CACHE_DIR = "./models/cache/"

if not os.path.exists("./models"):
    os.makedirs("./models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def download_from_modelscope():
    from modelscope import snapshot_download

    repo_id = "pzc163/chatTTS"

    snapshot_download(repo_id, cache_dir=CACHE_DIR)

    if os.system(f"mv {CACHE_DIR}/{repo_id}/* {MODEL_DIR}/") != 0:
        raise Exception("Failed to move the model to the specified directory")
    if os.system(f"rm -rf {CACHE_DIR}") != 0:
        raise Exception("Failed to remove the cache directory")

    print(f"Model downloaded from ModelScope successfully, saved at: {MODEL_DIR}")


def download_from_huggingface():
    from huggingface_hub import snapshot_download

    repo_id = "2Noise/ChatTTS"

    snapshot_download(
        repo_id, cache_dir=CACHE_DIR, local_dir=MODEL_DIR, local_dir_use_symlinks=False
    )
    if os.system(f"rm -rf {CACHE_DIR}") != 0:
        raise Exception("Failed to remove the cache directory")

    print(f"Model downloaded from HuggingFace successfully, saved at: {MODEL_DIR}")


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

    if args.source == "modelscope":
        download_from_modelscope()
    elif args.source == "huggingface":
        download_from_huggingface()


if __name__ == "__main__":
    main()
