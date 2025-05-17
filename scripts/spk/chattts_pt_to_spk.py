import argparse

import torch

from modules.core.spk.TTSSpeaker import TTSSpeaker


def parse_args():
    parser = argparse.ArgumentParser(description="ChatTTS voice file to SPK")
    parser.add_argument(
        "--file",
        required=True,
        help="ChatTTS Voice *.pt file path",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output file path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    这个脚本用于将 ChatTTS 中适用的 pt 文件转换位 spkv1.json

    例子:
    python -m scripts.spk.chattts_pt_to_spk --file "seed_181_restored_emb.pt" --out "seed_181.spkv1.json"
    """

    args = parse_args()
    file = args.file
    out = args.out

    spk_stat: torch.Tensor = torch.load(
        file,
        map_location=torch.device("cpu"),
    )

    spk = TTSSpeaker.from_token("chat-tts", [spk_stat])

    with open(out, "wb") as f:
        json_str = spk.to_json_str()
        f.write(json_str.encode("utf-8"))
