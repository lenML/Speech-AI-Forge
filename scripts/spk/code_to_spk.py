import argparse
import lzma

import numpy as np
import pybase16384 as b14
import torch

from modules.core.spk.TTSSpeaker import TTSSpeaker


def parse_args():
    parser = argparse.ArgumentParser(description="Code to SPK")
    parser.add_argument(
        "--code",
        required=True,
        help="Speaker code",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output file path",
    )
    args = parser.parse_args()
    return args


def _decode_b14_str(b14_str: str) -> np.ndarray:
    return np.frombuffer(
        lzma.decompress(
            b14.decode_from_string(b14_str),
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
        ),
        dtype=np.float16,
    ).copy()


if __name__ == "__main__":
    """
    这个脚本从 b14 code 中生成 spk

    例子:
    python -m scripts.spk.code_to_spk --code "啊吧啊吧" --out "test.spkv1.json"
    """

    args = parse_args()
    code = args.code
    out = args.out
    spk = TTSSpeaker.empty()
    token = _decode_b14_str(code)
    token = torch.from_numpy(token)
    spk.set_token(tokens=[token], model_id="chat-tts")

    with open(out, "wb") as f:
        json_str = spk.to_json_str()
        f.write(json_str.encode("utf-8"))
