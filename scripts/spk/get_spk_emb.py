import argparse
import base64
import lzma

import numpy as np
import numpy.typing as npt
import pybase16384 as b14
import torch

from modules.core.spk.TTSSpeaker import TTSSpeaker


def parse_args():
    parser = argparse.ArgumentParser(description="Get spk emb from spk file")
    parser.add_argument(
        "--spk",
        required=True,
        help="Speaker file path",
    )
    parser.add_argument(
        "--model_id",
        default="chat-tts",
        help="Model ID",
    )
    parser.add_argument(
        "--format",
        default="base64",
        choices=["base64", "b64", "b14"],
        help="Output format",
    )
    parser.add_argument(
        "--out",
        help="Output file path",
    )
    args = parser.parse_args()
    return args


def encode_to_b64(tensor: torch.Tensor) -> str:
    ndarr: npt.NDArray = tensor.cpu().float().detach().numpy()
    byts: bytes = ndarr.tobytes()
    return base64.b64encode(byts).decode("utf-8")


def encode_to_b14(spk_emb: torch.Tensor) -> str:
    arr: np.ndarray = spk_emb.to(dtype=torch.float16, device="cpu").numpy()
    s = b14.encode_to_string(
        lzma.compress(
            arr.tobytes(),
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
        ),
    )
    del arr
    return s


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
    此脚本用于将 spk_emb 输出为 字符串

    NOTE:  这个脚本只支持取第一个 token
    """
    args = parse_args()

    spk = TTSSpeaker.from_file(args.spk)

    token = spk.get_token(args.model_id)

    if token is None:
        print(f"No token found in spk file: {args.spk}")
        exit(1)

    emb: torch.Tensor = token.tokens[0]

    output = ""

    if args.format == "base64":
        output = encode_to_b64(emb)
    elif args.format == "b64":
        output = encode_to_b64(emb)
    elif args.format == "b14":
        output = encode_to_b14(emb)
    else:
        print(f"Invalid format: {args.format}")
        exit(1)

    if args.out is None:
        print(output)
    else:
        with open(args.out, "w") as f:
            f.write(token)

    # decoded = _decode_b14_str(output)
    # print(f"Decoded: {decoded.shape}")
