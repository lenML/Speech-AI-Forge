import argparse

import soundfile as sf

from modules.core.spk.TTSSpeaker import TTSSpeaker


def parse_args():
    parser = argparse.ArgumentParser(description="Edit TTSSpeaker data")
    parser.add_argument(
        "--spk",
        required=True,
        help="Speaker file path",
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
    此脚本用于检查 spk 文件的 wav 音频信息

    NOTE: 暂时只检查第一个，后续补充其他参数
    """
    args = parse_args()

    spk = TTSSpeaker.from_file(args.spk)

    sr, wav, text = spk.get_ref_wav()

    print(f"{wav.shape[0] / sr} seconds")
    print(f"{wav.shape[0]} samples")
    print(f"{sr} kz")
    print(f"Text: {text}")

    sf.write(args.out, wav, sr * 2)
