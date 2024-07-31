import argparse
import os

from pydub import AudioSegment

from modules.core.spk.dcls import DcSpkReference, DcSpkSample
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.utils.audio_utils import pydub_to_np


def parse_spk_edit_args():
    parser = argparse.ArgumentParser(description="Edit TTSSpeaker data")
    parser.add_argument(
        "--spk",
        required=True,
        help="Speaker file path",
    )
    parser.add_argument(
        "--set-name",
        help="Set speaker name",
    )
    parser.add_argument(
        "--set-gender",
        help="Set speaker gender",
    )
    parser.add_argument(
        "--set-author",
        help="Set speaker author",
    )
    parser.add_argument(
        "--set-desc",
        help="Set speaker description",
    )
    parser.add_argument(
        "--set-version",
        help="Set speaker version",
    )
    # parser.add_argument(
    #     "--add-token",
    #     nargs=3,
    #     metavar=("MODEL_ID", "TOKEN_LIST", "MODEL_HASH"),
    #     help="Add a token for the speaker. Provide model ID, token list (comma-separated), and optional model hash.",
    # )
    parser.add_argument(
        "--add-ref",
        nargs=3,
        metavar=("AUDIO_PATH", "TRANSCRIPT_OR_PATH", "EMOTION"),
        help="Add a reference audio with transcript. Provide audio file path and transcript.",
    )
    parser.add_argument(
        "--add-sample",
        nargs=2,
        metavar=("AUDIO_PATH", "TRANSCRIPT_OR_PATH"),
        help="Add a sample audio with transcript. Provide audio file path and transcript.",
    )
    parser.add_argument(
        "--output",
        help="Output file path for the modified speaker. Defaults to overwriting the input file.",
        default=None,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_spk_edit_args()

    spk = TTSSpeaker.from_file(args.spk)

    if args.set_name:
        spk.set_name(args.set_name)
    if args.set_gender:
        spk.set_gender(args.set_gender)
    if args.set_author:
        spk.set_author(args.set_author)
    if args.set_desc:
        spk.set_desc(args.set_desc)
    if args.set_version:
        spk.set_version(args.set_version)
    # if args.add_token:
    #     model_id, token_list, model_hash = args.add_token
    #     tokens = [int(t) for t in token_list.split(",")]
    #     spk.set_token(tokens=tokens, model_id=model_id, model_hash=model_hash)
    if args.add_ref:
        audio_path, transcript_or_path, emotion = args.add_ref
        transcript = transcript_or_path
        if os.path.isfile(transcript_or_path):
            with open(transcript_or_path, "r", encoding="utf-8") as f:
                transcript = f.read()
        audio_segment: AudioSegment = AudioSegment.from_file(audio_path)
        sr, ndarray = pydub_to_np(audio_segment)
        ref_obj = DcSpkReference(
            text=transcript,
            wav=ndarray.tobytes(),
            wav_sr=sr,
            emotion=emotion,
        )
        spk.add_ref(ref=ref_obj)
    if args.add_sample:
        audio_path, transcript_or_path = args.add_sample
        transcript = transcript_or_path
        if os.path.isfile(transcript_or_path):
            with open(transcript_or_path, "r", encoding="utf-8") as f:
                transcript = f.read()
        audio_segment: AudioSegment = AudioSegment.from_file(audio_path)
        sr, ndarray = pydub_to_np(audio_segment)
        spk.add_sample(
            sample=DcSpkSample(
                text=transcript,
                wav=ndarray.tobytes(),
                wav_sr=sr,
            )
        )

    output_path = (
        args.output
        if args.output
        else os.path.splitext(args.spk)[0] + "_edited.spkv1.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json_str = spk.to_json_str()
        f.write(json_str)

    print(f"Speaker data saved to {output_path}")


if __name__ == "__main__":
    main()

"""
examples:
- 添加 ref

python -m scripts.spk.edit --spk ./test.spkv1.json --add-ref output.wav "Hello world" "*"

- 修改名字 （json随便改，之后支持png就用脚本方便点）

python -m scripts.spk.edit --spk ./test.spkv1.json --set-name "test speaker name"
"""
