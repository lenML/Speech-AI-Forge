#!/usr/bin/env python
import argparse
import os
import sys
import tempfile
from pathlib import Path

import torchaudio
from torch.hub import download_url_to_file
from tqdm import tqdm

from matcha.utils.data.utils import _extract_zip

URLS = {
    "en-US": {
        "female": "https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_en-US_F.zip",
        "male": "https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_en-US_M.zip",
    },
    "ja-JP": {
        "female": "https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_ja-JP_F.zip",
        "male": "https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_ja-JP_M.zip",
    },
}

INFO_PAGE = "https://ast-astrec.nict.go.jp/en/release/hi-fi-captain/"

# On their website they say "We NICT open-sourced Hi-Fi-CAPTAIN",
# but they use this very-much-not-open-source licence.
# Dunno if this is open washing or stupidity.
LICENCE = "CC BY-NC-SA 4.0"

# I'd normally put the citation here. It's on their website.
# Boo to non-open-source stuff.


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save-dir", type=str, default=None, help="Place to store the downloaded zip files")
    parser.add_argument(
        "-r",
        "--skip-resampling",
        action="store_true",
        default=False,
        help="Skip resampling the data (from 48 to 22.05)",
    )
    parser.add_argument(
        "-l", "--language", type=str, choices=["en-US", "ja-JP"], default="en-US", help="The language to download"
    )
    parser.add_argument(
        "-g",
        "--gender",
        type=str,
        choices=["male", "female"],
        default="female",
        help="The gender of the speaker to download",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="data",
        help="Place to store the converted data. Top-level only, the subdirectory will be created",
    )

    return parser.parse_args()


def process_text(infile, outpath: Path):
    outmode = "w"
    if infile.endswith("dev.txt"):
        outfile = outpath / "valid.txt"
    elif infile.endswith("eval.txt"):
        outfile = outpath / "test.txt"
    else:
        outfile = outpath / "train.txt"
        if outfile.exists():
            outmode = "a"
    with (
        open(infile, encoding="utf-8") as inf,
        open(outfile, outmode, encoding="utf-8") as of,
    ):
        for line in inf.readlines():
            line = line.strip()
            fileid, rest = line.split(" ", maxsplit=1)
            outfile = str(outpath / f"{fileid}.wav")
            of.write(f"{outfile}|{rest}\n")


def process_files(zipfile, outpath, resample=True):
    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename in tqdm(_extract_zip(zipfile, tmpdirname)):
            if not filename.startswith(tmpdirname):
                filename = os.path.join(tmpdirname, filename)
            if filename.endswith(".txt"):
                process_text(filename, outpath)
            elif filename.endswith(".wav"):
                filepart = filename.rsplit("/", maxsplit=1)[-1]
                outfile = str(outpath / filepart)
                arr, sr = torchaudio.load(filename)
                if resample:
                    arr = torchaudio.functional.resample(arr, orig_freq=sr, new_freq=22050)
                torchaudio.save(outfile, arr, 22050)
            else:
                continue


def main():
    args = get_args()

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir()

    if not args.output_dir:
        print("output directory not specified, exiting")
        sys.exit(1)

    URL = URLS[args.language][args.gender]
    dirname = f"hi-fi_{args.language}_{args.gender}"

    outbasepath = Path(args.output_dir)
    if not outbasepath.is_dir():
        outbasepath.mkdir()
    outpath = outbasepath / dirname
    if not outpath.is_dir():
        outpath.mkdir()

    resample = True
    if args.skip_resampling:
        resample = False

    if save_dir:
        zipname = URL.rsplit("/", maxsplit=1)[-1]
        zipfile = save_dir / zipname
        if not zipfile.exists():
            download_url_to_file(URL, zipfile, progress=True)
        process_files(zipfile, outpath, resample)
    else:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as zf:
            download_url_to_file(URL, zf.name, progress=True)
            process_files(zf.name, outpath, resample)


if __name__ == "__main__":
    main()
