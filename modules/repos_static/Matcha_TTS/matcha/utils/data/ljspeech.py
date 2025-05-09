#!/usr/bin/env python
import argparse
import random
import tempfile
from pathlib import Path

from torch.hub import download_url_to_file

from matcha.utils.data.utils import _extract_tar

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

INFO_PAGE = "https://keithito.com/LJ-Speech-Dataset/"

LICENCE = "Public domain (LibriVox copyright disclaimer)"

CITATION = """
@misc{ljspeech17,
  author       = {Keith Ito and Linda Johnson},
  title        = {The LJ Speech Dataset},
  howpublished = {\\url{https://keithito.com/LJ-Speech-Dataset/}},
  year         = 2017
}
"""


def decision():
    return random.random() < 0.98


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save-dir", type=str, default=None, help="Place to store the downloaded zip files")
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="data",
        help="Place to store the converted data (subdirectory LJSpeech-1.1 will be created)",
    )

    return parser.parse_args()


def process_csv(ljpath: Path):
    if (ljpath / "metadata.csv").exists():
        basepath = ljpath
    elif (ljpath / "LJSpeech-1.1" / "metadata.csv").exists():
        basepath = ljpath / "LJSpeech-1.1"
    csvpath = basepath / "metadata.csv"
    wavpath = basepath / "wavs"

    with (
        open(csvpath, encoding="utf-8") as csvf,
        open(basepath / "train.txt", "w", encoding="utf-8") as tf,
        open(basepath / "val.txt", "w", encoding="utf-8") as vf,
    ):
        for line in csvf.readlines():
            line = line.strip()
            parts = line.split("|")
            wavfile = str(wavpath / f"{parts[0]}.wav")
            if decision():
                tf.write(f"{wavfile}|{parts[1]}\n")
            else:
                vf.write(f"{wavfile}|{parts[1]}\n")


def main():
    args = get_args()

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir()

    outpath = Path(args.output_dir)
    if not outpath.is_dir():
        outpath.mkdir()

    if save_dir:
        tarname = URL.rsplit("/", maxsplit=1)[-1]
        tarfile = save_dir / tarname
        if not tarfile.exists():
            download_url_to_file(URL, str(tarfile), progress=True)
        _extract_tar(tarfile, outpath)
        process_csv(outpath)
    else:
        with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=True) as zf:
            download_url_to_file(URL, zf.name, progress=True)
            _extract_tar(zf.name, outpath)
            process_csv(outpath)


if __name__ == "__main__":
    main()
