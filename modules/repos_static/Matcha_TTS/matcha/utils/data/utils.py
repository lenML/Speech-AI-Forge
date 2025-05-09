# taken from https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/utils.py
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
# Licence: BSD 2-Clause
# pylint: disable=C0123

import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Union

_LG = logging.getLogger(__name__)


def _extract_tar(from_path: Union[str, Path], to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    if type(from_path) is Path:
        from_path = str(Path)

    if to_path is None:
        to_path = os.path.dirname(from_path)

    with tarfile.open(from_path, "r") as tar:
        files = []
        for file_ in tar:  # type: Any
            file_path = os.path.join(to_path, file_.name)
            if file_.isfile():
                files.append(file_path)
                if os.path.exists(file_path):
                    _LG.info("%s already extracted.", file_path)
                    if not overwrite:
                        continue
            tar.extract(file_, to_path)
        return files


def _extract_zip(from_path: Union[str, Path], to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    if type(from_path) is Path:
        from_path = str(Path)

    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as zfile:
        files = zfile.namelist()
        for file_ in files:
            file_path = os.path.join(to_path, file_)
            if os.path.exists(file_path):
                _LG.info("%s already extracted.", file_path)
                if not overwrite:
                    continue
            zfile.extract(file_, to_path)
    return files
