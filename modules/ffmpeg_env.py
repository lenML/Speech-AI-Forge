import logging
import os

from modules.utils.constants import ROOT_DIR

logger = logging.getLogger(__name__)


def setup_ffmpeg_path():
    ffmpeg_path = os.path.join(ROOT_DIR, "ffmpeg")
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

    import pydub.utils

    if pydub.utils.which("ffmpeg") is None:
        logger.error("ffmpeg not found in PATH")
        raise Exception("ffmpeg not found in PATH")
