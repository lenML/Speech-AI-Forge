import sys

import torch

from modules.utils import ffmpeg, git
from modules.utils.JsonObject import JsonObject

# TODO impl RuntimeEnvVars() class
runtime_env_vars = JsonObject({})

auto_gc = True

api = None

versions = JsonObject(
    {
        "python_version": ".".join([str(x) for x in sys.version_info[0:3]]),
        "torch_version": getattr(torch, "__long_version__", torch.__version__),
        # "gradio_version":gr.__version__,
        "git_tag": git.git_tag(),
        "git_branch": git.branch_name(),
        "git_commit": git.commit_hash(),
        "ffmpeg_version": ffmpeg.ffmpeg_version(),
    }
)
