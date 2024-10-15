import os
import sys

REPO_DIR = lambda name: os.path.abspath(os.path.join(os.path.dirname(__file__), name))

paths = [
    REPO_DIR("cosyvoice"),
    REPO_DIR("openvoice"),
    REPO_DIR("fish_speech"),
    REPO_DIR("FireRedTTS"),
    REPO_DIR("F5TTS"),
]


def setup_repos_paths():
    for pth in paths:
        if pth not in sys.path:
            sys.path.insert(0, pth)
