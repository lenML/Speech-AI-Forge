import sys
import os

COSYVOICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./cosyvoice"))

paths = [COSYVOICE_DIR]


def setup_repos_paths():
    for pth in paths:
        if pth not in sys.path:
            sys.path.insert(0, pth)
