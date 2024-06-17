import os

cwd = os.getcwd()
utils_path = os.path.dirname(os.path.realpath(__file__))
modules_path = os.path.dirname(utils_path)

ROOT_DIR = os.path.dirname(modules_path)
DATA_DIR = os.path.join(ROOT_DIR, "data")

MODELS_DIR = os.path.join(ROOT_DIR, "models")

SPEAKERS_DIR = os.path.join(DATA_DIR, "speakers")
