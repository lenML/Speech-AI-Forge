import argparse
import os

import fastapi
from fastapi.testclient import TestClient
from pytest import fixture

try:
    from modules.repos_static.sys_paths import setup_repos_paths

    # NOTE: 需要在 api_setup 之前调用
    setup_repos_paths()
except Exception:
    pass

from modules import config
from modules.api.api_setup import create_api
from modules.devices import devices
from modules.models_setup import process_model_args, setup_model_args

parser = argparse.ArgumentParser(description="Test")
setup_model_args(parser)
args, unknown = parser.parse_known_args()
process_model_args(args)
config.runtime_env_vars.off_tqdm = True
devices.reset_device()
app_instance = create_api(fastapi.FastAPI())
app_instance.set_cors()

@fixture
def client():
    yield TestClient(app_instance.app)


test_inputs_dir = os.path.dirname(__file__) + "/test_inputs"
test_outputs_dir = os.path.dirname(__file__) + "/test_outputs"

if not os.path.exists(test_outputs_dir):
    os.makedirs(test_outputs_dir)

if not os.path.exists(test_inputs_dir):
    os.makedirs(test_inputs_dir)
