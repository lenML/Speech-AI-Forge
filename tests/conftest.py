import os
from pytest import fixture
from fastapi.testclient import TestClient

import fastapi

from modules.api.api_setup import create_api
from modules.devices import devices
from modules import config


config.runtime_env_vars.off_tqdm = True
devices.reset_device()
app_instance = create_api(fastapi.FastAPI())


@fixture
def client():
    yield TestClient(app_instance.app)


test_inputs_dir = os.path.dirname(__file__) + "/test_inputs"
test_outputs_dir = os.path.dirname(__file__) + "/test_outputs"

if not os.path.exists(test_outputs_dir):
    os.makedirs(test_outputs_dir)

if not os.path.exists(test_inputs_dir):
    os.makedirs(test_inputs_dir)
