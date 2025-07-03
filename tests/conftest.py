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
from modules.core.models.zoo.ModelZoo import model_zoo
from modules.devices import devices
from modules.models_setup import process_model_args, setup_model_args

parser = argparse.ArgumentParser(description="Test")
setup_model_args(parser)
args, unknown = parser.parse_known_args()
process_model_args(args)
config.runtime_env_vars.off_tqdm = True
# 使用cpu测试
# config.runtime_env_vars.use_cpu = "all"
devices.reset_device()
app_instance = create_api(fastapi.FastAPI())
app_instance.set_cors()

@fixture
def client():
    yield TestClient(app_instance.app)


@fixture(
    autouse=True,
    scope="module",
)
def after_each_test():
    yield  # 等待测试完成
    # NOTE: 测试机器 vram 不足的情况，可以开启这个 fixture
    # NOTE: 全部模型加载大概要 16gb 显存左右，开启卸载之后基本在 8gb 左右
    # 清空模型
    try:
        model_zoo.unload_all_models()
    except Exception as e:
        pass


test_inputs_dir = os.path.dirname(__file__) + "/test_inputs"
test_outputs_dir = os.path.dirname(__file__) + "/test_outputs"

if not os.path.exists(test_outputs_dir):
    os.makedirs(test_outputs_dir)

if not os.path.exists(test_inputs_dir):
    os.makedirs(test_inputs_dir)
