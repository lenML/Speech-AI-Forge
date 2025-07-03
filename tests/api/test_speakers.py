import os

from pytest import mark

from modules.utils import constants


# 标记为参数化测试
@mark.parametrize(
    "path, method, status_code",
    [
        ("/v1/speakers/list", "GET", 200),
        ("/v1/speakers/refresh", "POST", 200),
    ],
)
@mark.speakers_api
def test_api_endpoints(client, path, method, status_code):
    response = client.request(method, path)
    assert response.status_code == status_code


@mark.speakers_api
def test_create_speaker(client):
    data = {
        "name": "测试发言人",
        "gender": "male",
        "describe": "这是一个测试发言人",
        "tensor": [0.1, 0.2, 0.3],
        "save_file": True,
    }
    response = client.post("/v1/speaker/create", json=data)
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "ok"

    filepath = os.path.join(constants.SPEAKERS_DIR, "测试发言人.spkv1.json")

    assert os.path.exists(filepath)
    os.remove(filepath)
