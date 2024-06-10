import os
import pytest

import tests

default_tts_params = {
    "text": "Hello, this is a test. [lbreak]",
    "spk": "female2",
    "style": "chat",
    "temperature": 0.3,
    "top_P": 0.5,
    "top_K": 20,
    "seed": 42,
    "format": "mp3",
    "prompt1": "",
    "prompt2": "",
    "prefix": "",
    "bs": "8",
    "thr": "100",
}


@pytest.mark.tts_api
def test_synthesize_tts(client):
    tts_params = default_tts_params.copy()

    response = client.get("/v1/tts", params=tts_params)
    print(response.content)
    assert response.status_code == 200
    assert response.headers["content-type"] in ["audio/wav", "audio/mpeg"]

    with open(
        os.path.join(tests.conftest.test_outputs_dir, "tts_api_success.mp3"),
        "wb",
    ) as f:
        f.write(response.content)


@pytest.mark.tts_api
def test_synthesize_tts_missing_text(client):
    tts_params = default_tts_params.copy()
    tts_params["text"] = ""

    response = client.get("/v1/tts", params=tts_params)

    assert response.status_code == 422


@pytest.mark.tts_api
def test_synthesize_tts_invalid_temperature(client):
    tts_params = default_tts_params.copy()
    tts_params["temperature"] = -1

    response = client.get("/v1/tts", params=tts_params)

    assert response.status_code == 422


@pytest.mark.tts_api
def test_synthesize_tts_invalid_format(client):
    tts_params = default_tts_params.copy()
    tts_params["format"] = "invalid_format"

    response = client.get("/v1/tts", params=tts_params)

    assert response.status_code == 422


@pytest.mark.tts_api
def test_synthesize_tts_large_top_p(client):
    tts_params = default_tts_params.copy()
    tts_params["top_P"] = 1.5

    response = client.get("/v1/tts", params=tts_params)

    assert response.status_code == 422


@pytest.mark.tts_api
def test_synthesize_tts_large_top_k(client):
    tts_params = default_tts_params.copy()
    tts_params["top_K"] = 1000

    response = client.get("/v1/tts", params=tts_params)

    assert response.status_code == 422
