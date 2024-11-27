import os

import pytest
from fastapi.testclient import TestClient

import tests
from modules.api.impl.tts_api import TTSParams

default_tts_params = TTSParams(
    text="Hello, world!",
)


@pytest.mark.tts_api
def test_synthesize_tts(client: TestClient):
    tts_params = default_tts_params.model_copy()

    response = client.get("/v1/tts", params=tts_params.model_dump())
    assert response.status_code == 200
    assert response.headers["content-type"] in ["audio/wav", "audio/mpeg"]

    output_file = os.path.join(tests.conftest.test_outputs_dir, "tts_api_success.mp3")

    with open(
        output_file,
        "wb",
    ) as f:
        f.write(response.content)

    assert os.path.getsize(output_file) > 0, "Stream output file is empty"


@pytest.mark.tts_api
def test_long_same_text_synthesize_tts(client: TestClient):
    tts_params = default_tts_params.model_copy()
    tts_params.text = tts_params.text * 12

    response = client.get("/v1/tts", params=tts_params.model_dump())
    assert response.status_code == 200
    assert response.headers["content-type"] in ["audio/wav", "audio/mpeg"]


@pytest.mark.tts_api
def test_synthesize_tts_missing_text(client: TestClient):
    tts_params = default_tts_params.model_copy()
    tts_params.text = ""

    response = client.get("/v1/tts", params=tts_params.model_dump())

    assert response.status_code == 422


@pytest.mark.tts_api
def test_synthesize_tts_invalid_temperature(client: TestClient):
    tts_params = default_tts_params.model_copy()
    tts_params.temperature = -1

    response = client.get("/v1/tts", params=tts_params.model_dump())

    assert response.status_code == 422


@pytest.mark.tts_api
def test_synthesize_tts_invalid_format(client: TestClient):
    tts_params = default_tts_params.model_copy()
    tts_params.format = "invalid_format"

    response = client.get("/v1/tts", params=tts_params.model_dump())

    assert response.status_code == 422


@pytest.mark.tts_api
def test_synthesize_tts_large_top_p(client: TestClient):
    tts_params = default_tts_params.model_copy()
    tts_params.top_p = 1.5

    response = client.get("/v1/tts", params=tts_params.model_dump())

    assert response.status_code == 422


@pytest.mark.tts_api
def test_synthesize_tts_large_top_k(client: TestClient):
    tts_params = default_tts_params.model_copy()
    tts_params.top_k = 1000

    response = client.get("/v1/tts", params=tts_params.model_dump())

    assert response.status_code == 422


@pytest.mark.tts_api
def test_adjust_tts_generate(client: TestClient):
    tts_params = default_tts_params.model_copy()
    tts_params.text = "Hello, world! I am a test case."
    tts_params.speed = 1.5

    response = client.get("/v1/tts", params=tts_params.model_dump())
    assert response.status_code == 200
    assert response.headers["content-type"] in ["audio/wav", "audio/mpeg"]

    with open(
        os.path.join(tests.conftest.test_outputs_dir, "tts_api_adjust_success.mp3"),
        "wb",
    ) as f:
        f.write(response.content)


@pytest.mark.tts_api
@pytest.mark.stream_api
def test_stream_tts_generate(client: TestClient):
    tts_params = default_tts_params.model_copy()
    tts_params.text = "Hello, world! I am a test case."
    tts_params.stream = True

    response = client.get("/v1/tts", params=tts_params.model_dump())
    assert response.status_code == 200
    assert response.headers["content-type"] in ["audio/wav", "audio/mpeg"]

    output_file = os.path.join(
        tests.conftest.test_outputs_dir, "tts_api_stream_success.mp3"
    )
    with open(
        output_file,
        "wb",
    ) as f:
        f.write(response.content)

    assert os.path.getsize(output_file) > 0, "Stream output file is empty"
