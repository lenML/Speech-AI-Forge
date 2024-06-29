import os

import pytest
from fastapi.testclient import TestClient

import tests

default_tts_params = {
    "text": "Hello, world! [lbreak]",
    "speaker_wav": "female2",
    "language": "en",
}


@pytest.mark.xtts_v2
def test_synthesize_tts_to_audio(client: TestClient):
    response = client.post("/v1/xtts_v2/tts_to_audio", json=default_tts_params)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"

    with open(
        os.path.join(
            tests.conftest.test_outputs_dir,
            f"xtts_v2_api_success.mp3",
        ),
        "wb",
    ) as f:
        f.write(response.content)


@pytest.mark.xtts_v2
def test_synthesize_tts_to_audio_stream(client: TestClient):
    response = client.get(
        "/v1/xtts_v2/tts_stream",
        params=default_tts_params,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"

    with open(
        os.path.join(
            tests.conftest.test_outputs_dir,
            f"xtts_v2_api_stream_success.mp3",
        ),
        "wb",
    ) as f:
        f.write(response.content)


@pytest.mark.xtts_v2
def test_set_tts_settings(client: TestClient):
    tts_settings = {
        "stream_chunk_size": 100,
        "temperature": 0.3,
        "speed": 1,
        "length_penalty": 0.5,
        "repetition_penalty": 1.0,
        "top_p": 0.7,
        "top_k": 20,
        "enable_text_splitting": True,
    }
    response = client.post("/v1/xtts_v2/set_tts_settings", json=tts_settings)
    assert response.status_code == 200
    assert response.json() == {"message": "Settings successfully applied"}


@pytest.mark.xtts_v2
def test_invalid_speaker_id(client: TestClient):
    invalid_params = default_tts_params.copy()
    invalid_params["speaker_wav"] = "invalid_speaker_id"

    response = client.post("/v1/xtts_v2/tts_to_audio", json=invalid_params)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid speaker id"}


@pytest.mark.xtts_v2
def test_speakers(client: TestClient):
    response = client.get("/v1/xtts_v2/speakers")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert all("name" in spk and "voice_id" in spk for spk in response.json())


@pytest.mark.xtts_v2
def test_set_invalid_tts_settings(client: TestClient):
    invalid_settings = {
        "stream_chunk_size": -1,
        "temperature": -0.1,
        "speed": -1,
        "length_penalty": -0.5,
        "repetition_penalty": -1.0,
        "top_p": -0.7,
        "top_k": -20,
        "enable_text_splitting": True,
    }
    response = client.post("/v1/xtts_v2/set_tts_settings", json=invalid_settings)
    assert response.status_code == 400
    assert "detail" in response.json()
