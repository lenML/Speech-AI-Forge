import os
from pytest import fixture, mark, raises
from fastapi.testclient import TestClient
from modules.api.impl.openai_api import AudioSpeechRequest

from launch import create_api

import tests.conftest

app_instance = create_api()


@fixture
def client():
    yield TestClient(app_instance.app)


@mark.parametrize(
    "input_text, voice",
    [
        ("Hello, world", "female2"),
        ("Test text", "Alice"),
        ("Invalid voice", "unknown_voice"),
    ],
)
def test_openai_speech_api(client, input_text, voice):
    request = AudioSpeechRequest(input=input_text, voice=voice)
    response = client.post("/v1/audio/speech", json=request.model_dump())

    if voice == "unknown_voice":
        assert response.status_code == 400
        assert "Invalid voice" in response.json().get("detail", "")
    else:
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "audio/mp3"
        with open(
            os.path.join(
                tests.conftest.test_outputs_dir,
                f"{input_text.replace(' ', '_')}_{voice}.mp3",
            ),
            "wb",
        ) as f:
            f.write(response.content)


def test_openai_speech_api_with_invalid_style(client):
    request = AudioSpeechRequest(
        input="Test text", voice="female2", style="invalid_style"
    )
    response = client.post("/v1/audio/speech", json=request.model_dump())

    assert response.status_code == 400
    assert "Invalid style" in response.json().get("detail", "")


# def test_transcribe_not_implemented(client):
#     file = {"file": ("test.wav", b"test audio data")}
#     response = client.post("/v1/audio/transcriptions", files=file)

#     assert response.status_code == 200
#     assert response.json() == success_response("not implemented yet")


# TODO
# @mark.parametrize("file_name, file_content", [("test.wav", b"test audio data")])
# def test_transcribe_with_file(client, file_name, file_content):
#     file = {"file": (file_name, file_content)}
#     response = client.post("/v1/audio/transcriptions", files=file)

#     assert response.status_code == 200
#     assert isinstance(response.json(), TranscriptionsVerboseResponse)
#     assert response.json().text == "not implemented yet"
