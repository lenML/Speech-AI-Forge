import os
from modules.api.impl.openai_api import AudioSpeechRequest

import tests.conftest
import pytest


@pytest.mark.parametrize(
    "input_text, voice",
    [
        ("Hello, world [lbreak]", "female2"),
        ("Test text [lbreak]", "Alice"),
        ("Invalid voice", "unknown_voice"),
    ],
)
@pytest.mark.openai_api
def test_openai_speech_api(client, input_text, voice):
    request = AudioSpeechRequest(input=input_text, voice=voice)
    response = client.post("/v1/audio/speech", json=request.model_dump())

    if voice == "unknown_voice":
        assert response.status_code == 400
        assert "Invalid voice" in response.json().get("detail", "")
    else:
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "audio/mpeg"
        with open(
            os.path.join(
                tests.conftest.test_outputs_dir,
                f"{input_text.replace(' ', '_')}_{voice}.mp3",
            ),
            "wb",
        ) as f:
            f.write(response.content)


@pytest.mark.openai_api
def test_openai_speech_api_with_invalid_style(client):
    request = AudioSpeechRequest(
        input="Test text", voice="female2", style="invalid_style"
    )
    response = client.post("/v1/audio/speech", json=request.model_dump())

    assert response.status_code == 400
    assert "Invalid style" in response.json().get("detail", "")


# @pytest.mark.openai_api
# def test_transcribe_not_implemented(client):
#     file = {"file": ("test.wav", b"test audio data")}
#     response = client.post("/v1/audio/transcriptions", files=file)

#     assert response.status_code == 200
#     assert response.json() == success_response("not implemented yet")


# TODO
# @mark.parametrize("file_name, file_content", [("test.wav", b"test audio data")])
# @pytest.mark.openai_api
# def test_transcribe_with_file(client, file_name, file_content):
#     file = {"file": (file_name, file_content)}
#     response = client.post("/v1/audio/transcriptions", files=file)

#     assert response.status_code == 200
#     assert isinstance(response.json(), TranscriptionsVerboseResponse)
#     assert response.json().text == "not implemented yet"
