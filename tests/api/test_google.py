import base64
import os

import pytest

import tests.conftest
from modules.api.impl.google_api import (
    AudioConfig,
    EnhancerConfig,
    GoogleTextSynthesizeParams,
    SynthesisInput,
    VoiceSelectionParams,
)


@pytest.fixture
def body():
    req = GoogleTextSynthesizeParams(
        input=SynthesisInput(),
        voice=VoiceSelectionParams(),
        audioConfig=AudioConfig(),
        enhancerConfig=EnhancerConfig(),
    )
    req.input.text = "This is a test text."

    return req


@pytest.mark.parametrize(
    "enable_enhancer, speed",
    [
        (True, 1),
        (False, 1.5),
    ],
)
@pytest.mark.google_api
def test_google_text_synthesize_success(
    client, body: GoogleTextSynthesizeParams, enable_enhancer, speed
):
    body.enhancerConfig.enabled = enable_enhancer
    body.audioConfig.speakingRate = speed

    response = client.post("/v1/text:synthesize", json=body.model_dump())
    assert response.status_code == 200
    assert "audioContent" in response.json()

    with open(
        os.path.join(
            tests.conftest.test_outputs_dir,
            f"google_success_{'ehc' if enable_enhancer else 'no_ehc'}_s{str(speed)}.mp3",
        ),
        "wb",
    ) as f:
        b64_str = response.json()["audioContent"]
        b64_str = b64_str.split(",")[1]
        f.write(base64.b64decode(b64_str))


@pytest.mark.parametrize(
    "enable_enhancer, speed",
    [
        (True, 1),
        (False, 1.5),
    ],
)
@pytest.mark.google_api
def test_google_text_synthesize_ssml_success(
    client, body: GoogleTextSynthesizeParams, enable_enhancer, speed
):
    body.input.text = None
    body.input.ssml = """
    <speak version="0.1">
        <voice>这是一个测试文本。</voice>
    </speak>
    """
    body.enhancerConfig.enabled = enable_enhancer
    body.audioConfig.speakingRate = speed

    response = client.post("/v1/text:synthesize", json=body.model_dump())
    assert response.status_code == 200
    assert "audioContent" in response.json()

    with open(
        os.path.join(
            tests.conftest.test_outputs_dir,
            f"google_success_{'ehc' if enable_enhancer else 'no_ehc'}_s{str(speed)}.mp3",
        ),
        "wb",
    ) as f:
        b64_str = response.json()["audioContent"]
        b64_str = b64_str.split(",")[1]
        f.write(base64.b64decode(b64_str))


@pytest.mark.google_api
def test_google_text_synthesize_missing_input(
    client,
):
    response = client.post("/v1/text:synthesize", json={})
    assert response.status_code == 422
    assert "Field required" == response.json()["detail"][0]["msg"]


@pytest.mark.google_api
def test_google_text_synthesize_invalid_voice(client, body: GoogleTextSynthesizeParams):
    body.voice.name = "invalid_voice"

    response = client.post("/v1/text:synthesize", json=body.model_dump())
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.google_api
def test_google_text_synthesize_invalid_audio_encoding(
    client, body: GoogleTextSynthesizeParams
):
    body.audioConfig.audioEncoding = "invalid_format"

    response = client.post("/v1/text:synthesize", json=body.model_dump())
    assert response.status_code == 422
    assert "detail" in response.json()
