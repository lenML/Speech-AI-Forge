import pytest
from fastapi.testclient import TestClient


@pytest.mark.openai_api_stt
def test_openai_speech_api_with_invalid_style(client: TestClient):
    file_path = "./tests/test_inputs/cosyvoice_out1.wav"

    with open(file_path, "rb") as file:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": (file_path, file, "audio/wav")},
            params={
                "model": "whisper.large",
                "language": "zh",
                "prompt": "",
                "response_format": "txt",
                "temperature": 0,
                "timestamp_granularities": "segment",
            },
        )

    assert response.status_code == 200
    response_data = response.json()
    assert isinstance(response_data["text"], str)
    expected_text = "我们走的每一步都是我们策略的一部分\n你看到的所有一切\n包括我此刻与你交谈\n所做的一切\n所说的每一句话\n都有深远的含义\n"
    assert response_data["text"] == expected_text
