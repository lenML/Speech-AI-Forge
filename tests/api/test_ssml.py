import pytest

ssml_content = "<speak version='0.1'><voice>Hello, world!</voice></speak>"


@pytest.mark.ssml_api
def test_synthesize_ssml_success(client):
    response = client.post(
        "/v1/ssml",
        json={
            "ssml": ssml_content,
            "format": "mp3",
            "batch_size": 1,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"


@pytest.mark.ssml_api
def test_synthesize_ssml_empty_ssml(client):
    response = client.post(
        "/v1/ssml", json={"ssml": "", "format": "mp3", "batch_size": 4}
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "SSML content is required."}


@pytest.mark.ssml_api
def test_synthesize_ssml_invalid_batch_size(client):
    response = client.post(
        "/v1/ssml",
        json={"ssml": ssml_content, "format": "mp3", "batch_size": 0},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Batch size must be greater than 0."}


@pytest.mark.ssml_api
def test_synthesize_ssml_invalid_format(client):
    response = client.post(
        "/v1/ssml",
        json={
            "ssml": ssml_content,
            "format": "invalid_format",
            "batch_size": 4,
        },
    )
    assert response.status_code == 422
    assert "detail" in response.json()
