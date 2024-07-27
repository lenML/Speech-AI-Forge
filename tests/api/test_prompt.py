from pytest import mark


@mark.refine_api
def test_refine_success(client):
    response = client.post(
        "/v1/prompt/refine",
        json={
            "text": "你好，世界",
            "prompt": "[oral_2][laugh_0][break_6]",
            "seed": -1,
            "top_P": 0.7,
            "top_K": 20,
            "temperature": 0.7,
            "repetition_penalty": 1,
            "max_new_token": 384,
            "spliter_threshold": 300,
            "normalize": True,
        },
    )
    data = response.json()
    assert response.status_code == 200
    assert "message" in data
    assert data["message"] == "ok"
    assert "data" in data
    assert isinstance(data["data"], str)


@mark.parametrize(
    "text, pipe_id, status_code",
    [
        ("你好 123456789", "chat-tts", 200),
        ("你好 123456789", "cosy-voice", 200),
        ("你好 123456789", "fish-speech", 200),
    ],
)
@mark.normalize_api
def test_normalize_success(client, text, pipe_id, status_code):
    response = client.post(
        "/v1/text/normalize",
        json={"text": text, "pipe_id": pipe_id},
    )
    data = response.json()
    assert response.status_code == status_code
    if status_code == 200:
        assert "message" in data
        assert data["message"] == "ok"
        assert "data" in data
        assert isinstance(data["data"], str)
