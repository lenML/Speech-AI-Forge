import os

import pytest

import tests


@pytest.mark.routes
def test_openapi_json_output(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    with open(
        os.path.join(
            tests.conftest.test_outputs_dir,
            f"openapi.json",
        ),
        "wb",
    ) as f:
        f.write(response.content)


@pytest.mark.routes
def test_cors(client):
    response = client.get("/v1/ping")
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == "*"
    assert response.headers.get("Access-Control-Allow-Methods") == "*"
    assert response.headers.get("Access-Control-Allow-Headers") == "*"
    assert response.headers.get("Access-Control-Allow-Credentials") == "true"
