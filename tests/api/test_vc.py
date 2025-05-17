import os
from typing import Union

import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import tests.conftest


@pytest.mark.vc_api
def test_vc_ref_audio_file(client: TestClient):
    src_audio_path = "./tests/test_inputs/cosyvoice_out1.wav"
    ref_audio_path = "./tests/test_inputs/chattts_out1.wav"

    with open(src_audio_path, "rb") as src_audio_file:
        with open(ref_audio_path, "rb") as ref_audio_file:
            response = client.post(
                "/v1/vc",
                files={"src_audio": src_audio_file, "ref_audio": ref_audio_file},
                params={"model": "open-voice", "tau": 0.3, "format": "mp3"},
            )

    assert response.status_code == 200
    assert response.headers["content-type"] in ["audio/wav", "audio/mpeg"]

    output_file = os.path.join(
        tests.conftest.test_outputs_dir, "vc_api_ref_audio_success.mp3"
    )

    with open(
        output_file,
        "wb",
    ) as f:
        f.write(response.content)

    assert os.path.getsize(output_file) > 0, "Stream output file is empty"


@pytest.mark.vc_api
def test_vc_ref_spk(client: TestClient):
    src_audio_path = "./tests/test_inputs/cosyvoice_out1.wav"

    with open(src_audio_path, "rb") as src_audio_file:
        response = client.post(
            "/v1/vc",
            files={"src_audio": src_audio_file},
            params={
                "model": "open-voice",
                "tau": 0.3,
                "format": "mp3",
                "ref_spk": "mona",
            },
        )
    if response.status_code != 200:
        print(response.json())
    assert response.status_code == 200
    assert response.headers["content-type"] in ["audio/wav", "audio/mpeg"]

    output_file = os.path.join(
        tests.conftest.test_outputs_dir, "vc_api_ref_spk_success.mp3"
    )

    with open(
        output_file,
        "wb",
    ) as f:
        f.write(response.content)

    assert os.path.getsize(output_file) > 0, "Stream output file is empty"
