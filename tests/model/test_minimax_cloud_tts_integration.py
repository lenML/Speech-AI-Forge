"""Integration tests for MiniMax Cloud TTS model.

These tests make actual API calls to the MiniMax TTS API.
They require the MINIMAX_API_KEY environment variable to be set.

Run with: pytest tests/model/test_minimax_cloud_tts_integration.py -m minimax_integration
"""

import os

import numpy as np
import pytest

from modules.core.models.tts.MiniMaxCloudTTSModel import MiniMaxCloudTTSModel
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment

pytestmark = pytest.mark.minimax_integration

SKIP_REASON = "MINIMAX_API_KEY not set; skipping MiniMax integration tests"


@pytest.fixture
def api_key():
    key = os.environ.get("MINIMAX_API_KEY", "")
    if not key:
        pytest.skip(SKIP_REASON)
    return key


@pytest.fixture
def hd_model(api_key):
    model = MiniMaxCloudTTSModel("speech-2.8-hd")
    model.load()
    return model


@pytest.fixture
def turbo_model(api_key):
    model = MiniMaxCloudTTSModel("speech-2.8-turbo")
    model.load()
    return model


class TestMiniMaxCloudTTSIntegration:
    def test_hd_model_generate(self, hd_model):
        segment = TTSSegment(_type="audio", text="Hello, this is a test.")
        context = TTSPipelineContext()
        context.infer_config.no_cache = True

        results = hd_model.generate_batch([segment], context)

        assert len(results) == 1
        sr, samples = results[0]
        assert sr == 24000
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert len(samples) > 1000  # should have substantial audio

    def test_turbo_model_generate(self, turbo_model):
        segment = TTSSegment(_type="audio", text="Quick test for turbo model.")
        context = TTSPipelineContext()
        context.infer_config.no_cache = True

        results = turbo_model.generate_batch([segment], context)

        assert len(results) == 1
        sr, samples = results[0]
        assert sr == 24000
        assert len(samples) > 1000

    def test_chinese_text(self, hd_model):
        segment = TTSSegment(_type="audio", text="你好，这是一个测试。")
        context = TTSPipelineContext()
        context.infer_config.no_cache = True

        results = hd_model.generate_batch([segment], context)

        assert len(results) == 1
        sr, samples = results[0]
        assert sr == 24000
        assert len(samples) > 1000
