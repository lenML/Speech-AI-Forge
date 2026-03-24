"""Unit tests for MiniMax Cloud TTS model.

These tests mock the MiniMax API to verify model behavior without making
actual API calls. Run from project root:

    python -m pytest tests/model/test_minimax_cloud_tts.py -v
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydub import AudioSegment as PydubSegment


def _make_mp3_bytes(duration_ms=100, sample_rate=24000):
    """Create minimal valid MP3 bytes for testing."""
    samples = np.zeros(int(sample_rate * duration_ms / 1000), dtype=np.int16)
    segment = PydubSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()


def _make_api_response(mp3_bytes):
    """Create a mock MiniMax TTS API response."""
    return {
        "base_resp": {"status_code": 0, "status_msg": "success"},
        "data": {"audio": mp3_bytes.hex()},
    }


# ---------- lazy import helper to bypass heavy deps ----------

_module = None


def _get_module():
    """Import the MiniMax Cloud TTS module, handling import failures gracefully."""
    global _module
    if _module is not None:
        return _module

    # Ensure project root is on path
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # We need to mock heavy dependencies that the import chain pulls in
    # The MiniMaxCloudTTSModel itself only needs: TTSModel base, dcls, processor
    # But TTSModel imports BaseZooModel which imports AutoModelDownloader, etc.
    # We'll mock the modules that require torch/torchaudio

    mock_modules = {}
    for mod_name in [
        "modules.repos_static",
        "modules.repos_static.sys_paths",
        "modules.devices",
        "modules.downloader",
        "modules.downloader.AutoModelDownloader",
        "modules.downloader.dl_base",
        "modules.downloader.dl_registry",
        "modules.downloader.dls",
        "torchaudio",
        "torchaudio.transforms",
        "torch",
    ]:
        if mod_name not in sys.modules:
            mock_modules[mod_name] = MagicMock()

    with patch.dict(sys.modules, mock_modules):
        # Mock devices module functions
        devices_mock = sys.modules.get("modules.devices", MagicMock())
        devices_mock.devices = MagicMock()
        devices_mock.devices.after_gc = lambda: (lambda f: f)
        devices_mock.devices.get_device_for = MagicMock(return_value="cpu")
        devices_mock.devices.dtype = "float32"

        # Mock AutoModelDownloader
        adl_mock = sys.modules.get(
            "modules.downloader.AutoModelDownloader", MagicMock()
        )
        adl_mock.AutoModelDownloader = MagicMock

        from modules.core.models.tts.MiniMaxCloudTTSModel import (
            MINIMAX_TTS_API_URL,
            MINIMAX_VOICE_IDS,
            MiniMaxCloudTTSModel,
        )

        _module = type(
            "Module",
            (),
            {
                "MiniMaxCloudTTSModel": MiniMaxCloudTTSModel,
                "MINIMAX_VOICE_IDS": MINIMAX_VOICE_IDS,
                "MINIMAX_TTS_API_URL": MINIMAX_TTS_API_URL,
            },
        )
        return _module


@pytest.fixture(scope="module")
def mod():
    return _get_module()


# ==================== Tests ====================


class TestMiniMaxCloudTTSModelInit:
    def test_hd_model_init(self, mod):
        model = mod.MiniMaxCloudTTSModel("speech-2.8-hd")
        assert model.model_id == "minimax-hd"
        assert model.model_variant == "speech-2.8-hd"

    def test_turbo_model_init(self, mod):
        model = mod.MiniMaxCloudTTSModel("speech-2.8-turbo")
        assert model.model_id == "minimax-turbo"
        assert model.model_variant == "speech-2.8-turbo"

    def test_default_model_init(self, mod):
        model = mod.MiniMaxCloudTTSModel()
        assert model.model_id == "minimax-hd"
        assert model.model_variant == "speech-2.8-hd"

    def test_sample_rate(self, mod):
        model = mod.MiniMaxCloudTTSModel()
        assert model.get_sample_rate() == 24000

    def test_not_loaded_initially(self, mod):
        model = mod.MiniMaxCloudTTSModel()
        assert not model.is_loaded()


class TestMiniMaxCloudTTSModelApiKey:
    def test_is_downloaded_with_key(self, mod):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = mod.MiniMaxCloudTTSModel()
            assert model.is_downloaded(verbose=False)

    def test_is_downloaded_without_key(self, mod):
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            model = mod.MiniMaxCloudTTSModel()
            assert not model.is_downloaded(verbose=False)

    def test_load_with_key(self, mod):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = mod.MiniMaxCloudTTSModel()
            model.load()
            assert model.is_loaded()

    def test_load_without_key_raises(self, mod):
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            model = mod.MiniMaxCloudTTSModel()
            with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
                model.load()

    def test_unload(self, mod):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            model = mod.MiniMaxCloudTTSModel()
            model.load()
            assert model.is_loaded()
            model.unload()
            assert not model.is_loaded()

    def test_can_auto_download(self, mod):
        model = mod.MiniMaxCloudTTSModel()
        assert not model.can_auto_download()

    def test_download_returns_none(self, mod):
        model = mod.MiniMaxCloudTTSModel()
        assert model.download() is None


class TestMiniMaxCloudTTSModelVoiceResolution:
    def test_default_voice(self, mod):
        from modules.core.pipeline.dcls import TTSSegment

        model = mod.MiniMaxCloudTTSModel()
        segment = TTSSegment(_type="audio", text="test")
        voice = model._get_voice_id(segment)
        assert voice == "Friendly_Person"

    def test_voice_from_speaker_name(self, mod):
        from modules.core.pipeline.dcls import TTSSegment

        model = mod.MiniMaxCloudTTSModel()
        segment = TTSSegment(_type="audio", text="test")
        spk = MagicMock()
        spk.name = "English_Graceful_Lady"
        segment.spk = spk
        voice = model._get_voice_id(segment)
        assert voice == "English_Graceful_Lady"

    def test_voice_from_emotion_male(self, mod):
        from modules.core.pipeline.dcls import TTSSegment

        model = mod.MiniMaxCloudTTSModel()
        segment = TTSSegment(_type="audio", text="test", emotion="male narrator")
        voice = model._get_voice_id(segment)
        assert voice == "English_Persuasive_Man"

    def test_voice_from_emotion_girl(self, mod):
        from modules.core.pipeline.dcls import TTSSegment

        model = mod.MiniMaxCloudTTSModel()
        segment = TTSSegment(_type="audio", text="test", emotion="cute girl")
        voice = model._get_voice_id(segment)
        assert voice == "lovely_girl"

    def test_voice_from_emotion_robot(self, mod):
        from modules.core.pipeline.dcls import TTSSegment

        model = mod.MiniMaxCloudTTSModel()
        segment = TTSSegment(_type="audio", text="test", emotion="robot voice")
        voice = model._get_voice_id(segment)
        assert voice == "English_Lucky_Robot"


class TestMiniMaxCloudTTSModelApiCall:
    def test_call_tts_api_success(self, mod):
        model = mod.MiniMaxCloudTTSModel("speech-2.8-hd")
        mp3_bytes = _make_mp3_bytes()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_api_response(mp3_bytes)
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch(
                "modules.core.models.tts.MiniMaxCloudTTSModel.requests.post",
                return_value=mock_response,
            ) as mock_post:
                result = model._call_tts_api("Hello world", "Friendly_Person")

        assert isinstance(result, bytes)
        assert len(result) > 0
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["model"] == "speech-2.8-hd"
        assert call_kwargs[1]["json"]["text"] == "Hello world"
        assert (
            call_kwargs[1]["json"]["voice_setting"]["voice_id"] == "Friendly_Person"
        )

    def test_call_tts_api_error_response(self, mod):
        model = mod.MiniMaxCloudTTSModel("speech-2.8-hd")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 1000, "status_msg": "Invalid API key"},
            "data": {"audio": ""},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "bad-key"}):
            with patch(
                "modules.core.models.tts.MiniMaxCloudTTSModel.requests.post",
                return_value=mock_response,
            ):
                with pytest.raises(RuntimeError, match="MiniMax TTS API error"):
                    model._call_tts_api("Hello", "Friendly_Person")

    def test_call_tts_api_empty_audio(self, mod):
        model = mod.MiniMaxCloudTTSModel("speech-2.8-hd")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": ""},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch(
                "modules.core.models.tts.MiniMaxCloudTTSModel.requests.post",
                return_value=mock_response,
            ):
                with pytest.raises(RuntimeError, match="empty audio"):
                    model._call_tts_api("Hello", "Friendly_Person")

    def test_turbo_model_sends_correct_variant(self, mod):
        model = mod.MiniMaxCloudTTSModel("speech-2.8-turbo")
        mp3_bytes = _make_mp3_bytes()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_api_response(mp3_bytes)
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch(
                "modules.core.models.tts.MiniMaxCloudTTSModel.requests.post",
                return_value=mock_response,
            ) as mock_post:
                model._call_tts_api("Hello", "Friendly_Person")

        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["model"] == "speech-2.8-turbo"


class TestMiniMaxCloudTTSModelMp3Conversion:
    def test_mp3_bytes_to_np_audio(self, mod):
        model = mod.MiniMaxCloudTTSModel()
        mp3_bytes = _make_mp3_bytes(duration_ms=200)
        sr, samples = model._mp3_bytes_to_np_audio(mp3_bytes)
        assert sr == 24000
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert len(samples) > 0

    def test_mp3_bytes_to_np_audio_mono(self, mod):
        # Create stereo MP3 and verify it gets converted to mono
        model = mod.MiniMaxCloudTTSModel()
        samples = np.zeros((48000, 2), dtype=np.int16)
        segment = PydubSegment(
            samples.tobytes(),
            frame_rate=48000,
            sample_width=2,
            channels=2,
        )
        buffer = io.BytesIO()
        segment.export(buffer, format="mp3")
        mp3_bytes = buffer.getvalue()

        sr, result = model._mp3_bytes_to_np_audio(mp3_bytes)
        assert sr == 24000
        assert result.ndim == 1  # mono


class TestMiniMaxCloudTTSModelGenerateBatch:
    def test_generate_batch_single_segment(self, mod):
        from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment

        model = mod.MiniMaxCloudTTSModel()
        mp3_bytes = _make_mp3_bytes()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_api_response(mp3_bytes)
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch(
                "modules.core.models.tts.MiniMaxCloudTTSModel.requests.post",
                return_value=mock_response,
            ):
                segment = TTSSegment(_type="audio", text="Hello world")
                context = TTSPipelineContext()
                context.infer_config.no_cache = True
                results = model.generate_batch([segment], context)

        assert len(results) == 1
        sr, samples = results[0]
        assert sr == 24000
        assert isinstance(samples, np.ndarray)

    def test_generate_batch_multiple_segments(self, mod):
        from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment

        model = mod.MiniMaxCloudTTSModel()
        mp3_bytes = _make_mp3_bytes()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_api_response(mp3_bytes)
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch(
                "modules.core.models.tts.MiniMaxCloudTTSModel.requests.post",
                return_value=mock_response,
            ) as mock_post:
                segments = [
                    TTSSegment(_type="audio", text="Hello"),
                    TTSSegment(_type="audio", text="World"),
                ]
                context = TTSPipelineContext()
                context.infer_config.no_cache = True
                results = model.generate_batch(segments, context)

        assert len(results) == 2
        assert mock_post.call_count == 2

    def test_generate_batch_empty_text(self, mod):
        from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment

        model = mod.MiniMaxCloudTTSModel()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            segment = TTSSegment(_type="audio", text="   ")
            context = TTSPipelineContext()
            context.infer_config.no_cache = True
            results = model.generate_batch([segment], context)

        assert len(results) == 1
        sr, samples = results[0]
        assert sr == 24000
        assert len(samples) > 0  # silence

    def test_generate_batch_respects_stop(self, mod):
        from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment

        model = mod.MiniMaxCloudTTSModel()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            segments = [
                TTSSegment(_type="audio", text="Hello"),
                TTSSegment(_type="audio", text="World"),
            ]
            context = TTSPipelineContext()
            context.infer_config.no_cache = True
            context.stop = True
            results = model.generate_batch(segments, context)

        assert len(results) == 0

    def test_generate_batch_stream(self, mod):
        from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment

        model = mod.MiniMaxCloudTTSModel()
        mp3_bytes = _make_mp3_bytes()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_api_response(mp3_bytes)
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch(
                "modules.core.models.tts.MiniMaxCloudTTSModel.requests.post",
                return_value=mock_response,
            ):
                segment = TTSSegment(_type="audio", text="Hello")
                context = TTSPipelineContext()
                context.infer_config.no_cache = True
                batches = list(model.generate_batch_stream([segment], context))

        assert len(batches) == 1
        assert len(batches[0]) == 1


class TestMiniMaxVoiceIds:
    def test_voice_ids_not_empty(self, mod):
        assert len(mod.MINIMAX_VOICE_IDS) > 0

    def test_friendly_person_in_list(self, mod):
        assert "Friendly_Person" in mod.MINIMAX_VOICE_IDS

    def test_no_invalid_chinese_voices(self, mod):
        invalid = [
            "Chinese_Empress",
            "Chinese_Gentle_Boy",
            "Chinese_Cute_Girl",
            "Chinese_Storyteller_Male",
        ]
        for v in invalid:
            assert v not in mod.MINIMAX_VOICE_IDS


class TestMiniMaxApiUrl:
    def test_api_url(self, mod):
        assert mod.MINIMAX_TTS_API_URL == "https://api.minimax.io/v1/t2a_v2"
