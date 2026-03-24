import io
import logging
import os
from typing import Generator

import numpy as np
import requests
from pydub import AudioSegment as PydubSegment

from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.processor import NP_AUDIO

logger = logging.getLogger(__name__)

# Verified MiniMax TTS voice IDs
MINIMAX_VOICE_IDS = [
    "English_Graceful_Lady",
    "English_Insightful_Speaker",
    "English_radiant_girl",
    "English_Persuasive_Man",
    "English_Lucky_Robot",
    "Wise_Woman",
    "cute_boy",
    "lovely_girl",
    "Friendly_Person",
    "Inspirational_girl",
    "Deep_Voice_Man",
    "sweet_girl",
]

MINIMAX_TTS_API_URL = "https://api.minimax.io/v1/t2a_v2"


class MiniMaxCloudTTSModel(TTSModel):
    """MiniMax Cloud TTS model using MiniMax T2A v2 API.

    This is a cloud-based TTS model that calls MiniMax's text-to-audio API
    instead of running local model inference. No GPU or model download required.

    Supported models:
    - speech-2.8-hd: Higher quality, slower
    - speech-2.8-turbo: Faster, slightly lower quality

    Requires MINIMAX_API_KEY environment variable to be set.
    """

    def __init__(self, model_variant: str = "speech-2.8-hd") -> None:
        model_id = (
            "minimax-hd" if model_variant == "speech-2.8-hd" else "minimax-turbo"
        )
        super().__init__(model_id=model_id, model_name=model_variant)

        self.model_variant = model_variant
        self._loaded = False

    def _get_api_key(self) -> str:
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY environment variable is not set. "
                "Get your API key from https://platform.minimaxi.com/"
            )
        return api_key

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        # Verify API key is available
        self._get_api_key()
        self._loaded = True
        logger.info(f"MiniMax Cloud TTS ({self.model_variant}) ready.")

    def unload(self) -> None:
        self._loaded = False

    def is_downloaded(self, verbose=True) -> bool:
        # Cloud model - always "downloaded" if API key is set
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            if verbose:
                logger.warning(
                    "MINIMAX_API_KEY not set. Set it to use MiniMax Cloud TTS."
                )
            return False
        return True

    def can_auto_download(self) -> bool:
        # No download needed for cloud model
        return False

    def download(self, force=False):
        # No-op for cloud model
        return None

    def get_sample_rate(self) -> int:
        # MiniMax TTS returns MP3 audio; after decoding we resample to 24000
        return 24000

    def _get_voice_id(self, segment: TTSSegment) -> str:
        """Resolve voice ID from segment speaker or use default."""
        if segment.spk is not None:
            spk_name = segment.spk.name if hasattr(segment.spk, "name") else ""
            # Check if speaker name matches a MiniMax voice ID
            if spk_name in MINIMAX_VOICE_IDS:
                return spk_name
            # Check speaker description or custom field
            if hasattr(segment.spk, "desc") and segment.spk.desc:
                desc = segment.spk.desc
                for vid in MINIMAX_VOICE_IDS:
                    if vid.lower() in desc.lower():
                        return vid

        # Use emotion as voice hint if available
        emotion = segment.emotion or ""
        if emotion:
            emotion_lower = emotion.lower()
            if "male" in emotion_lower or "man" in emotion_lower:
                return "English_Persuasive_Man"
            elif "robot" in emotion_lower:
                return "English_Lucky_Robot"
            elif "girl" in emotion_lower or "cute" in emotion_lower:
                return "lovely_girl"

        return "Friendly_Person"

    def _call_tts_api(self, text: str, voice_id: str, speed: float = 1.0) -> bytes:
        """Call MiniMax TTS API and return raw audio bytes (MP3)."""
        api_key = self._get_api_key()

        payload = {
            "model": self.model_variant,
            "text": text,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
            },
            "audio_setting": {
                "format": "mp3",
            },
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            MINIMAX_TTS_API_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()

        # Check for API errors
        base_resp = result.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            error_msg = base_resp.get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax TTS API error: {error_msg}")

        # Extract hex-encoded audio data
        audio_hex = result.get("data", {}).get("audio", "")
        if not audio_hex:
            raise RuntimeError("MiniMax TTS API returned empty audio data")

        return bytes.fromhex(audio_hex)

    def _mp3_bytes_to_np_audio(self, mp3_bytes: bytes) -> NP_AUDIO:
        """Convert MP3 bytes to NP_AUDIO (sample_rate, numpy_array)."""
        audio_segment = PydubSegment.from_mp3(io.BytesIO(mp3_bytes))

        # Convert to target sample rate
        target_sr = self.get_sample_rate()
        if audio_segment.frame_rate != target_sr:
            audio_segment = audio_segment.set_frame_rate(target_sr)

        # Convert to mono
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        # Convert to numpy float32 array
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / np.iinfo(np.int16).max

        return (target_sr, samples)

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        cached = self.get_cache(segments=segments, context=context)
        if cached is not None:
            return cached

        self.load()

        results = []
        for segment in segments:
            if context.stop:
                break

            voice_id = self._get_voice_id(segment)
            text = segment.text
            if not text.strip():
                # Return silence for empty text
                sr = self.get_sample_rate()
                results.append((sr, np.zeros(int(sr * 0.1), dtype=np.float32)))
                continue

            logger.info(
                f"MiniMax Cloud TTS: generating with model={self.model_variant}, "
                f"voice={voice_id}, text_len={len(text)}"
            )

            mp3_bytes = self._call_tts_api(text=text, voice_id=voice_id)
            np_audio = self._mp3_bytes_to_np_audio(mp3_bytes)
            results.append(np_audio)

        if not context.stop:
            self.set_cache(segments=segments, context=context, value=results)

        return results

    def generate_batch_stream(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> Generator[list[NP_AUDIO], None, None]:
        # MiniMax TTS API doesn't support streaming; yield full result
        results = self.generate_batch(segments, context)
        yield results
