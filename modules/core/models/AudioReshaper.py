import librosa
import numpy as np

from modules.core.pipeline.processor import NP_AUDIO


class AudioReshaper:

    @staticmethod
    def resample_audio(audio: NP_AUDIO, target_sr: int) -> NP_AUDIO:
        sr, data = audio

        if sr == target_sr:
            return sr, data
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        return target_sr, data

    @staticmethod
    def ensure_float32(audio: NP_AUDIO) -> NP_AUDIO:
        sr, data = audio
        if data.dtype == np.int16:
            data = data.astype(np.float32)
            data /= np.iinfo(np.int16).max
        elif data.dtype == np.int32:
            data = data.astype(np.float32)
            data /= np.iinfo(np.int32).max
        elif data.dtype == np.float64:
            data = data.astype(np.float32)
        elif data.dtype == np.float32:
            pass
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")

        return sr, data

    @staticmethod
    def ensure_stereo_to_mono(audio: NP_AUDIO) -> NP_AUDIO:
        sr, data = audio
        if data.ndim == 2:
            data = data.mean(axis=1)
        return sr, data

    @staticmethod
    def normalize_audio_type(audio: NP_AUDIO) -> NP_AUDIO:
        audio = AudioReshaper.ensure_float32(audio=audio)
        audio = AudioReshaper.ensure_stereo_to_mono(audio=audio)
        return audio

    @staticmethod
    def normalize_audio(audio: NP_AUDIO, target_sr: int) -> NP_AUDIO:
        audio = AudioReshaper.normalize_audio_type(audio=audio)
        audio = AudioReshaper.resample_audio(audio=audio, target_sr=target_sr)
        return audio
