import librosa
import numpy as np
import soundfile as sf

from modules.core.models.AudioReshaper import AudioReshaper
from modules.webui.speaker.wav_misc import encode_to_wav


def save_audio(file_path: str, sample_rate: int, audio_data: np.ndarray):
    sf.write(file_path, audio_data, sample_rate)


def load_audio(file_path: str) -> tuple[int, np.ndarray]:
    audio, sample_rate = librosa.load(file_path, sr=None)
    return sample_rate, audio


def load_audio_wav(file_path: str) -> tuple[int, bytes]:
    sr, audio = load_audio(file_path)
    # 这里编码的意义主要是压缩到 int16
    sr, audio_bytes = encode_to_wav((sr, audio))
    return sr, audio_bytes
