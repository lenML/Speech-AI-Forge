import io
import numpy as np
from scipy.io import wavfile


def encode_to_wav(audio_tuple: tuple[int, np.ndarray]):
    if not isinstance(audio_tuple, tuple) or len(audio_tuple) != 2:
        raise ValueError(
            "Invalid audio data format. Expected a tuple (sample_rate, audio_data)."
        )

    sample_rate, audio_data = audio_tuple

    if not isinstance(sample_rate, int) or not isinstance(audio_data, np.ndarray):
        raise ValueError("Invalid types for audio data. Expected (int, np.ndarray).")

    if audio_data.size == 0:
        raise ValueError("Audio data is empty.")

    # 如果音频数据是多声道的，取第一个声道或对声道进行平均处理
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        dtype = audio_data.dtype
        audio_data: np.ndarray = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(dtype)

    # Ensure the audio data is within the valid range before converting to int16
    if np.issubdtype(audio_data.dtype, np.floating):
        audio_data = np.clip(audio_data, -1.0, 1.0)  # Ensure data is within [-1.0, 1.0]
        audio_data = (audio_data * 32767).astype(np.int16)  # Convert to int16 range
    else:
        audio_data = audio_data.astype(np.int16)

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, audio_data)
    wav_bytes = wav_buffer.getvalue()

    return sample_rate, wav_bytes
