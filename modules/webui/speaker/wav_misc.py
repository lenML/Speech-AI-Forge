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
    if len(audio_data.shape) > 1:
        dtype = audio_data.dtype
        if audio_data.shape[0] < audio_data.shape[1]:
            # 假设格式为 (channels, samples)
            audio_data: np.ndarray = np.mean(audio_data, axis=0)
        else:
            # 假设格式为 (samples, channels)
            audio_data: np.ndarray = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(dtype)

    # Ensure the audio data is within the valid range before converting to int16
    if np.issubdtype(audio_data.dtype, np.floating):
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)
    elif np.issubdtype(audio_data.dtype, np.integer):
        info = np.iinfo(audio_data.dtype)
        audio_data = audio_data.astype(np.float64)
        audio_data = (audio_data - info.min) / (info.max - info.min)
        audio_data = (audio_data * 2 - 1) * 32767
        audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Unsupported audio data type: {audio_data.dtype}")

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, audio_data)
    wav_buffer.seek(0)
    wav_bytes = wav_buffer.getvalue()

    return sample_rate, wav_bytes


if __name__ == "__main__":
    import soundfile as sf

    # 测试函数，读取音频处理并保存
    input_file = "./参考音频.wav"
    output_file = "./test_cosyvoice_encode.wav"
    audio_data, sample_rate = sf.read(input_file)

    out_sr, wav_bytes = encode_to_wav((sample_rate, audio_data))
    print(f"Output sample rate: {sample_rate} -> {out_sr}")
    with open(output_file, "wb") as f:
        f.write(wav_bytes)
