import io
import sys
from io import BytesIO

import librosa
import numpy as np
import numpy.typing as npt
import pyrubberband as pyrb
import scipy.io.wavfile as wavfile
import soundfile as sf
from pydub import AudioSegment, effects
import base64

INT16_MAX = np.iinfo(np.int16).max


def bytes_to_librosa_array(audio_bytes: bytes, sample_rate: int) -> npt.NDArray:
    """
    Converts bytes to librosa array.

    NOTE: 注意 audio_bytes 假设为 np.int16 类型，其他类型不会报错，但是load出来时噪音

    Args:
        audio_bytes: bytes
        sample_rate: int

    Returns:
        librosa array
    """
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sample_rate, audio_np)
    byte_io.seek(0)

    try:
        audio_data, read_sr = sf.read(byte_io, dtype="float32")
        if read_sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {read_sr} != {sample_rate}")
        return audio_data
    except:
        _, audio_data = wavfile.read(byte_io)
        audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        return audio_float


def audio_to_int16(audio_data: np.ndarray) -> np.ndarray:
    """
    Converts audio data to int16.

    NOTE: 这个转换将丢失精度

    Args:
        audio_data: np.ndarray

    Returns:
        np.ndarray
    """
    if np.issubdtype(audio_data.dtype, np.floating):
        audio_data = (audio_data * INT16_MAX).astype(np.int16)
    return audio_data


def pydub_to_np(audio: AudioSegment) -> tuple[int, np.ndarray]:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """
    nd_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels != 1:
        nd_array = nd_array.reshape((-1, audio.channels))
    nd_array = nd_array / (1 << (8 * audio.sample_width - 1))

    return (
        audio.frame_rate,
        nd_array,
    )


def audiosegment_to_librosawav(audiosegment: AudioSegment) -> np.ndarray:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    """
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr


def ndarray_to_segment(
    ndarray: np.ndarray, frame_rate: int, sample_width: int = None, channels: int = None
) -> AudioSegment:
    buffer = BytesIO()
    sf.write(buffer, ndarray, frame_rate, format="wav", subtype="PCM_16")
    buffer.seek(0)
    sound: AudioSegment = AudioSegment.from_wav(buffer)

    if sample_width is None:
        sample_width = sound.sample_width
    if channels is None:
        channels = sound.channels

    return (
        sound.set_frame_rate(frame_rate)
        .set_sample_width(sample_width)
        .set_channels(channels)
    )


def apply_prosody_to_audio_segment(
    audio_segment: AudioSegment,
    rate: float = 1,
    volume: float = 0,
    pitch: int = 0,
    sr: int = 24000,
) -> AudioSegment:
    audio_data = audiosegment_to_librosawav(audio_segment)

    audio_data = apply_prosody_to_audio_data(audio_data, rate, volume, pitch, sr)

    audio_segment = ndarray_to_segment(
        audio_data, sr, audio_segment.sample_width, audio_segment.channels
    )

    return audio_segment


def apply_prosody_to_audio_data(
    audio_data: np.ndarray,
    rate: float = 1,
    volume: float = 0,
    pitch: float = 0,
    sr: int = 24000,
) -> np.ndarray:
    if audio_data.dtype == np.int16:
        # NOTE: 其实感觉一个报个错...
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.float16:
        audio_data = audio_data.astype(np.float32)

    if rate != 1:
        audio_data = pyrb.time_stretch(audio_data, sr=sr, rate=rate)

    if volume != 0:
        volume = max(min(volume, 6), -20)
        gain = 10 ** (volume / 20)
        audio_data = audio_data * gain

    if pitch != 0:
        audio_data = pyrb.pitch_shift(audio_data, sr=sr, n_steps=pitch)

    return audio_data


def apply_normalize(
    audio_data: np.ndarray,
    headroom: float = 1,
    sr: int = 24000,
):
    segment = ndarray_to_segment(audio_data, sr)
    segment = effects.normalize(seg=segment, headroom=headroom)

    return pydub_to_np(segment)


def silence_np(duration_s: float, sample_rate: int = 24000) -> tuple[int, np.ndarray]:
    silence = AudioSegment.silent(duration=duration_s * 1000, frame_rate=sample_rate)
    return pydub_to_np(silence)


def remove_silence_edges(
    audio: np.ndarray, silence_threshold: float = -42
) -> np.ndarray:
    """
    去除音频两端的静音区域。

    参数:
    - audio: 一维 ndarray，音频波形
    - silence_threshold: 静音阈值（单位 dB），低于该值将被认为是静音
    - sr: 采样率（用于能量计算，默认 22050）

    返回:
    - 去除静音后的音频 ndarray
    """
    # 计算非静音区间
    intervals = librosa.effects.split(audio, top_db=-silence_threshold)

    if len(intervals) == 0:
        return np.array([])  # 全是静音

    start, end = intervals[0][0], intervals[-1][1]
    return audio[start:end]


def np_audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int = 24000,
) -> bytes:
    bytes_ret = io.BytesIO()
    sf.write(bytes_ret, audio, sample_rate, format="wav", subtype="PCM_16")
    bytes_ret.seek(0)
    return bytes_ret.read()


def load_audio(audio_filepath: str) -> tuple[int, np.ndarray]:
    return pydub_to_np(AudioSegment.from_file(audio_filepath))


def get_wav_sr(audio: bytes) -> int:
    byte_io = io.BytesIO(audio)
    audio_data, read_sr = sf.read(byte_io, dtype="float32")
    return read_sr


def read_base64_audio(base64_audio: str) -> tuple[int, np.ndarray]:
    # 去除 data URI
    if ";base64," in base64_audio:
        base64_audio = base64_audio.split(";base64,")[-1]

    audio_bytes = base64.b64decode(base64_audio)
    audio_buffer = io.BytesIO(audio_bytes)
    audio_np, sr = sf.read(audio_buffer)
    return sr, audio_np


if __name__ == "__main__":
    input_file = sys.argv[1]

    time_stretch_factors = [0.5, 0.75, 1.5, 1.0]
    pitch_shift_factors = [-12, -5, 0, 5, 12]

    input_sound = AudioSegment.from_mp3(input_file)

    for time_factor in time_stretch_factors:
        output_wav = f"{input_file}_time_{time_factor}.wav"
        output_sound = apply_prosody_to_audio_segment(input_sound, rate=time_factor)
        output_sound.export(output_wav, format="wav")

    for pitch_factor in pitch_shift_factors:
        output_wav = f"{input_file}_pitch_{pitch_factor}.wav"
        output_sound = apply_prosody_to_audio_segment(input_sound, pitch=pitch_factor)
        output_sound.export(output_wav, format="wav")
