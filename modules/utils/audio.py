import sys
from io import BytesIO

import numpy as np
import pyrubberband as pyrb
import soundfile as sf
from pydub import AudioSegment

INT16_MAX = np.iinfo(np.int16).max


def audio_to_int16(audio_data):
    if (
        audio_data.dtype == np.float32
        or audio_data.dtype == np.float64
        or audio_data.dtype == np.float128
        or audio_data.dtype == np.float16
    ):
        audio_data = (audio_data * INT16_MAX).astype(np.int16)
    return audio_data


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


def pydub_to_np(audio: AudioSegment) -> tuple[int, np.ndarray]:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """
    return (
        audio.frame_rate,
        np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(
            (-1, audio.channels)
        )
        / (1 << (8 * audio.sample_width - 1)),
    )


def ndarray_to_segment(ndarray, frame_rate):
    buffer = BytesIO()
    sf.write(buffer, ndarray, frame_rate, format="wav")
    buffer.seek(0)
    sound = AudioSegment.from_wav(
        buffer,
    )
    return sound


def time_stretch(input_segment: AudioSegment, time_factor: float) -> AudioSegment:
    """
    factor range -> [0.2,10]
    """
    time_factor = np.clip(time_factor, 0.2, 10)
    sr = input_segment.frame_rate
    y = audiosegment_to_librosawav(input_segment)
    y_stretch = pyrb.time_stretch(y, sr, time_factor)

    sound = ndarray_to_segment(
        y_stretch,
        frame_rate=sr,
    )
    return sound


def pitch_shift(
    input_segment: AudioSegment,
    pitch_shift_factor: float,
) -> AudioSegment:
    """
    factor range -> [-12,12]
    """
    pitch_shift_factor = np.clip(pitch_shift_factor, -12, 12)
    sr = input_segment.frame_rate
    y = audiosegment_to_librosawav(input_segment)
    y_shift = pyrb.pitch_shift(y, sr, pitch_shift_factor)

    sound = ndarray_to_segment(
        y_shift,
        frame_rate=sr,
    )
    return sound


def apply_prosody_to_audio_data(
    audio_data: np.ndarray,
    rate: float = 1,
    volume: float = 0,
    pitch: float = 0,
    sr: int = 24000,
) -> np.ndarray:
    if rate != 1:
        audio_data = pyrb.time_stretch(audio_data, sr=sr, rate=rate)

    if volume != 0:
        audio_data = audio_data * volume

    if pitch != 0:
        audio_data = pyrb.pitch_shift(audio_data, sr=sr, n_steps=pitch)

    return audio_data


if __name__ == "__main__":
    input_file = sys.argv[1]

    time_stretch_factors = [0.5, 0.75, 1.5, 1.0]
    pitch_shift_factors = [-12, -5, 0, 5, 12]

    input_sound = AudioSegment.from_mp3(input_file)

    for time_factor in time_stretch_factors:
        output_wav = f"time_stretched_{int(time_factor * 100)}.wav"
        sound = time_stretch(input_sound, time_factor)
        sound.export(output_wav, format="wav")

    for pitch_factor in pitch_shift_factors:
        output_wav = f"pitch_shifted_{int(pitch_factor * 100)}.wav"
        sound = pitch_shift(input_sound, pitch_factor)
        sound.export(output_wav, format="wav")
