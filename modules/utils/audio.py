import sys
from io import BytesIO

import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects

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


def ndarray_to_segment(ndarray: np.ndarray, frame_rate: int) -> AudioSegment:
    buffer = BytesIO()
    sf.write(buffer, ndarray, frame_rate, format="wav")
    buffer.seek(0)
    sound = AudioSegment.from_wav(
        buffer,
    )
    return sound


def apply_prosody_to_audio_segment(
    audio_segment: AudioSegment,
    rate: float = 1,
    volume: float = 0,
    pitch: int = 0,
    sr: int = 24000,
) -> AudioSegment:
    # Adjust rate (speed)
    if rate != 1:
        audio_segment = effects.speedup(audio_segment, playback_speed=rate)

    # Adjust volume
    if volume != 0:
        audio_segment = audio_segment + volume

    # Adjust pitch
    if pitch != 0:
        audio_segment = audio_segment._spawn(
            audio_segment.raw_data,
            overrides={
                "frame_rate": int(audio_segment.frame_rate * (2.0 ** (pitch / 12.0)))
            },
        ).set_frame_rate(sr)

    return audio_segment


def apply_prosody_to_audio_data(
    audio_data: np.ndarray,
    rate: float = 1,
    volume: float = 0,
    pitch: int = 0,
    sr: int = 24000,
) -> np.ndarray:
    audio_segment = ndarray_to_segment(audio_data, sr)

    audio_segment = apply_prosody_to_audio_segment(
        audio_segment, rate=rate, volume=volume, pitch=pitch, sr=sr
    )

    processed_audio_data = np.array(audio_segment.get_array_of_samples())

    return processed_audio_data


def apply_normalize(
    audio_data: np.ndarray,
    headroom: float = 1,
    sr: int = 24000,
):
    segment = ndarray_to_segment(audio_data, sr)
    segment = effects.normalize(seg=segment, headroom=headroom)

    return pydub_to_np(segment)


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
