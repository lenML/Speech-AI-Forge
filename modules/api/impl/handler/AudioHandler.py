import base64
import io
import wave
from typing import Generator

import numpy as np
from pydub import AudioSegment

from modules.api.impl.model.audio_model import AudioFormat
from modules.utils.audio import ndarray_to_segment


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


wav_header = wave_header_chunk()


def read_to_wav(audio_data: np.ndarray, buffer: io.BytesIO):
    audio_data = audio_data / np.max(np.abs(audio_data))
    chunk = (audio_data * 32768).astype(np.int16)
    buffer.write(chunk.tobytes())
    return buffer


def align_audio(audio_data: np.ndarray, channels=1) -> np.ndarray:
    samples_per_frame = channels
    total_samples = len(audio_data)
    aligned_samples = total_samples - (total_samples % samples_per_frame)
    return audio_data[:aligned_samples]


def pad_audio_frame(audio_data: np.ndarray, frame_size=1152, channels=1) -> np.ndarray:
    samples_per_frame = frame_size * channels
    padding_needed = (
        samples_per_frame - len(audio_data) % samples_per_frame
    ) % samples_per_frame
    return np.pad(audio_data, (0, padding_needed), mode="constant")


class AudioHandler:
    def enqueue(self) -> tuple[np.ndarray, int]:
        raise NotImplementedError("Method 'enqueue' must be implemented by subclass")

    def enqueue_stream(self) -> Generator[tuple[np.ndarray, int], None, None]:
        raise NotImplementedError(
            "Method 'enqueue_stream' must be implemented by subclass"
        )

    def encode_audio(
        self, audio_data: np.ndarray, sample_rate: int, format: AudioFormat
    ) -> io.BytesIO:
        buffer = io.BytesIO()

        audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data = (audio_data * 32767).astype(np.int16)

        audio_segment: AudioSegment = ndarray_to_segment(
            audio_data, frame_rate=sample_rate
        )

        if format == AudioFormat.mp3:
            audio_segment.export(buffer, format="mp3")
            buffer.seek(0)
        elif format == AudioFormat.wav:
            audio_segment.export(buffer, format="wav")
            buffer.seek(len(wav_header))
        elif format == AudioFormat.ogg:
            # FIXME: 流式输出有 bug，会莫名其妙中断输出...
            audio_segment.export(buffer, format="ogg")
            buffer.seek(0)
        else:
            raise ValueError(f"Invalid audio format: {format}")

        return buffer

    def enqueue_to_stream(self, format: AudioFormat) -> Generator[bytes, None, None]:
        if format == AudioFormat.wav:
            yield wav_header

        for audio_data, sample_rate in self.enqueue_stream():
            yield self.encode_audio(audio_data, sample_rate, format).read()

        # print("AudioHandler: enqueue_to_stream done")

    # just for test
    def enqueue_to_stream_join(
        self, format: AudioFormat
    ) -> Generator[bytes, None, None]:
        if format == AudioFormat.wav:
            yield wav_header

        data = None
        for audio_data, sample_rate in self.enqueue_stream():
            data = audio_data if data is None else np.concatenate((data, audio_data))
        buffer = self.encode_audio(data, sample_rate, format)
        yield buffer.read()

    def enqueue_to_buffer(self, format: AudioFormat) -> io.BytesIO:
        audio_data, sample_rate = self.enqueue()
        buffer = self.encode_audio(audio_data, sample_rate, format)
        if format == AudioFormat.wav:
            buffer = io.BytesIO(wav_header + buffer.read())
        return buffer

    def enqueue_to_bytes(self, format: AudioFormat) -> bytes:
        buffer = self.enqueue_to_buffer(format=format)
        binary = buffer.read()
        return binary

    def enqueue_to_base64(self, format: AudioFormat) -> str:
        binary = self.enqueue_to_bytes(format=format)

        base64_encoded = base64.b64encode(binary)
        base64_string = base64_encoded.decode("utf-8")

        return base64_string
