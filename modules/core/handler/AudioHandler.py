import base64
import io
import wave
from typing import AsyncGenerator, Generator

import numpy as np
from fastapi import Request

from modules.core.handler.encoder.StreamEncoder import StreamEncoder
from modules.core.handler.encoder.WavFile import WAVFileBytes
from modules.core.handler.datacls.audio_model import AudioFormat
from modules.core.handler.encoder.encoders import (
    AacEncoder,
    FlacEncoder,
    Mp3Encoder,
    OggEncoder,
    WavEncoder,
)
from modules.core.models.zoo.ChatTTSInfer import ChatTTSInfer
from modules.core.pipeline.processor import NP_AUDIO


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


# NOTE: 这个可能只适合 chattts
wav_header = wave_header_chunk()


def remove_wav_bytes_header(wav_bytes: bytes):
    wav_file = WAVFileBytes(wav_bytes=wav_bytes)
    wav_file.read()
    return wav_file.get_body_data()


def read_np_to_wav(audio_data: np.ndarray) -> bytes:
    audio_data: np.ndarray = audio_data / np.max(np.abs(audio_data))
    audio_data = (audio_data * 32767).astype(np.int16)
    return audio_data.tobytes()


class AudioHandler:
    def enqueue(self) -> NP_AUDIO:
        raise NotImplementedError("Method 'enqueue' must be implemented by subclass")

    def enqueue_stream(self) -> Generator[NP_AUDIO, None, None]:
        raise NotImplementedError(
            "Method 'enqueue_stream' must be implemented by subclass"
        )

    def get_encoder(self, format: AudioFormat) -> StreamEncoder:
        # TODO 这里可以增加 编码器配置
        if format == AudioFormat.wav:
            encoder = WavEncoder()
        elif format == AudioFormat.mp3:
            encoder = Mp3Encoder()
        elif format == AudioFormat.flac:
            encoder = FlacEncoder()
        # OGG 和 ACC 编码有问题，不知道为啥
        # FIXME: BrokenPipeError: [Errno 32] Broken pipe
        elif format == AudioFormat.acc:
            encoder = AacEncoder()
        # FIXME: BrokenPipeError: [Errno 32] Broken pipe
        elif format == AudioFormat.ogg:
            encoder = OggEncoder()
        else:
            raise ValueError(f"Unsupported audio format: {format}")
        encoder.open()
        encoder.write(wav_header)

        return encoder

    def enqueue_to_stream(self, format: AudioFormat) -> Generator[bytes, None, None]:
        encoder = self.get_encoder(format)
        chunk_data = bytes()
        # NOTE sample_rate 写在文件头里了所以用不到
        for sample_rate, audio_data in self.enqueue_stream():
            audio_bytes = read_np_to_wav(audio_data=audio_data)
            encoder.write(audio_bytes)
            chunk_data = encoder.read()
            while len(chunk_data) > 0:
                yield chunk_data
                chunk_data = encoder.read()

        encoder.close()
        while len(chunk_data) > 0:
            yield chunk_data
            chunk_data = encoder.read()

    async def enqueue_to_stream_with_request(
        self, request: Request, format: AudioFormat
    ) -> AsyncGenerator[bytes, None]:
        for chunk in self.enqueue_to_stream(format=AudioFormat(format)):
            disconnected = await request.is_disconnected()
            if disconnected:
                # TODO: 这个逻辑应该传递给 zoo
                ChatTTSInfer.interrupt()
                break
            yield chunk

    # just for test
    def enqueue_to_stream_join(
        self, format: AudioFormat
    ) -> Generator[bytes, None, None]:
        encoder = self.get_encoder(format)
        chunk_data = bytes()
        for sample_rate, audio_data in self.enqueue_stream():
            audio_bytes = read_np_to_wav(audio_data=audio_data)
            encoder.write(audio_bytes)
            chunk_data = encoder.read()

        encoder.close()
        while len(chunk_data) > 0:
            yield chunk_data
            chunk_data = encoder.read()

    def enqueue_to_bytes(self, format: AudioFormat) -> bytes:
        encoder = self.get_encoder(format)
        sample_rate, audio_data = self.enqueue()
        audio_bytes = read_np_to_wav(audio_data=audio_data)
        encoder.write(audio_bytes)
        encoder.close()
        return encoder.read_all()

    def enqueue_to_buffer(self, format: AudioFormat) -> io.BytesIO:
        audio_bytes = self.enqueue_to_bytes(format=format)
        return io.BytesIO(audio_bytes)

    def enqueue_to_base64(self, format: AudioFormat) -> str:
        binary = self.enqueue_to_bytes(format=format)

        base64_encoded = base64.b64encode(binary)
        base64_string = base64_encoded.decode("utf-8")

        return base64_string
