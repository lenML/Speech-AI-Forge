import base64
import io
import numpy as np
import soundfile as sf

from modules.api.impl.model.audio_model import AudioFormat
from modules.api import utils as api_utils


class AudioHandler:
    def enqueue(self) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def enqueue_to_buffer(self, format: AudioFormat) -> io.BytesIO:
        audio_data, sample_rate = self.enqueue()

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="wav")
        buffer.seek(0)

        if format == AudioFormat.mp3:
            buffer = api_utils.wav_to_mp3(buffer)

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
