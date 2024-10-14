import io
import queue
import subprocess
import threading
import wave
from time import sleep

import pydub
import pydub.utils


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


class StreamEncoder:
    def __init__(self) -> None:
        self.encoder = pydub.utils.get_encoder_name()
        self.p: subprocess.Popen = None
        self.output_queue = queue.Queue()
        self.read_thread = None
        self.chunk_size = 1024
        self.header = None
        self.timeout = 0.1

        self.channels = 1
        self.sample_width = 2
        self.sample_rate = 24000

    def set_header(
        self, *, frame_input=b"", channels=1, sample_width=2, sample_rate=24000
    ):
        if self.header:
            return
        header_bytes = wave_header_chunk(
            frame_input, channels, sample_width, sample_rate
        )
        self.header = header_bytes
        self.write(header_bytes)

    def open(
        self, format: str = "mp3", acodec: str = "libmp3lame", bitrate: str = "320k"
    ):
        encoder = self.encoder
        self.p = subprocess.Popen(
            [
                encoder,
                "-re",
                "-threads",
                "4",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-f",
                format,
                "-acodec",
                acodec,
                "-b:a",
                bitrate,
                "-flush_packets",
                "1",
                "-max_delay",
                "0",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # 禁用缓冲
        )
        self.read_thread = threading.Thread(target=self._read_output)
        self.read_thread.daemon = True
        self.read_thread.start()

    def _read_output(self):
        while self.p:
            data: bytes = self.p.stdout.read(self.chunk_size)
            if data:
                self.output_queue.put(data)
            else:
                sleep(self.timeout)

    def write(self, data: bytes):
        if self.p is None:
            raise Exception("Encoder is not open")
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def read(self) -> bytes:
        if self.p is None:
            raise Exception("Encoder is not open")
        try:
            return self.output_queue.get(timeout=self.timeout)
        except queue.Empty:
            return b""

    def read_all(self) -> bytes:
        if self.p is None:
            raise Exception("Encoder is not open")
        data = b""

        def is_end():
            if not isinstance(self.p, subprocess.Popen):
                return True
            return self.p.poll() is not None

        while not is_end() or not self.output_queue.empty():
            try:
                while not is_end() or not self.output_queue.empty():
                    data += self.output_queue.get(timeout=self.timeout)
            except queue.Empty:
                pass
            sleep(self.timeout)
        return data

    def close(self):
        if self.p is None:
            return
        if not self.p.stdin.closed:
            self.p.stdin.close()
        self.p.wait()

    def terminate(self):
        if self.p is None:
            return
        self.p.terminate()
        self.p = None

    # NOTE: 貌似因为多线程导致这个函数不会触发，所以需要手动调用 terminate
    def __del__(self):
        self.close()
        self.terminate()
