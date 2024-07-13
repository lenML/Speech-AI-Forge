import io
import queue
import subprocess
import threading

import pydub
import pydub.utils


class StreamEncoder:
    def __init__(self) -> None:
        self.encoder = pydub.utils.get_encoder_name()
        self.p: subprocess.Popen = None
        self.output_queue = queue.Queue()
        self.read_thread = None
        self.chunk_size = 1024

    def open(
        self, format: str = "mp3", acodec: str = "libmp3lame", bitrate: str = "320k"
    ):
        encoder = self.encoder
        self.p = subprocess.Popen(
            [
                encoder,
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
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.read_thread = threading.Thread(target=self._read_output)
        self.read_thread.daemon = True
        self.read_thread.start()

    def _read_output(self):
        while self.p:
            data = self.p.stdout.read(self.chunk_size)
            if not data:
                break
            self.output_queue.put(data)

    def write(self, data: bytes):
        if self.p is None:
            raise Exception("Encoder is not open")
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def read(self, timeout=0.1) -> bytes:
        if self.p is None:
            raise Exception("Encoder is not open")
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def read_all(self, timeout=5) -> bytes:
        if self.p is None:
            raise Exception("Encoder is not open")
        data = b""
        while True:
            try:
                data += self.output_queue.get(timeout=timeout)
            except queue.Empty:
                break
        return data

    def close(self):
        if self.p is None:
            return
        self.p.stdin.close()
        self.p.wait()
        if self.read_thread:
            self.read_thread.join()
