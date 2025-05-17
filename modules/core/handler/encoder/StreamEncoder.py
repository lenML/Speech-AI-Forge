import io
import logging
import os
import queue
import subprocess
import threading
import wave
from time import sleep

import pydub
import pydub.utils

logger = logging.getLogger(__name__)


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
        self.header = None
        self.timeout = 0.1

        self.channels = 1
        self.sample_width = 2
        self.sample_rate = 24000
        self.stderr_thread = None

    def set_header(
        self, *, frame_input=b"", channels=1, sample_width=2, sample_rate=24000
    ):
        """
        基本上只需要改 sample_rate 因为我们输入的都是 pcm s16le (int16)
        """
        self.channels = channels
        self.sample_width = sample_width
        self.sample_rate = sample_rate

        logger.info(
            f"StreamEncoder header set, channels: {channels}, sample_width: {sample_width}, sample_rate: {sample_rate}"
        )

    def write_header_data(self):
        if self.header:
            return
        header_bytes = wave_header_chunk(
            channels=self.channels,
            sample_width=self.sample_width,
            sample_rate=self.sample_rate,
        )
        self.header = header_bytes
        self.write(header_bytes)

        logger.info(
            f"StreamEncoder header written, channels: {self.channels}, sample_width: {self.sample_width}, sample_rate: {self.sample_rate}"
        )

    def open(
        self,
        format: str = "mp3",
        acodec: str = "libmp3lame",
        bitrate: str = "320k",
        input_dtype: str = "s16le",  # s16le or s32le
    ):
        """
        打开编码器

        :param format: 输出格式
        :param acodec: 输出编码器
        :param bitrate: 输出比特率
        :param input_dtype: 输入数据类型 s16le or s32le
        """
        encoder = self.encoder
        self.p = subprocess.Popen(
            [
                encoder,
                "-re",
                "-threads",
                str(os.cpu_count() or 4),
                # NOTE: 指定输入格式为 16 位 PCM
                "-f",
                input_dtype,
                "-ar",
                str(self.sample_rate),  # 输入采样率
                "-ac",
                str(self.channels),  # 输入单声道
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
            # NOTE: 这里设置为0可以低延迟解码，但是容易阻塞影响ffmpeg效率，所以最好还是设置上，因为编码相较于生成其实多不了多少时间
            bufsize=65536,
        )
        self.read_thread = threading.Thread(target=self._read_output)
        self.read_thread.daemon = True
        self.read_thread.start()
        self.stderr_thread = threading.Thread(target=self._read_stderr)
        self.stderr_thread.daemon = True
        self.stderr_thread.start()

        logger.info(
            f"StreamEncoder opened, encoder: {encoder}, format: {format}, acodec: {acodec}, bitrate: {bitrate}, sample_rate: {self.sample_rate}, channels: {self.channels}, sample_width: {self.sample_width}"
        )

    def _read_output(self):
        buffer_size = 65536
        stdout_buffer = io.BufferedReader(self.p.stdout, buffer_size=buffer_size)
        while self.p:
            peeked_data = stdout_buffer.peek(buffer_size)
            if peeked_data:
                data = stdout_buffer.read1(len(peeked_data))
                # logger.debug(f"Read {len(data)} bytes dynamically from stdout")
                self.output_queue.put(data)
                # print("queue: ", len(data))
            else:
                # 无数据时短暂休眠
                sleep(self.timeout)
        stdout_buffer.close()

    def _read_stderr(self):
        while self.p and self.p.stderr:
            line = self.p.stderr.readline()
            if line:
                logger.debug(f"FFmpeg stderr: {line.decode().strip()}")
            else:
                sleep(self.timeout)

    def write(self, data: bytes):
        if self.p is None:
            raise Exception("Encoder is not open")
        # print("write:", len(data))
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def read(self) -> bytes:
        try:
            data = self.output_queue.get(timeout=self.timeout)
            # print("read: ", len(data))
            return data
        except queue.Empty:
            return b""

    def read_all(self) -> bytes:
        data = b""

        def is_end():
            if self.p is None:
                return True
            if not isinstance(self.p, subprocess.Popen):
                return True
            return self.p.poll() is not None

        while not is_end() or not self.output_queue.empty():
            try:
                while not is_end() or not self.output_queue.empty():
                    data += self.output_queue.get(timeout=self.timeout)
                    # print("read_all: ", len(data))
            except queue.Empty:
                pass
            sleep(self.timeout)
        return data

    def close(self):
        if self.p is None:
            return
        if not self.p.stdin.closed:
            self.p.stdin.close()
        try:
            self.p.wait(timeout=10)  # 等待最多10秒
        except subprocess.TimeoutExpired:
            self.p.terminate()  # 超时则强制结束
            self.p.wait()  # 确保进程已结束

    def terminate(self):
        if self.p is None:
            return
        self.p.terminate()
        self.p = None

    # NOTE: 貌似因为多线程导致这个函数不会触发，所以需要手动调用 terminate
    def __del__(self):
        self.close()
        self.terminate()
