import io
import typing
import wave
import struct
import numpy as np
import pydub
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)


def convert_bytes_to_wav_bytes(audio_bytes: bytes) -> typing.Tuple[bytes, int]:
    """
    尝试将字节数据解码为音频，并返回 WAV 格式的字节数据和采样率。

    Args:
        audio_bytes: bytes 类型的音频数据.

    Returns:
        一个元组 (wav_bytes, sample_rate) 如果转换成功.

    Raises:
        Exception: 如果转换失败，抛出异常.
    """

    try:
        # 1. 尝试使用 pydub 自动检测格式并转换
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))  # 让 pydub 自动推断格式
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_bytes = wav_io.getvalue()
        sample_rate = audio.frame_rate  # 获取采样率
        return wav_bytes, sample_rate
    except Exception as e:
        logger.warning(f"pydub 转换失败: {e}")

    try:
        # 2. 尝试原始 PCM 解码 (假设是未经压缩的 PCM)
        # 假设采样率 44100Hz, 16-bit, 单声道 (如果不是，需要调整参数)
        sample_rate = 44100
        sample_width = 2  # 2 bytes = 16 bits
        channels = 1

        # 检查数据长度是否是 sample_width 的倍数
        if len(audio_bytes) % sample_width != 0:
            raise ValueError("字节数据长度不是 sample width 的倍数，可能不是原始 PCM")

        # 将字节数据转换为数值
        num_samples = len(audio_bytes) // sample_width
        samples = struct.unpack(
            f"<{num_samples}h", audio_bytes
        )  # '<h' 表示小端 16-bit signed short

        # 创建 WAV 格式的字节数据
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(
                struct.pack(f"<{len(samples)}h", *samples)
            )  # 将数值转换回字节数据并写入
        wav_bytes = wav_io.getvalue()
        return wav_bytes, sample_rate
    except Exception as e:
        logger.warning(f"原始 PCM 解码失败: {e}")

    raise Exception("所有转换尝试都失败")


# 示例用法:
if __name__ == "__main__":

    """
    配置日志
    """
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为INFO
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 假设你有一个名为 `audio_data` 的 bytes 变量
    # 这里我们创建一个简单的示例 bytes 数据，你可以替换成你的实际数据
    # 这是一个假的 PCM 数据，仅仅用于演示目的
    sample_rate = 44100
    duration = 1  # 秒
    frequency = 440  # Hz (A4 音符)
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, False)
    amplitude = np.iinfo(np.int16).max
    noise = amplitude * np.sin(2 * np.pi * frequency * t)
    pcm_data = noise.astype(np.int16).tobytes()

    audio_data = pcm_data  # 替换成你的 bytes 数据

    try:
        wav_bytes, sr = convert_bytes_to_wav_bytes(audio_data)
        print("音频转换完成!")
        print(f"WAV 字节数据长度: {len(wav_bytes)}")
        print(f"采样率: {sr}")

        # 可选: 将 WAV 字节数据保存到文件进行验证
        with open("output.wav", "wb") as f:
            f.write(wav_bytes)
        print("WAV 字节数据已保存到 output.wav")

    except Exception as e:
        print(f"音频转换失败: {e}")
        print(f"错误原因: {e}")
