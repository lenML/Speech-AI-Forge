from dataclasses import dataclass

import numpy as np
from faster_whisper.transcribe import Segment, TranscriptionInfo, Word
from faster_whisper.vad import get_speech_timestamps
from tqdm import tqdm

from modules import config as global_config
from modules.core.handler.datacls.stt_model import STTConfig
from modules.core.models.stt.STTModel import STTModel, TranscribeResult
from modules.core.models.stt.whisper.whisper_dcls import SttResult, SttSegment, SttWord
from modules.core.models.stt.whisper.writer import get_writer
from modules.core.pipeline.processor import NP_AUDIO
from modules.utils.detect_lang import guess_lang
from modules.utils.monkey_tqdm import disable_tqdm


@dataclass(frozen=True, eq=False)
class STTChunkData:
    audio: NP_AUDIO
    start_s: int
    end_s: int


class RefrenceTranscript:
    def __init__(self, content: str):
        self.raw = content
        self.buffer = content
        self.raw_lines = content.split("\n")
        self.raw_lines = [
            line.strip() for line in self.raw_lines if len(line.strip()) > 0
        ]

    def dequeue(self, word: str) -> bool:
        """
        如果当前 buffer 开头为 word 即可出队
        如果不是则返回 False
        """
        word = word.strip()
        self.buffer = self.buffer.strip()
        if self.buffer.startswith(word):
            self.buffer = self.buffer[len(word) :].strip()
            return True
        else:
            return False

    def is_empty(self) -> bool:
        """
        判断当前 buffer 是否为空
        """
        return len(self.buffer) == 0

    def dequeue_content(self, content: str):
        """
        输入长文本，分词之后逐个调用dequeue直到False
        """
        words = content.split()
        for word in words:
            if not self.dequeue(word):
                return

class STTChunker:
    """
    对于长音频，asr模型一般效果都很差

    此类将会使用vad将长输入音频切分为小于30s的短音频，然后逐个进行asr识别
    然后拼接，产生最终的asr识别结果
    """

    def __init__(self, model: STTModel):
        self.model = model

    def get_chunk(self, audio: NP_AUDIO, start_s: int, end_s: int):
        sr, data = audio
        start = int(start_s * sr)
        end = int(end_s * sr)
        return STTChunkData(
            audio=(sr, data[start:end]),
            start_s=start_s,
            end_s=end_s,
        )

    def get_chunks(self, audio: NP_AUDIO):
        """
        根据vad结果，尽量将audio分为小于30s的短音频

        如果小于30s就尝试和后续的合并
        如果大于30s直接作为chunk
        """
        sr, data = audio
        index_to_s = lambda index: int(index / sr)
        get_duration_s = lambda start, end: index_to_s(end - start)
        duration_s = get_duration_s(0, len(data))
        MAX_DURATION = 30.0  # seconds

        if duration_s < MAX_DURATION:
            return [
                STTChunkData(
                    audio=audio,
                    start_s=0,
                    end_s=duration_s,
                )
            ]
        speech_timestamps = get_speech_timestamps(data.astype(np.float32))
        chunks: list[STTChunkData] = []

        buffer_start = None
        buffer_end = None

        for speech in speech_timestamps:
            start, end = speech["start"], speech["end"]
            speech_duration = get_duration_s(start, end)

            if speech_duration >= MAX_DURATION:
                if buffer_start is not None:
                    chunks.append(
                        STTChunkData(
                            audio=(sr, data[buffer_start:buffer_end]),
                            start_s=index_to_s(buffer_start),
                            end_s=index_to_s(buffer_end),
                        )
                    )
                    buffer_start = None
                    buffer_end = None
                # 大段，直接作为 chunk
                chunks.append(
                    STTChunkData(
                        audio=(sr, data[start:end]),
                        start_s=index_to_s(start),
                        end_s=index_to_s(end),
                    )
                )
                continue

            if buffer_start is None:
                buffer_start = start
                buffer_end = end
            else:
                # 尝试合并
                new_duration = get_duration_s(buffer_start, end)
                if new_duration <= MAX_DURATION:
                    buffer_end = end  # 合并
                else:
                    # 提交当前 buffer
                    chunks.append(
                        STTChunkData(
                            audio=(sr, data[buffer_start:buffer_end]),
                            start_s=index_to_s(buffer_start),
                            end_s=index_to_s(buffer_end),
                        )
                    )
                    buffer_start = start
                    buffer_end = end

        # 收尾处理剩余的 buffer
        if buffer_start is not None:
            chunks.append(
                STTChunkData(
                    audio=(sr, data[buffer_start:buffer_end]),
                    start_s=index_to_s(buffer_start),
                    end_s=index_to_s(buffer_end),
                )
            )

        # 根据 start_s 排序
        return chunks

    def merge_results(
        self, chunks: list[STTChunkData], results: list[SttResult]
    ) -> SttResult:
        merged_segments = []
        for chunk, result in zip(chunks, results):
            for segment in result.segments:
                # segment.start += chunk.start_s
                # segment.end += chunk.start_s
                merged_segments.append(
                    SttSegment(
                        start=segment.start + chunk.start_s,
                        end=segment.end + chunk.start_s,
                        text=segment.text,
                        words=(
                            [
                                SttWord(
                                    word=w.word,
                                    start=w.start + chunk.start_s,
                                    end=w.end + chunk.start_s,
                                )
                                for w in segment.words
                            ]
                            if segment.words
                            else None
                        ),
                    )
                )
        return SttResult(
            duration=-1,
            segments=merged_segments,
            language=results[0].language if len(results) != 0 else "zh",
        )

    def transcribe_to_result(self, audio: NP_AUDIO, config: STTConfig) -> SttResult:
        ref_script = RefrenceTranscript(config.refrence_transcript.strip())
        chunks = self.get_chunks(audio)
        results: list[SttResult] = []
        for chunk in tqdm(
            chunks,
            desc="Transcribing audio chunks",
            disable=global_config.runtime_env_vars.off_tqdm,
        ):
            config.refrence_transcript = ref_script.buffer
            result = self.model.transcribe_to_result(chunk.audio, config)

            # check last segement if broken
            # 检查最后一个识别是否是损坏的，如果只识别了一部分，而不是完整的一句，那么我们需要放弃这个识别，并调整下一个 chunk 的数据，包含错误识别的部分
            while (
                len(result.segments) != 0
                and len(ref_script.raw_lines) != 0
                and result.segments[-1].text not in ref_script.raw_lines
            ):
                # TODO: 这个识别好像有点问题，应该关注前一个句子来定位会好点
                result.segments = result.segments[:-1]

            chunk_idx = chunks.index(chunk)
            if len(result.segments) == 0:
                if chunk_idx + 1 < len(chunks):
                    # 如果这个片段没识别出来或者只能识别一部分，就直接合并到下一个片段
                    orig_next_chunk = chunks[chunk_idx + 1]
                    next_chunk = self.get_chunk(
                        audio=audio, start_s=chunk.start_s, end_s=orig_next_chunk.end_s
                    )
                    chunks[chunk_idx + 1] = next_chunk
                continue

            # 不管有没有 broken 都调整下一个 chunk 的范围
            next_start_s: int = result.segments[-1].end + chunk.start_s
            if chunk_idx + 1 < len(chunks):
                orig_next_chunk = chunks[chunk_idx + 1]
                next_chunk = self.get_chunk(
                    audio=audio, start_s=next_start_s, end_s=orig_next_chunk.end_s
                )
                chunks[chunk_idx + 1] = next_chunk

            results.append(result)

            result_content = ""
            for seg in result.segments:
                if seg.words:
                    for w in seg.words:
                        result_content += w.word + " "
                else:
                    result_content += seg.text
            ref_script.dequeue_content(result_content)

        result = self.merge_results(chunks, results)
        return result

    def convert_result_with_format(self, config: STTConfig, result: SttResult) -> str:
        writer_options = {
            "highlight_words": config.highlight_words,
            "max_line_count": config.max_line_count,
            "max_line_width": config.max_line_width,
            "max_words_per_line": config.max_words_per_line,
        }

        format = config.format

        writer = get_writer(format.value)
        with disable_tqdm(enabled=global_config.runtime_env_vars.off_tqdm):
            output = writer.write(result=result, options=writer_options)

        return TranscribeResult(
            text=output,
            segments=writer.subtitles,
            language=result.language,
        )

    def transcribe(self, audio: NP_AUDIO, config: STTConfig) -> str:
        result = self.transcribe_to_result(audio, config)
        sr, data = audio
        result.duration = len(data) / sr
        return self.convert_result_with_format(config, result)
