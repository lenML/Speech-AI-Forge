from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from modules.core.handler.datacls.stt_model import STTConfig
from modules.core.models.stt.STTModel import STTModel, TranscribeResult
from modules.core.models.stt.whisper.whisper_dcls import WhisperTranscribeResult
from modules.core.models.stt.whisper.writer import get_writer
from modules.core.pipeline.processor import NP_AUDIO
from modules.utils.monkey_tqdm import disable_tqdm
from modules.utils.detect_lang import guess_lang
from modules.utils.monkey_tqdm import disable_tqdm
from modules import config as global_config
from faster_whisper.vad import get_speech_timestamps
from faster_whisper.transcribe import Segment, TranscriptionInfo, Word


@dataclass(frozen=True, eq=False)
class STTChunkData:
    audio: NP_AUDIO
    start_s: int
    end_s: int


class RefrenceTranscript:
    def __init__(self, content: str):
        self.raw = content
        self.buffer = content

    def dequeue(self, word: str) -> bool:
        """
        如果当前 buffer 开头为 word 即可出队
        如果不是则返回 False
        """
        word = word.strip()
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

        return chunks

    def merge_results(
        self, chunks: list[STTChunkData], results: list[WhisperTranscribeResult]
    ) -> WhisperTranscribeResult:
        merged_segments = []
        for chunk, result in zip(chunks, results):
            for segment in result.segments:
                # segment.start += chunk.start_s
                # segment.end += chunk.start_s
                merged_segments.append(
                    Segment(
                        id=segment.id,
                        seek=segment.seek,
                        start=segment.start + chunk.start_s,
                        end=segment.end + chunk.start_s,
                        text=segment.text,
                        tokens=segment.tokens,
                        temperature=segment.temperature,
                        avg_logprob=segment.avg_logprob,
                        compression_ratio=segment.compression_ratio,
                        no_speech_prob=segment.no_speech_prob,
                        words=[
                            Word(
                                word=w.word,
                                start=w.start + chunk.start_s,
                                end=w.end + chunk.start_s,
                                probability=w.probability,
                            )
                            for w in segment.words
                        ],
                    )
                )
        return WhisperTranscribeResult(
            duration=-1,
            segments=merged_segments,
            language=results[0].language,
        )

    def transcribe_to_result(
        self, audio: NP_AUDIO, config: STTConfig
    ) -> WhisperTranscribeResult:
        ref_script = RefrenceTranscript(config.refrence_transcript)
        chunks = self.get_chunks(audio)
        results: list[WhisperTranscribeResult] = []
        for chunk in tqdm(
            chunks,
            desc="Transcribing audio chunks",
            disable=global_config.runtime_env_vars.off_tqdm,
        ):
            config.refrence_transcript = ref_script.buffer
            result = self.model.transcribe_to_result(chunk.audio, config)
            results.append(result)

            result_content = ""
            for seg in result.segments:
                for w in seg.words:
                    result_content += w.word + " "
                # result_content += "\n"
            ref_script.dequeue_content(result_content)

        result = self.merge_results(chunks, results)
        return result

    def convert_result_with_format(
        self, config: STTConfig, result: WhisperTranscribeResult
    ) -> str:
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
