import logging
import threading
from pathlib import Path
from typing import Optional

import jieba
import librosa
import numpy as np
import stable_whisper
import torch
from faster_whisper import WhisperModel as FasterWhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo, Word
from whisper import audio
from whisper.tokenizer import get_tokenizer

from modules import config as global_config
from modules.core.handler.datacls.stt_model import STTConfig
from modules.core.models.stt.STTModel import STTModel, TranscribeResult
from modules.core.models.stt.whisper.whisper_dcls import SttResult, SttSegment, SttWord
from modules.core.models.stt.whisper.writer import get_writer
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.utils.detect_lang import guess_lang
from modules.utils.monkey_tqdm import disable_tqdm

# ref https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-temperature
DEFAULT_TEMPERATURE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

whisper_tokenizer = get_tokenizer(multilingual=True)
number_tokens = [
    i
    for i in range(whisper_tokenizer.eot)
    if all(c in "0123456789" for c in whisper_tokenizer.decode([i]).removeprefix(" "))
]


def st_word2stt_word(word: stable_whisper.WordTiming | Word) -> SttWord:
    return SttWord(start=word.start, end=word.end, word=word.word)


def st_result2result(result: stable_whisper.WhisperResult, duration: int) -> SttResult:
    segments = [
        SttSegment(
            text=seg.text,
            start=seg.start,
            end=seg.end,
            words=[st_word2stt_word(w) for w in seg.words],
        )
        for seg in result.segments
    ]
    language = result.language

    return SttResult(segments=segments, language=language, duration=duration)


def fw_result2result(result: tuple, duration: int):
    segments, info = result
    language = info.language
    segments = [
        SttSegment(
            text=seg.text,
            start=seg.start,
            end=seg.end,
            words=[st_word2stt_word(w) for w in seg.words],
        )
        for seg in segments
    ]
    return SttResult(
        segments=segments, language=language, duration=get_audio_duration(audio)
    )


def get_audio_duration(audio: NP_AUDIO) -> float:
    sr, data = audio
    return data.shape[0] / sr


def wordify_transcript(content: str) -> str:
    """
    每一行都使用 jieba 分词，并且使用空格连接，为对齐过程提供分词语义
    """
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        new_lines.append(" ".join(jieba.lcut(line)))
    return "\n".join(new_lines)


model_dir_mapping = {
    "large": Path("./models/faster-whisper-large-v3"),
    "turbo": Path("./models/faster-whisper-large-v3-turbo-ct2"),
}

class WhisperModel(STTModel):
    SAMPLE_RATE = audio.SAMPLE_RATE

    lock = threading.Lock()

    logger = logging.getLogger(__name__)

    model: Optional[FasterWhisperModel] = None

    def __init__(self, model_id: str):
        super().__init__(model_id)
        # example: `whisper.large` or `whisper` or `whisper.small`
        model_ver = model_id.lower().split(".")

        assert model_ver[0] == "whisper", f"Invalid model id: {model_id}"

        self.model_size = model_ver[1] if len(model_ver) > 1 else "large"
        # self.model_dir = Path("./models/whisper")
        self.model_dir = (
            model_dir_mapping[self.model_size]
            if self.model_size in model_dir_mapping
            else model_dir_mapping["large"]
        )

    def get_device(self):
        return devices.get_device_for("whisper")

    def is_loaded(self) -> bool:
        return WhisperModel.model is not None

    def load(self):
        device = self.get_device()
        dtype = self.get_dtype()
        if WhisperModel.model is None:
            with self.lock:
                self.logger.info(
                    f"Loading Whisper model [{device.type}:{dtype}] [{self.model_size}]..."
                )
                WhisperModel.model = FasterWhisperModel(
                    model_size_or_path=str(self.model_dir),
                    download_root=str(self.model_dir),
                    device=device.type,
                    compute_type=("float16" if dtype == torch.float16 else "float32"),
                    local_files_only=True,
                )
                WhisperModel.model = stable_whisper.load_faster_whisper(
                    model_size_or_path=str(self.model_dir),
                    local_files_only=True,
                    device=device.type,
                    compute_type=("float16" if dtype == torch.float16 else "float32"),
                )
                self.logger.info("Whisper model loaded.")
        return WhisperModel.model

    @devices.after_gc()
    def unload(self):
        if WhisperModel.model is None:
            return
        with self.lock:
            self.model = None
            del WhisperModel.model
            WhisperModel.model = None
            del self.model

    def resample_audio(self, audio: NP_AUDIO):
        sr, data = audio

        if sr == self.SAMPLE_RATE:
            return sr, data
        data = librosa.resample(data, orig_sr=sr, target_sr=self.SAMPLE_RATE)
        return self.SAMPLE_RATE, data

    def ensure_float32(self, audio: NP_AUDIO):
        sr, data = audio
        if data.dtype == np.int16:
            data = data.astype(np.float32)
            data /= np.iinfo(np.int16).max
        elif data.dtype == np.int32:
            data = data.astype(np.float32)
            data /= np.iinfo(np.int32).max
        elif data.dtype == np.float64:
            data = data.astype(np.float32)
        elif data.dtype == np.float32:
            pass
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")

        return sr, data

    def ensure_stereo_to_mono(self, audio: NP_AUDIO):
        sr, data = audio
        if data.ndim == 2:
            data = data.mean(axis=1)
        return sr, data

    def normalize_audio(self, audio: NP_AUDIO):
        audio = self.ensure_float32(audio=audio)
        audio = self.resample_audio(audio=audio)
        audio = self.ensure_stereo_to_mono(audio=audio)
        return audio

    def generate_transcribe(self, audio: NP_AUDIO, config: STTConfig) -> SttResult:
        prompt = config.prompt
        prefix = config.prefix

        language = config.language
        tempperature = config.temperature
        sample_len = config.sample_len
        best_of = config.best_of
        beam_size = config.beam_size
        patience = config.patience
        length_penalty = config.length_penalty

        _, audio_data = self.normalize_audio(audio=audio)
        # _, audio_data = audio

        if tempperature is None or tempperature <= 0:
            tempperature = DEFAULT_TEMPERATURE

        model = self.load()

        # 这里必须 disable tqdm ，因为 stable_whisper 似乎会抛出 gradio 不支持的 progress...
        with disable_tqdm(
            enabled=global_config.runtime_env_vars.is_webui
            or global_config.runtime_env_vars.off_tqdm
        ):
            result = model.transcribe(
                audio=audio_data,
                language=language,
                initial_prompt=prompt,
                prefix=prefix,
                # sample_len=sample_len,
                temperature=tempperature or DEFAULT_TEMPERATURE,
                best_of=best_of or 1,
                beam_size=beam_size or 5,
                patience=patience or 1,
                length_penalty=length_penalty or 1,
                word_timestamps=True,
                suppress_tokens=[-1] + number_tokens,
                # fp16=self.dtype == torch.float16,
                # vad_filter=True,
            )
        duration = get_audio_duration(audio)
        if isinstance(result, tuple):
            # 兼容原始 faster_whisper
            return fw_result2result(result, duration)
        elif isinstance(result, stable_whisper.WhisperResult):
            result = st_result2result(result, duration)
            return result
        else:
            raise ValueError(f"Unknown result type: {type(result)}")

    def transcribe_to_result(self, audio: NP_AUDIO, config: STTConfig) -> SttResult:
        has_ref = config.refrence_transcript.strip() != ""
        if has_ref:
            result = self.force_align(audio=audio, config=config)
        else:
            result = self.generate_transcribe(audio=audio, config=config)
        return result

    def force_align_after_refine(
        self, result: stable_whisper.WhisperResult, refrence_transcript: str
    ) -> stable_whisper.WhisperResult:
        """
        根据传入的 refrence_transcript 再次 refine

        这里 refine 的目标是基于换行分割、合并 segment
        """
        refrence_segments = refrence_transcript.split("\n")
        refrence_segments = [s for s in refrence_segments if s.strip() != ""]
        refrence_segments = [s for s in refrence_segments if len(s) > 0]
        is_chinese = guess_lang(refrence_transcript) == "zh"

        words: list[stable_whisper.WordTiming] = []
        for seg in result.segments:
            for w in seg.words:
                if is_chinese:
                    w.word = w.word.strip()
                words.append(w)
        word_index = 0

        def next_word():
            nonlocal word_index
            index = word_index
            word_index += 1
            return words[index] if index < len(words) else None

        refined_segments = []
        for ref_seg in refrence_segments:
            seg_words: list[stable_whisper.WordTiming] = []
            ref_buf = ref_seg.strip()
            while ref_buf != "":
                w = next_word()
                if w is None:
                    break
                word = w.word.strip()
                if ref_buf.startswith(word):
                    seg_words.append(w)
                    ref_buf = ref_buf[len(word) :].strip()
                else:
                    break
            if len(seg_words) == 0:
                continue
            start = seg_words[0].start
            end = seg_words[-1].end
            seq = "" if is_chinese else " "
            real_text = seq.join([w.word.strip() for w in seg_words])
            segment = stable_whisper.Segment(
                start=start,
                end=end,
                text=real_text,
                words=seg_words,
            )
            refined_segments.append(segment)

        result.segments = refined_segments
        return result

    def force_align(self, audio: NP_AUDIO, config: STTConfig) -> SttResult:
        """
        文稿匹配
        """
        model = self.load()
        prompt = config.prompt
        prefix = config.prefix

        language = config.language
        # 似乎都用不到...因为对齐的时候好像不需要模型输出？
        tempperature = config.temperature
        sample_len = config.sample_len
        best_of = config.best_of
        beam_size = config.beam_size
        patience = config.patience
        length_penalty = config.length_penalty
        refrence_transcript = config.refrence_transcript
        refrence_transcript = wordify_transcript(refrence_transcript)

        _, audio_data = self.normalize_audio(audio=audio)
        # 这里必须 disable tqdm ，因为 stable_whisper 似乎会抛出 gradio 不支持的 progress...
        with disable_tqdm(
            enabled=global_config.runtime_env_vars.is_webui
            or global_config.runtime_env_vars.off_tqdm
        ):
            # Perform alignment
            # NOTE: 下面这个三个函数是 stable_st 注入的
            aligned_result = model.align(
                audio=audio_data, text=refrence_transcript, language=language
            )
            aligned_words = model.align_words(
                audio=audio_data, result=aligned_result, language=language
            )
            result = model.refine(audio=audio_data, result=aligned_words)
            result = self.force_align_after_refine(
                result=result, refrence_transcript=refrence_transcript
            )
        result = st_result2result(result, get_audio_duration(audio))
        return result


if __name__ == "__main__":
    import json

    from pydub import AudioSegment
    from scipy.io import wavfile

    from modules.core.handler.datacls.stt_model import STTOutputFormat

    devices.reset_device()
    devices.dtype = torch.float32

    def pydub_to_numpy(audio_segment: AudioSegment):
        raw_data = audio_segment.raw_data
        sample_width = audio_segment.sample_width
        channels = audio_segment.channels
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        if channels > 1:
            audio_data = audio_data.reshape((-1, channels))
            audio_data = audio_data.mean(axis=1).astype(np.int16)
        return audio_data

    model = WhisperModel("whisper.large-v3")

    # sr1, wav1 = wavfile.read("./test_cosyvoice.wav")

    # print(f"Input audio sample rate: {sr1}")
    # print(f"Input audio shape: {wav1.shape}")
    # print(f"Input audio duration: {len(wav1) / sr1}s")

    # input_audio_path = "./input_audio_1.mp3"
    input_audio_path = "./test_cosyvoice.wav"
    audio1: AudioSegment = AudioSegment.from_file(input_audio_path, format="mp3")

    sr = audio1.frame_rate
    input_data = pydub_to_numpy(audio1)

    print(f"Input audio sample rate: {sr}")
    print(f"Input audio shape: {input_data.shape}")
    print(f"Input audio duration: {len(input_data) / sr}s")

    result = model.transcribe(
        audio=(sr, input_data),
        config=STTConfig(
            mid="whisper.large-v3", language="zh", format=STTOutputFormat.srt
        ),
    )

    # print(result.text)
    with open("test_whisper_result.json", "w") as f:
        json.dump(result.__dict__, f, ensure_ascii=False, indent=2)

    with open("test_whisper_result.srt", "w") as f:
        f.write(result.text)
