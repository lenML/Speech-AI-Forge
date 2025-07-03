import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt

# from funasr.models.sense_voice.model import SenseVoiceSmall
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from resampy.core import resample
from tqdm import tqdm

from modules.core.models.stt.STTModel import STTModel
from modules.core.models.stt.whisper.whisper_dcls import SttResult, SttSegment
from modules.devices import devices
from modules.utils import audio_utils

logger = logging.getLogger(__name__)

lang_dict = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|yue|>": "yue",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|nospeech|>": "nospeech",
}


def get_lang(text: str) -> str:
    for k, v in lang_dict.items():
        if k in text:
            return v
    return ""


class SenseVoiceModel(STTModel):
    def __init__(self):
        model_id = "sensevoice"
        super().__init__(model_id=model_id)

        self.model_dir = Path("./models/SenseVoiceSmall")
        self.asr_model: AutoModel = None
        self.vad_model: AutoModel = None

        self.max_single_segment_time_seconds = 30
        self.with_punct = True
        self.offset_in_seconds = -0.25

    def load(self):
        if self.asr_model:
            return self.asr_model
        self.asr_model = AutoModel(
            model=str(self.model_dir),
            vad_model=None,  # We'll handle VAD separately
            punc_model=None,
            ban_emo_unks=True,
            device=str(self.get_device()),
            # 不确定这个dtype能不能用
            dtype=str(self.get_dtype()),
            disable_update=True,
        )
        vad_model_dir = Path("./models/fsmn-vad")
        self.vad_model = AutoModel(
            # 如果有预先下载就使用 Models 下面的，如果没有就用它内部的自动下载逻辑，会下载到 ~/.cache 目录下面
            model=str(vad_model_dir) if vad_model_dir.exists() else "fsmn-vad",
            max_single_segment_time=self.max_single_segment_time_seconds * 1000,
            disable_update=True,
        )
        logger.info(f"Loaded SenseVoice model from {self.model_dir}")
        return self.asr_model

    @devices.after_gc()
    def unload(self):
        if self.asr_model:
            del self.asr_model
            self.asr_model = None
            logger.info(f"Unloaded SenseVoice model")

    def asr_transcribe(
        self,
        sr: int,
        speech: npt.NDArray,
        language: str = "auto",
    ) -> tuple[List[SttSegment], str]:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_file (str): Path to audio file

        Returns:
            List[Segment]: List of transcription results

        参考：
        https://github.com/hon9kon9ize/yuesub-api/blob/main/transcriber/Transcriber.py
        """

        # if self.use_denoiser:
        #     logger.info("Denoising speech...")
        #     speech, _ = denoiser(speech, sr)

        if sr != 16_000:
            speech = resample(speech, sr, 16_000, filter="kaiser_best", parallel=True)

        # Get VAD segments
        logger.info("Segmenting speech...")

        start_time = time.time()
        vad_results = self.vad_model.generate(input=speech, disable_pbar=True)
        logger.info("VAD took %.2f seconds", time.time() - start_time)

        if not vad_results or not vad_results[0]["value"]:
            return [], language

        vad_segments = vad_results[0]["value"]

        # Process each segment
        results: list[SttSegment] = []

        start_time = time.time()
        seg_idx = 0
        languages: list[str] = []
        for segment in tqdm(vad_segments, desc="Transcribing"):
            start_sample = int(segment[0] * 16)  # Convert ms to samples
            end_sample = int(segment[1] * 16)
            segment_audio = speech[start_sample:end_sample]

            # Get ASR results for segment
            asr_result = self.asr_model.generate(
                input=segment_audio,
                language=language,
                use_itn=self.with_punct,
                disable_pbar=True,
            )

            if not asr_result:
                continue

            start_time = max(0, segment[0] / 1000.0 + self.offset_in_seconds)
            end_time = segment[1] / 1000.0 + self.offset_in_seconds

            text = asr_result[0]["text"]
            lang = get_lang(text)
            languages.append(lang)

            # Convert ASR result to TranscribeResult format
            segment_result = SttSegment(
                text=rich_transcription_postprocess(text),
                start=start_time,  # Convert ms to seconds
                end=end_time,
            )
            results.append(segment_result)
            seg_idx += 1

        logger.info("ASR took %.2f seconds", time.time() - start_time)

        # 返回命中最多的 lang
        hit_lang = max(set(languages), key=languages.count)
        return results, hit_lang

    # TODO: config 没有用上
    def transcribe_to_result(self, audio, config):
        self.load()

        sr, data = audio
        # 这个模型只有 language 能用上，其他参数用不上
        language = config.language

        ref_text = config.refrence_transcript
        if ref_text.strip():
            # 这个模型不支持
            logger.warning("SenseVoiceModel doesn't support refrence_transcript")
            ref_text = ""

        segments, language = self.asr_transcribe(sr, data, language=language)
        return SttResult(segments=segments, language=language, duration=len(data) / sr)


if __name__ == "__main__":
    import json

    model = SenseVoiceModel()

    input_audio_path = "tests/test_inputs/chattts_out1.wav"
    audio = audio_utils.load_audio(input_audio_path)
    res = model.transcribe_to_result(audio, {})
    print(json.dumps(res, ensure_ascii=False, indent=2, default=lambda o: o.__dict__))
