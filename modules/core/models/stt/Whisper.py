import logging
import threading
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel as FasterWhisperModel
from whisper import audio
from whisper.tokenizer import get_tokenizer

from modules import config as global_config
from modules.core.handler.datacls.stt_model import STTConfig
from modules.core.models.stt.STTModel import STTModel, TranscribeResult
from modules.core.models.stt.whisper.whisper_dcls import WhisperTranscribeResult
from modules.core.models.stt.whisper.writer import get_writer
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.utils.monkey_tqdm import disable_tqdm

# ref https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-temperature
DEFAULT_TEMPERATURE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

whisper_tokenizer = get_tokenizer(multilingual=True)
number_tokens = [
    i
    for i in range(whisper_tokenizer.eot)
    if all(c in "0123456789" for c in whisper_tokenizer.decode([i]).removeprefix(" "))
]


class WhisperModel(STTModel):
    SAMPLE_RATE = audio.SAMPLE_RATE

    lock = threading.Lock()

    logger = logging.getLogger(__name__)

    model: Optional[FasterWhisperModel] = None

    def __init__(self, model_id: str):
        # example: `whisper.large` or `whisper` or `whisper.small`
        model_ver = model_id.lower().split(".")

        assert model_ver[0] == "whisper", f"Invalid model id: {model_id}"

        self.model_size = model_ver[1] if len(model_ver) > 1 else "large"
        # self.model_dir = Path("./models/whisper")
        self.model_dir = Path("./models/faster-whisper-large-v3")

        self.device = devices.get_device_for("whisper")
        self.dtype = devices.dtype

    def is_loaded(self) -> bool:
        return WhisperModel.model is not None

    def load(self):
        if WhisperModel.model is None:
            with self.lock:
                self.logger.info(f"Loading Whisper model [{self.model_size}]...")

                # WhisperModel.model = FasterWhisperModel(
                #     model_size_or_path=self.model_size,
                #     download_root=str(self.model_dir),
                #     device=self.device.type,
                #     compute_type=(
                #         "float16" if self.dtype == torch.float16 else "float32"
                #     ),
                # )
                WhisperModel.model = FasterWhisperModel(
                    model_size_or_path=str(self.model_dir),
                    local_files_only=True,
                    device=self.device.type,
                    compute_type=(
                        "float16" if self.dtype == torch.float16 else "float32"
                    ),
                )
                self.logger.info("Whisper model loaded.")
        return WhisperModel.model

    @devices.after_gc()
    def unload(self):
        if WhisperModel.model is None:
            return
        with self.lock:
            del self.model
            self.model = None
            WhisperModel.unload()
            del WhisperModel.model
            WhisperModel.model = None

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

    def transcribe_to_result(
        self, audio: NP_AUDIO, config: STTConfig
    ) -> WhisperTranscribeResult:
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

        segments, info = model.transcribe(
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
        )

        return WhisperTranscribeResult(segments=segments, info=info)

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
            info=result.info._asdict(),
        )

    def transcribe(self, audio: NP_AUDIO, config: STTConfig) -> TranscribeResult:
        result = self.transcribe_to_result(audio=audio, config=config)
        result_formated = self.convert_result_with_format(config=config, result=result)
        return result_formated


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
