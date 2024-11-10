import logging
import re
import threading
from typing import Generator, Optional
import torch
import torchaudio
import numpy as np
import tempfile
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment, silence
from modules.core.models.tts.F5.F5Annotation import F5Annotation
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.repos_static.F5TTS.f5_tts.model import CFM, UNetT, DiT, MMDiT
from modules.repos_static.F5TTS.f5_tts.model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
    save_spectrogram,
)
from transformers import pipeline
import soundfile as sf
import tomli
import argparse
import tqdm
from pathlib import Path
import codecs
from vocos.feature_extractors import FeatureExtractor, EncodecFeatures

from modules.core.models.TTSModel import TTSModel
from modules.utils.SeedContext import SeedContext

logger = logging.getLogger(__name__)

# --------------------- Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0
# fix_duration = 27  # None or float (duration in seconds)
fix_duration = None
F5TTS_model_cfg = dict(
    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
)


class F5TtsModel(TTSModel):
    load_lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__("f5-tts")

        self.model_path = Path("models/F5-TTS/F5TTS_Base/model_1200000.safetensors")
        self.vocos_path = Path("models/vocos-mel-24khz")

        self.model: Optional[CFM] = None
        self.vocos: Optional[Vocos] = None

        self.device = devices.get_device_for("f5-tts")

        self.annotation = F5Annotation()

    def check_files(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.vocos_path.exists():
            raise FileNotFoundError(f"Vocos model file not found: {self.vocos_path}")

    def is_downloaded(self) -> bool:
        return self.model_path.exists() and self.vocos_path.exists()

    def load(self) -> tuple[CFM, Vocos]:
        self.check_files()

        with self.load_lock:
            if self.model is None:
                self.model = self._load_f5()
                self.vocos = self._load_vocos()
        return self.model, self.vocos

    def _load_f5(self):
        logger.info("Loading F5-TTS model...")
        ckpt_path = self.model_path
        model_cls = DiT
        model_cfg = F5TTS_model_cfg
        device = self.device

        vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
        model = CFM(
            transformer=model_cls(
                **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
            ),
            mel_spec_kwargs=dict(
                target_sample_rate=target_sample_rate,
                n_mel_channels=n_mel_channels,
                hop_length=hop_length,
            ),
            odeint_kwargs=dict(
                method=ode_method,
            ),
            vocab_char_map=vocab_char_map,
        ).to(device)

        model = load_checkpoint(model, str(ckpt_path), str(device), use_ema=True)
        logger.info("Loaded F5-TTS model.")
        return model

    def _load_vocos(self):
        logger.info("Loading Vocos model...")
        config_path = self.vocos_path / "config.yaml"
        model_path = self.vocos_path / "pytorch_model.bin"
        model = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(model.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in model.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Loaded Vocos model.")
        return model

    def unload(self) -> None:
        if self.model is None:
            return
        del self.model
        del self.vocos
        self.model = None
        self.vocos = None
        devices.do_gc()
        logger.info("F5-TTS model unloaded.")

    def get_sample_rate(self) -> int:
        return target_sample_rate

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        # TODO: 缓存
        self.load()

        sr = self.get_sample_rate()

        texts = [segment.text for segment in segments]

        seg0 = segments[0]
        # NOTE: 虽然用不到这些参数...但是还是列出来先
        top_P = seg0.top_p
        top_K = seg0.top_k
        temperature = seg0.temperature
        # repetition_penalty = seg0.repetition_penalty
        # max_new_token = seg0.max_new_token
        prompt = seg0.prompt
        prompt1 = seg0.prompt1
        prompt2 = seg0.prompt2
        prefix = seg0.prefix
        # use_decoder = seg0.use_decoder
        seed = seg0.infer_seed
        chunk_size = context.infer_config.stream_chunk_size

        ref_wav, ref_txt = self.get_ref_wav(seg0)
        if ref_wav is None:
            # NOTE: 必须要有 reference audio
            raise RuntimeError("Reference audio not found.")

        ref_audio = (sr, ref_wav)

        with SeedContext(seed=seg0.infer_seed):
            generated_waves = self.infer_batch(
                ref_audio=ref_audio, ref_text=ref_txt, gen_text_batches=texts
            )

        return [(sr, generated_wave) for generated_wave in generated_waves]

    def infer_batch(
        self, ref_audio: NP_AUDIO, ref_text: str, gen_text_batches: list[str]
    ):
        device = self.device
        ema_model = self.model
        vocos = self.vocos
        sr, audio = ref_audio

        if ema_model is None:
            raise RuntimeError("F5-TTS model not loaded.")
        if vocos is None:
            raise RuntimeError("Vocos model not loaded.")

        audio = torch.Tensor(audio).unsqueeze(0).to(device)

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            rms = rms.to(audio.device)
            audio = audio * target_rms / rms
        audio = audio.to(device)

        generated_waves: list[np.ndarray] = []
        # NOTE: 这个现在用不到
        spectrograms = []

        for i, gen_text in enumerate(tqdm.tqdm(gen_text_batches)):
            # Prepare the text
            if len(ref_text[-1].encode("utf-8")) == 1:
                ref_text = ref_text + " "
            text_list = [ref_text + gen_text]

            # final_text_list = convert_char_to_pinyin(text_list)
            final_text_list = [
                self.annotation.convert_to_pinyin(text) for text in text_list
            ]
            # print(final_text_list)

            # Calculate duration
            ref_audio_len = audio.shape[-1] // hop_length
            zh_pause_punc = r"。，、；：？！"
            ref_text_len = len(ref_text.encode("utf-8")) + 3 * len(
                re.findall(zh_pause_punc, ref_text)
            )
            gen_text_len = len(gen_text.encode("utf-8")) + 3 * len(
                re.findall(zh_pause_punc, gen_text)
            )
            duration = ref_audio_len + int(
                ref_audio_len / ref_text_len * gen_text_len / speed
            )

            # inference
            with torch.inference_mode():
                generated, _ = ema_model.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )

            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
            generated_wave = vocos.decode(generated_mel_spec.cpu())
            if rms < target_rms:
                rms = rms.to(generated_wave.device)
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

        return generated_waves

    def generate_batch_stream(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> Generator[list[NP_AUDIO], None, None]:
        # NOTE: 不支持 stream
        generated_waves = self.generate_batch(segments=segments, context=context)
        yield generated_waves
