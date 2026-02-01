import io
import os
from pathlib import Path
import tempfile
from typing import Generator

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from modules.core.models.TTSModel import TTSModel
from modules.core.models.tts.IndexTTS.infer.infer_v2 import IndexTTS2
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.repos_static.index_tts.indextts.utils.front import TextTokenizer
from modules.utils.SeedContext import SeedContext
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from modules.repos_static.index_tts.indextts.s2mel.modules.bigvgan import bigvgan

import logging

logger = logging.getLogger(__name__)


# IndexTTS v2 的推理
class IndexTTSV2Model(TTSModel):
    model_id = "index-tts-v2"

    def __init__(self):
        super().__init__("index-tts-v2", "Index-TTS-2")
        self.tts: IndexTTS2 = None
        self.model_dir = Path("./models/Index-TTS-2")
        self.semantic_code_ckpt_path = Path(
            "./models/amphion/MaskGCT/semantic_codec/model.safetensors"
        )
        self.campplus_ckpt_path = Path(
            "./models/funasr/campplus/campplus_cn_common.bin"
        )
        self.bigvgan_path = Path("./models/nvidia/bigvgan_v2_22khz_80band_256x")
        self.extract_features_model_path = Path("./models/facebook/w2v-bert-2.0")
        self.tokenizer: TextTokenizer = None

    def is_loaded(self):
        return self.tts is not None

    def load_tokenizer(self):
        if self.tokenizer is None:
            cfg_path = self.model_dir / "config.yaml"
            cfg = OmegaConf.load(cfg_path)
            bpe_path = self.model_dir / cfg.dataset["bpe_model"]
            self.tokenizer = TextTokenizer(str(bpe_path))
        return self.tokenizer

    def encode(self, text):
        self.load_tokenizer()
        return self.tokenizer.encode(text)

    def decode(self, ids):
        self.load_tokenizer()
        return self.tokenizer.decode(ids)

    def get_sample_rate(self):
        # 来自 modules/repos_static/index_tts/checkpoints/config.yaml
        # NOTE: 其实应该从 config.yaml 里取，但是 v1 v1.5 都一样，所以直接写死得了，因为加载配置需要外部依赖库
        return 24000

    def is_loaded(self):
        return self.tts is not None

    def load(self):
        if self.tts:
            return
        cfg_path = self.model_dir / "config.yaml"

        extract_features_model: SeamlessM4TFeatureExtractor = (
            SeamlessM4TFeatureExtractor.from_pretrained(
                str(self.extract_features_model_path), local_files_only=True
            )
        )
        semantic_model = Wav2Vec2BertModel.from_pretrained(
            str(self.extract_features_model_path), local_files_only=True
        )
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            str(self.bigvgan_path), use_cuda_kernel=False
        )

        self.tts = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(self.model_dir),
            use_fp16=self.get_dtype() == torch.float16,
            # 这个好像可以加速，但是需要冷启动
            # TODO: 暂时不实现，有需要的自行打开即可
            use_cuda_kernel=False,
            device=self.get_device(),
            extract_features_model=extract_features_model,
            campplus_ckpt_path=str(self.campplus_ckpt_path),
            semantic_model=semantic_model,
            semantic_code_ckpt_path=str(self.semantic_code_ckpt_path),
            bigvgan_model=bigvgan_model,
        )

    @devices.after_gc()
    def unload(self):
        if self.tts is not None:
            del self.tts.gpt
            del self.tts.bigvgan
            del self.tts.extract_features
            del self.tts.s2mel
            del self.tts.campplus_model
            del self.tts.semantic_model
            self.tts = None

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[NP_AUDIO]:
        generator = self.generate_batch_stream(segments, context)
        return next(generator)

    def generate_batch_stream(
        self, segments, context
    ) -> Generator[list[NP_AUDIO], None, None]:
        cached = self.get_cache(segments=segments, context=context)
        if cached is not None:
            yield cached
            return

        self.load()

        seg0 = segments[0]
        infer_seed = seg0.infer_seed
        top_p = seg0.top_p
        top_k = seg0.top_k
        temperature = seg0.temperature
        emotion_text = seg0.emotion_prompt

        # NOTE: 这个模型不需要 ref_txt
        sr = self.get_sample_rate()
        ref_wav, ref_txt = self.get_ref_wav(seg0)
        if ref_wav is None:
            # NOTE: 必须要有 reference audio
            raise RuntimeError("Reference audio not found.")

        # 这里用 mkstemp 是为了兼容 windows
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            # NOTE: 这里不在内存中操作是因为，
            # model.infer 逻辑很奇怪，会进行两次 resample ，直接写成文件让它自己转，我们不关心了
            sf.write(tmp_path, ref_wav, sr, format="WAV")
            ret = []
            for segment in segments:
                with SeedContext(infer_seed):
                    wav_sr, wav_data = self.tts.infer(
                        spk_audio_prompt=tmp_path,
                        text=segment.text,
                        verbose=False,
                        output_path=None,
                        emo_text=emotion_text,
                        use_emo_text=True if emotion_text else False,
                        # 这个值控制 emotion 的强度，太强了会导致不像原来的音色，所以设置这个值的来自于 官方仓库中的 emo_weight
                        # ref: https://github.com/index-tts/index-tts/blob/a05850286588600c632675bb128f628dc58b9070/webui.py#L115-L168
                        emo_alpha=0.8 * 0.8,
                        # 下面是 generation_kwargs
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                wav_data: np.ndarray = wav_data
                wav_data = (
                    wav_data.reshape(-1).astype(np.float32) / np.iinfo(np.int16).max
                )
                ret.append((wav_sr, wav_data))
        finally:
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {tmp_path}: {e}")
        if not context.stop:
            self.set_cache(segments=segments, context=context, value=ret)
        yield ret
