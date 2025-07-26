import io
import os
import random
import sys
from importlib.resources import files
from pathlib import Path
from typing import Generator, Literal, Tuple

import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch
import torchaudio
import tqdm
from hydra.utils import get_class
from omegaconf import OmegaConf

from modules.repos_static.F5TTS.f5_tts.model.utils import seed_everything

from .f5_infer import (
    infer_batch_process,
    infer_process,
    load_model,
    load_vocoder,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


# NOTE: 目前不支持 bigvgan 因为需要引入外部库，并且似乎没什么特别区别
class F5TTS:

    def __init__(
        self,
        ckpt_file: Path,
        model: Literal["F5TTS_v1_Base", "F5TTS_Base"] = "F5TTS_v1_Base",
        ode_method="euler",
        use_ema=True,
        vocoder_local_path: str = None,
        device=None,
        dtype=None,
    ):
        cfg_file = (
            Path(os.getcwd())
            / "modules"
            / "repos_static"
            / "F5TTS"
            / "f5_tts"
            / "configs"
            / f"{model}.yaml"
        )
        vocab_file = (
            Path(os.getcwd())
            / "modules"
            / "repos_static"
            / "F5TTS"
            / "f5_tts"
            / "data"
            / "Emilia_ZH_EN_pinyin"
            / f"vocab.txt"
        )

        # Path to str path, 因为f5的函数不支持Path
        ckpt_file = str(ckpt_file)
        vocab_file = str(vocab_file)

        model_cfg = OmegaConf.load(cfg_file)

        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

        self.ode_method = ode_method
        self.use_ema = use_ema

        self.dtype = dtype

        if device is not None:
            self.device = device
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "xpu"
                    if torch.xpu.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )

        if isinstance(self.device, torch.device):
            self.device = str(self.device)

        if self.mel_spec_type != "vocos":
            raise NotImplementedError("Only vocos is supported now")

        # Load models
        self.vocoder = load_vocoder(
            vocoder_name=self.mel_spec_type,
            is_local=True,
            local_path=vocoder_local_path,
            device=self.device,
            # NOTE: 不支持在这里控制 hf 下载，没有就是没有，下载是其他模块的事
            hf_cache_dir=None,
        )

        self.ema_model = load_model(
            model_cls,
            model_arc,
            ckpt_file,
            self.mel_spec_type,
            vocab_file,
            self.ode_method,
            self.use_ema,
            self.device,
            self.dtype,
        )

    def unload(self):
        del self.ema_model
        del self.vocoder

    # NOTE: 对于我们的系统，不需要这个，他里面是用 whisper pipeline
    # def transcribe(self, ref_audio, language=None):
    #     return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

    def apply_seed(self, seed: int):
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

    def infer(
        self,
        ref_file: io.BytesIO | str | os.PathLike,
        ref_text: str,
        gen_text: str,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spec=None,
        seed=None,
    ):
        self.apply_seed(seed)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spec is not None:
            self.export_spectrogram(spec, file_spec)

        return wav, sr, spec

    def infer_batch(
        self,
        ref_file: io.BytesIO | str | os.PathLike,
        ref_text: str,
        gen_text_batches: list[str],
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        # NOTE: 这里可能有问题，因为有某种情况会返回 [None, int, None] ，暂时不知道具体什么情况会输出空
    ) -> list[Tuple[npt.NDArray, int, npt.NDArray]]:
        audio, sr = torchaudio.load(ref_file)
        audios: list[Tuple[npt.NDArray, int, npt.NDArray]] = list(
            infer_batch_process(
                (audio, sr),
                ref_text,
                gen_text_batches,
                model_obj=self.ema_model,
                vocoder=self.vocoder,
                mel_spec_type=self.mel_spec_type,
                progress=tqdm,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                fix_duration=fix_duration,
                device=self.device,
            )
        )
        return audios

    def infer_batch_stream(
        self,
        ref_file: io.BytesIO | str | os.PathLike,
        ref_text: str,
        gen_text_batches: list[str],
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        # NOTE: 这里可能有问题，因为有某种情况会返回 [None, int] 的 generator ，暂时不知道具体什么情况会输出空
    ) -> Generator[Tuple[npt.NDArray, int], None, None]:
        audio, sr = torchaudio.load(ref_file)
        return infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj=self.ema_model,
            vocoder=self.vocoder,
            mel_spec_type=self.mel_spec_type,
            progress=tqdm,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
            streaming=True,
        )


if __name__ == "__main__":
    """
    NOTE: 下面这个跑不了因为ref_file我们没copy过来，但是保留一下当作代码示例
    """

    f5tts = F5TTS()

    wav, sr, spec = f5tts.infer(
        ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        ref_text="some call me nature, others call me mother nature.",
        gen_text="""I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.""",
        file_wave=str(files("f5_tts").joinpath("../../tests/api_out.wav")),
        file_spec=str(files("f5_tts").joinpath("../../tests/api_out.png")),
        seed=None,
    )

    print("seed :", f5tts.seed)
