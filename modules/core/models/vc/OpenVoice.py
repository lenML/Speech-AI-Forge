if __name__ == "__main__":
    from modules.repos_static.sys_paths import setup_repos_paths

    setup_repos_paths()

import threading
from pathlib import Path
from typing import Optional

import numpy.typing as npt
import torch

from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.models.AudioReshaper import AudioReshaper
from modules.core.models.vc.VCModel import VCModel
from modules.core.pipeline.processor import NP_AUDIO
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.devices import devices
from modules.repos_static.openvoice.openvoice.api import ToneColorConverter
from modules.repos_static.openvoice.openvoice.mel_processing import spectrogram_torch


class OpenVoiceModel(VCModel):
    model: Optional[ToneColorConverter] = None

    lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__("open-voice")

        self.model_dir = Path("./models/OpenVoiceV2/converter")

    @devices.after_gc()
    def load(self) -> None:
        with self.lock:
            if OpenVoiceModel.model is None:
                # FIXME: 用上 dtype 配置
                model = ToneColorConverter(
                    config_path=self.model_dir / "config.json",
                    device=str(self.get_device()),
                    enable_watermark=False,
                )
                model.load_ckpt(ckpt_path=self.model_dir / "checkpoint.pth")
                OpenVoiceModel.model = model
        return OpenVoiceModel.model

    @devices.after_gc()
    def unload(self) -> None:
        with self.lock:
            if OpenVoiceModel.model is not None:
                del OpenVoiceModel.model
                OpenVoiceModel.model = None

    @property
    def sampling_rate(self) -> int:
        # hps = self.model.hps
        # return hps.data.sampling_rate

        # 因为这里有可能在 load 之前调用，所以直接写死了
        # 22050 来自 models/OpenVoiceV2/converter/config.json
        return 22050

    def get_sample_rate(self):
        return self.sampling_rate

    def audio_to_se(self, audio: NP_AUDIO) -> torch.Tensor:
        hps = self.model.hps
        device = self.get_device()
        model = self.model.model
        target_sr = hps.data.sampling_rate

        sr, audio_ref = AudioReshaper.normalize_audio(audio=audio, target_sr=target_sr)

        y = torch.FloatTensor(audio_ref)
        y = y.to(device)
        y = y.unsqueeze(0)
        y = spectrogram_torch(
            y,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        ).to(device)

        with torch.no_grad():
            g = model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
            return g

    @torch.inference_mode()
    def _convert(
        self, audio: npt.NDArray, src_se: torch.Tensor, tgt_se: torch.Tensor, tau: float
    ) -> npt.NDArray:
        hps = self.model.hps
        device = self.get_device()
        model = self.model.model

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(device)
            audio = (
                model.voice_conversion(
                    spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
            return audio

    def convert_audio(
        self, src_audio: NP_AUDIO, ref_audio: NP_AUDIO, tau=0.3
    ) -> NP_AUDIO:
        self.load()

        tau = tau or 0.3

        src_se = self.audio_to_se(src_audio)
        # TODO 支持多ref mean
        ref_se = self.audio_to_se(ref_audio)

        sr, audio = AudioReshaper.normalize_audio(
            audio=src_audio, target_sr=self.sampling_rate
        )
        output: npt.NDArray = self._convert(
            audio=audio, src_se=src_se, tgt_se=ref_se, tau=tau
        )

        return sr, output

    def get_ref_audio(self, ref_spk: TTSSpeaker, config: VCConfig) -> NP_AUDIO:
        spk = ref_spk
        emotion = config.emotion

        if not isinstance(spk, TTSSpeaker):
            raise ValueError("spk must be a TTSSpeaker")

        sr, wav, text = spk.get_ref_wav(
            lambda spk_ref: True if emotion is None else spk_ref.emotion == emotion
        )
        if wav is None:
            raise ValueError("this speaker has no reference audio")

        return sr, wav

    def convert(
        self, src_audio: NP_AUDIO, ref_spk: TTSSpeaker, config: VCConfig
    ) -> NP_AUDIO:
        if config.enabled is False:
            return src_audio

        ref_audio = self.get_ref_audio(ref_spk, config)

        return self.convert_audio(src_audio, ref_audio, tau=config.tau)


if __name__ == "__main__":
    import soundfile as sf
    from scipy.io import wavfile

    model = OpenVoiceModel()
    model.load()

    src_audio_path = "./tests/test_inputs/cosyvoice_out1.wav"
    ref_audio_path = "./tests/test_inputs/chattts_out1.wav"

    src_audio, sr1 = sf.read(src_audio_path, dtype="int16")
    ref_audio, sr2 = sf.read(ref_audio_path, dtype="int16")
    ref_spk = TTSSpeaker.from_ref_wav(ref_wav=(sr2, ref_audio))

    sr, output = model.convert(
        src_audio=(sr1, src_audio), ref_spk=ref_spk, config=VCConfig()
    )

    wavfile.write("./openvoice_output.wav", sr, output)
