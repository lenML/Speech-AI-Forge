import os
from typing import List, Literal
from modules.devices import devices
from modules.repos_static.resemble_enhance.enhancer.enhancer import Enhancer
from modules.repos_static.resemble_enhance.enhancer.hparams import HParams
from modules.repos_static.resemble_enhance.inference import inference

import torch

from modules.utils.constants import MODELS_DIR
from pathlib import Path

from threading import Lock

resemble_enhance = None
lock = Lock()


def load_enhancer(device: torch.device):
    global resemble_enhance
    with lock:
        if resemble_enhance is None:
            resemble_enhance = ResembleEnhance(device)
            resemble_enhance.load_model()
    return resemble_enhance


class ResembleEnhance:
    def __init__(self, device: torch.device):
        self.device = device

        self.enhancer: HParams = None
        self.hparams: Enhancer = None

    def load_model(self):
        hparams = HParams.load(Path(MODELS_DIR) / "resemble-enhance")
        enhancer = Enhancer(hparams)
        state_dict = torch.load(
            Path(MODELS_DIR) / "resemble-enhance" / "mp_rank_00_model_states.pt",
            map_location="cpu",
        )["module"]
        enhancer.load_state_dict(state_dict)
        enhancer.to(self.device).eval()

        self.hparams = hparams
        self.enhancer = enhancer

    @torch.inference_mode()
    def denoise(self, dwav, sr, device) -> tuple[torch.Tensor, int]:
        assert self.enhancer is not None, "Model not loaded"
        assert self.enhancer.denoiser is not None, "Denoiser not loaded"
        enhancer = self.enhancer
        return inference(model=enhancer.denoiser, dwav=dwav, sr=sr, device=device)

    @torch.inference_mode()
    def enhance(
        self,
        dwav,
        sr,
        device,
        nfe=32,
        solver: Literal["midpoint", "rk4", "euler"] = "midpoint",
        lambd=0.5,
        tau=0.5,
    ) -> tuple[torch.Tensor, int]:
        assert 0 < nfe <= 128, f"nfe must be in (0, 128], got {nfe}"
        assert solver in (
            "midpoint",
            "rk4",
            "euler",
        ), f"solver must be in ('midpoint', 'rk4', 'euler'), got {solver}"
        assert 0 <= lambd <= 1, f"lambd must be in [0, 1], got {lambd}"
        assert 0 <= tau <= 1, f"tau must be in [0, 1], got {tau}"
        assert self.enhancer is not None, "Model not loaded"
        enhancer = self.enhancer
        enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
        return inference(model=enhancer, dwav=dwav, sr=sr, device=device)


if __name__ == "__main__":
    import torchaudio
    import gradio as gr

    device = torch.device("cuda")

    # def enhance(file):
    #     print(file)
    #     ench = load_enhancer(device)
    #     dwav, sr = torchaudio.load(file)
    #     dwav = dwav.mean(dim=0).to(device)
    #     enhanced, e_sr = ench.enhance(dwav, sr)
    #     return e_sr, enhanced.cpu().numpy()

    # # 随便一个示例
    # gr.Interface(
    #     fn=enhance, inputs=[gr.Audio(type="filepath")], outputs=[gr.Audio()]
    # ).launch()

    # load_chat_tts()

    # ench = load_enhancer(device)

    # devices.torch_gc()

    # wav, sr = torchaudio.load("test.wav")

    # print(wav.shape, type(wav), sr, type(sr))
    # # exit()

    # wav = wav.squeeze(0).cuda()

    # print(wav.device)

    # denoised, d_sr = ench.denoise(wav, sr)
    # denoised = denoised.unsqueeze(0)
    # print(denoised.shape)
    # torchaudio.save("denoised.wav", denoised.cpu(), d_sr)

    # for solver in ("midpoint", "rk4", "euler"):
    #     for lambd in (0.1, 0.5, 0.9):
    #         for tau in (0.1, 0.5, 0.9):
    #             enhanced, e_sr = ench.enhance(
    #                 wav, sr, solver=solver, lambd=lambd, tau=tau, nfe=128
    #             )
    #             enhanced = enhanced.unsqueeze(0)
    #             print(enhanced.shape)
    #             torchaudio.save(
    #                 f"enhanced_{solver}_{lambd}_{tau}.wav", enhanced.cpu(), e_sr
    #             )
