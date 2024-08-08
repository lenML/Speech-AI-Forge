import gc
import logging
from pathlib import Path
from threading import Lock
from typing import Literal

import torch

from modules import config
from modules.devices import devices
from modules.repos_static.resemble_enhance.enhancer.enhancer import Enhancer
from modules.repos_static.resemble_enhance.enhancer.hparams import HParams
from modules.repos_static.resemble_enhance.inference import inference
from modules.utils.constants import MODELS_DIR
from modules.utils.monkey_tqdm import disable_tqdm

logger = logging.getLogger(__name__)

resemble_enhance = None
lock = Lock()


class ResembleEnhance:
    def __init__(self, device: torch.device, dtype=torch.float32):
        self.device = device
        self.dtype = dtype

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
        enhancer.to(device=self.device, dtype=self.dtype).eval()

        self.hparams = hparams
        self.enhancer = enhancer

    @torch.inference_mode()
    def denoise(self, dwav, sr) -> tuple[torch.Tensor, int]:
        assert self.enhancer is not None, "Model not loaded"
        assert self.enhancer.denoiser is not None, "Denoiser not loaded"
        enhancer = self.enhancer

        with disable_tqdm(enabled=config.runtime_env_vars.off_tqdm):
            return inference(
                model=enhancer.denoiser,
                dwav=dwav,
                sr=sr,
                device=self.devicem,
                dtype=self.dtype,
            )

    @torch.inference_mode()
    def enhance(
        self,
        dwav,
        sr,
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

        with disable_tqdm(enabled=config.runtime_env_vars.off_tqdm):
            return inference(
                model=enhancer, dwav=dwav, sr=sr, device=self.device, dtype=self.dtype
            )


def load_enhancer() -> ResembleEnhance:
    global resemble_enhance
    with lock:
        if resemble_enhance is None:
            logger.info("Loading ResembleEnhance model")
            resemble_enhance = ResembleEnhance(
                device=devices.get_device_for("enhancer"), dtype=devices.dtype
            )
            resemble_enhance.load_model()
            logger.info("ResembleEnhance model loaded")
    return resemble_enhance


def unload_enhancer():
    global resemble_enhance
    with lock:
        if resemble_enhance is not None:
            logger.info("Unloading ResembleEnhance model")
            del resemble_enhance
            resemble_enhance = None
            devices.torch_gc()
            gc.collect()
            logger.info("ResembleEnhance model unloaded")


def reload_enhancer():
    logger.info("Reloading ResembleEnhance model")
    unload_enhancer()
    load_enhancer()
    logger.info("ResembleEnhance model reloaded")


if __name__ == "__main__":
    import gradio as gr
    import torchaudio

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
