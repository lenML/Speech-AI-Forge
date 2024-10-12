import threading
from typing import Literal, Optional

import numpy as np
import torch

from modules.core.models.BaseZooModel import BaseZooModel
from modules.core.models.enhancer.ResembleEnhance import (
    ResembleEnhance,
    load_enhancer,
    unload_enhancer,
)


class ResembleEnhanceModel(BaseZooModel):
    model: Optional[ResembleEnhance] = None

    lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__("resemble-enhance")

    def is_loaded(self) -> bool:
        return ResembleEnhanceModel.model is not None

    def load(self):
        with self.lock:
            if ResembleEnhanceModel.model is None:
                ResembleEnhanceModel.model = load_enhancer()
            return ResembleEnhanceModel.model

    def unload(self):
        with self.lock:
            ResembleEnhanceModel.model = None
            unload_enhancer()

    def apply_audio_enhance_full(
        self,
        audio_data: np.ndarray,
        sr: int,
        nfe=32,
        solver: Literal["midpoint", "rk4", "euler"] = "midpoint",
        lambd=0.5,
        tau=0.5,
    ):
        model = self.load()

        # FIXME: 这里可能改成 to(device) 会优化一点？
        tensor = torch.from_numpy(audio_data).float().squeeze().cpu()

        tensor, sr = model.enhance(
            tensor, sr, tau=tau, nfe=nfe, solver=solver, lambd=lambd
        )

        audio_data = tensor.cpu().numpy()
        return audio_data, int(sr)

    def apply_audio_enhance(
        self,
        audio_data: np.ndarray,
        sr: int,
        enable_denoise: bool,
        enable_enhance: bool,
    ):
        model = self.load()

        if not enable_denoise and not enable_enhance:
            return audio_data, sr

        # FIXME: 这里可能改成 to(device) 会优化一点？
        tensor = torch.from_numpy(audio_data).float().squeeze().cpu()

        if enable_enhance or enable_denoise:
            lambd = 0.9 if enable_denoise else 0.1
            tensor, sr = model.enhance(
                tensor, sr, tau=0.5, nfe=64, solver="rk4", lambd=lambd
            )

        audio_data = tensor.cpu().numpy()
        return audio_data, int(sr)
