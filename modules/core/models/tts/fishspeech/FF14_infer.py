import logging
from typing import Optional

import numpy as np
import torch

from modules.core.models.tts.fishspeech.FF14_llama import FF14_llama
from modules.core.models.tts.fishspeech.FF14_vqgan import FF14_vqgan
from modules.devices import devices

logger = logging.getLogger(__name__)


class FF14_infer:
    """
    fishspeech inference
    """

    def __init__(self) -> None:
        self.llama = FF14_llama()
        self.vqgan = FF14_vqgan()

    def unload(self) -> None:
        self.llama.unload()
        self.vqgan.unload()

    @torch.no_grad()
    @torch.inference_mode()
    def generate(
        self,
        text: str,
        ref_text: Optional[str] = None,
        ref_wav: Optional[np.ndarray] = None,
        max_new_tokens: int = 0,
        top_p: int = 0.7,
        repetition_penalty: float = 1.5,
        temperature: float = 0.7,
        chunk_length: int = 150,
    ):
        if ref_text and ref_wav is None:
            raise ValueError("ref_wav must be provided if ref_text is provided")

        indices = self.vqgan.encode(ref_wav) if ref_wav is not None else None
        codes = self.llama.generate(
            text=text,
            prompt_text=ref_text,
            prompt_tokens=indices,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            chunk_length=chunk_length,
        )
        generated_wav = self.vqgan.decode(codes)
        return generated_wav
