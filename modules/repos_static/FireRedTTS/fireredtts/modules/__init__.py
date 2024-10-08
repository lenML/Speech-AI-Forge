import json
import torch
import torch.nn as nn
from fireredtts.modules.bigvgan import get_bigvgan_backend
from fireredtts.modules.flow import get_flow_frontend, MelSpectrogramExtractor


class Token2Wav(nn.Module):
    def __init__(
        self,
        flow: nn.Module,
        generator: nn.Module,
    ):
        super().__init__()
        self.flow = flow
        self.generator = generator

    @torch.no_grad()
    def inference(
        self, tokens: torch.Tensor, prompt_mel: torch.Tensor, n_timesteps: int = 10
    ) -> torch.Tensor:
        token_len = torch.tensor([tokens.shape[1]], dtype=torch.long).to(tokens.device)
        prompt_mel_len = torch.tensor([prompt_mel.shape[1]], dtype=torch.long).to(
            prompt_mel.device
        )
        # flow
        mel = self.flow.inference(
            token=tokens,
            token_len=token_len,
            prompt_mel=prompt_mel,
            prompt_mel_len=prompt_mel_len,
            n_timesteps=n_timesteps,
        )
        # bigvgan
        audio = self.generator(mel)  # (b=1, 1, t)
        return audio.squeeze(1)

    @classmethod
    def init_from_config(cls, config) -> "Token2Wav":
        flow = get_flow_frontend(config["flow"])
        bigvgan = get_bigvgan_backend(config["bigvgan"])
        return cls(flow, bigvgan)
