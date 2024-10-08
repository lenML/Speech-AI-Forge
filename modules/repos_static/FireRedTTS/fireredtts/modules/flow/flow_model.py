import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from fireredtts.modules.flow.utils import make_pad_mask


class InterpolateRegulator(nn.Module):
    def __init__(
            self,
            channels: int = 512,
            num_blocks: int = 4,
            groups: int = 1,
    ):
        super().__init__()
        model = []
        for _ in range(num_blocks):
            model.extend([
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.GroupNorm(groups, channels),
                nn.Mish(),
            ])
        model.append(
            nn.Conv1d(channels, channels, 1, 1)
        )
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens=None):
        # x in (B, T, D)
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens


class CrossAttnFlowMatching(nn.Module):
    def __init__(self, 
                 output_size: int,
                 input_embedding: nn.Module,
                 encoder: nn.Module,
                 length_regulator: nn.Module,
                 mel_encoder: nn.Module,
                 decoder: nn.Module,
                 ):
        super().__init__()
        self.input_embedding = input_embedding
        self.encoder = encoder
        self.length_regulator = length_regulator
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size, output_size)
        self.prompt_prenet = mel_encoder
        self.decoder = decoder

    def inference(self, 
                  token: torch.Tensor, 
                  token_len: torch.Tensor, 
                  prompt_mel: torch.Tensor, 
                  prompt_mel_len: torch.Tensor, 
                  n_timesteps:int=10,
        ):
        # prompt projection
        prompt_feat = self.prompt_prenet(prompt_mel)
        prompt_feat_len = torch.ceil(prompt_mel_len/self.prompt_prenet.reduction_rate).long()

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(token_len.device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # 40ms shift to 10ms shift
        feat_len = (token_len *4).int()

        # first encoder
        h, _ = self.encoder(token, token_len, prompt_feat, prompt_feat_len)
        # length regulate
        h, _ = self.length_regulator(h, feat_len)
        # final projection
        h = self.encoder_proj(h)
        
        mask = (~make_pad_mask(feat_len)).to(h)

        feat = self.decoder.inference(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            n_timesteps=n_timesteps,
        )
        return feat


