import numpy as np
import torch
import torch.nn as nn


class HHGCodecEmbedding(nn.Module):
    def __init__(self, out_channels, codebook_path: str, freeze=True):
        super().__init__()
        # (2, 128, 128)
        codebook = torch.from_numpy(np.load(codebook_path).copy())
        assert codebook.shape[0] == 2 and codebook.shape[1] == 128
        self.codebook_dim = codebook.shape[2]

        self.codebook = torch.nn.ModuleList(
            [
                torch.nn.Embedding.from_pretrained(codebook[i], freeze=freeze)
                for i in range(codebook.shape[0])
            ]
        )
        if self.codebook_dim * 2 != out_channels:
            self.proj = nn.Linear(self.codebook_dim * 2, out_channels)
        else:
            self.proj = nn.Identity()

    def forward(self, tokens):
        token_embs = torch.cat(
            [self.codebook[0](tokens % 128), self.codebook[1](tokens // 128)], dim=-1
        )
        token_embs = self.proj(token_embs)
        return token_embs
