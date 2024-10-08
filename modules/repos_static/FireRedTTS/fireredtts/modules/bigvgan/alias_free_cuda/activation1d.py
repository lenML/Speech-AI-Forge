# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import torch
import torch.nn as nn
from token2wav.alias_free_torch.resample import UpSample1d, DownSample1d

# load fused CUDA kernel: this enables importing anti_alias_activation_cuda
from token2wav.alias_free_cuda import load

load.load()


class FusedAntiAliasActivation(torch.autograd.Function):
    """
    Assumes filter size 12, replication padding on upsampling, and logscale alpha/beta parameters as inputs
    """

    @staticmethod
    def forward(ctx, inputs, ftr, alpha, beta):
        import anti_alias_activation_cuda

        activation_results = anti_alias_activation_cuda.forward(
            inputs, ftr, alpha, beta
        )
        return activation_results

    @staticmethod
    def backward(ctx, output_grads):
        # TODO: implement bwd pass
        raise NotImplementedError
        return output_grads, None, None


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        fused: bool = True,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

        self.fused = fused  # whether to use fused CUDA kernel or not

    def forward(self, x):
        if not self.fused:
            x = self.upsample(x)
            x = self.act(x)
            x = self.downsample(x)
            return x
        else:
            if self.act.__class__.__name__ == "Snake":
                beta = self.act.alpha.data  # snake uses same params for alpha and beta
            else:
                beta = (
                    self.act.beta.data
                )  # snakebeta uses different params for alpha and beta
            alpha = self.act.alpha.data
            if (
                not self.act.alpha_logscale
            ):  # exp baked into cuda kernel, cancel it out with a log
                alpha = torch.log(alpha)
                beta = torch.log(beta)
            x = FusedAntiAliasActivation.apply(x, self.upsample.filter, alpha, beta)
            x = self.downsample(x)
            return x
