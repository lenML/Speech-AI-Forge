# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import math
import torch
import alias_free_cuda
from alias_free_cuda import activation1d
from activations import Snake, SnakeBeta


def test_load_fused_kernels():
    try:
        import alias_free_cuda
        import torch

        print("[Success] load_fused_kernels")
    except ImportError as e:
        print("[Fail] load_fused_kernels")
        raise e


def test_anti_alias_activation():
    data = torch.rand((10, 10, 50000), device="cuda")

    # check activations.Snake cuda vs. torch
    fused_anti_alias_activation = activation1d.Activation1d(
        activation=SnakeBeta(10), fused=True
    ).cuda()
    fused_activation_output = fused_anti_alias_activation(data)

    torch_anti_alias_activation = activation1d.Activation1d(
        activation=SnakeBeta(10), fused=False
    ).cuda()
    torch_activation_output = torch_anti_alias_activation(data)

    test_result = (fused_activation_output - torch_activation_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    if diff <= 1e-3:
        print(
            f"\n[Success] test_fused_anti_alias_activation"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={fused_activation_output[-1][-1][-100:].tolist()}"
            f"\n > torch_values={torch_activation_output[-1][-1][-100:].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_fused_anti_alias_activation"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={fused_activation_output[-1][-1][-30:].tolist()}, "
            f"\n > torch_values={torch_activation_output[-1][-1][-30:].tolist()}"
        )


if __name__ == "__main__":
    from alias_free_cuda import load

    load.load()
    test_load_fused_kernels()
    test_anti_alias_activation()
