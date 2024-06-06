import numpy as np
import torch

TORCH_RNG_MAX = 0xFFFF_FFFF_FFFF_FFFF
TORCH_RNG_MIN = -0x8000_0000_0000_0000

NP_RNG_MAX = np.iinfo(np.uint32).max
NP_RNG_MIN = 0


def torch_rng(seed: int):
    torch.manual_seed(seed)
    random_float = torch.empty(1).uniform_().item()
    torch_rn = int(random_float * (TORCH_RNG_MAX - TORCH_RNG_MIN) + TORCH_RNG_MIN)
    np_rn = int(random_float * (NP_RNG_MAX - NP_RNG_MIN) + NP_RNG_MIN)
    return torch_rn, np_rn


def convert_np_to_torch(np_rn: int):
    random_float = (np_rn - NP_RNG_MIN) / (NP_RNG_MAX - NP_RNG_MIN)
    torch_rn = int(random_float * (TORCH_RNG_MAX - TORCH_RNG_MIN) + TORCH_RNG_MIN)
    return torch_rn


def np_rng():
    return int(np.random.randint(NP_RNG_MIN, NP_RNG_MAX, dtype=np.uint32))


if __name__ == "__main__":
    import random

    print(TORCH_RNG_MIN, TORCH_RNG_MAX)
    s1 = np_rng()
    s2 = torch_rng(s1)
    print(f"s1 {s1}  => s2: {s2}")
