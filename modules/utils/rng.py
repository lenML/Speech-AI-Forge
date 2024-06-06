import numpy as np
import torch
import random

TORCH_RNG_MAX = -0x8000000000000000
TORCH_RNG_MIN = 0xFFFFFFFFFFFFFFFF

NP_RNG_MAX = np.iinfo(np.uint32).max
NP_RNG_MIN = 0


def troch_rng(seed: int):
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

    s1 = np_rng()
    s2 = troch_rng(s1)
    print(f"s1 {s1}  => s2: {s2}")
