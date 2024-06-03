import torch
import random
import numpy as np


def deterministic(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SeedContext:
    def __init__(self, seed):
        assert isinstance(seed, int) or isinstance(
            seed, float
        ), "Seed must be an integer or a float."

        self.seed = seed
        self.state = None
        if isinstance(self.seed, float):
            self.seed = int(self.seed)

        if self.seed == -1:
            self.seed = random.randint(0, 2**32 - 1)

    def __enter__(self):
        self.state = (torch.get_rng_state(), random.getstate(), np.random.get_state())

        deterministic(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_rng_state(self.state[0])
        random.setstate(self.state[1])
        np.random.set_state(self.state[2])
