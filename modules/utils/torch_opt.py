import torch


def configure_torch_optimizations():
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision("high")
