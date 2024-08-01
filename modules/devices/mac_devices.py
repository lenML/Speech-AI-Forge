import logging

import torch
import torch.backends
import torch.backends.mps
from packaging import version

logger = logging.getLogger(__name__)


def check_for_mps() -> bool:
    if version.parse(torch.__version__) <= version.parse("2.0.1"):
        if not getattr(torch, "has_mps", False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False
    else:
        try:
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except Exception as e:
            logger.warning("MPS check failed: %s", exc_info=True)
            return False


has_mps = check_for_mps()


def torch_mps_gc() -> None:
    try:
        from torch.mps import empty_cache

        empty_cache()
    except Exception:
        logger.warning("MPS garbage collection failed", exc_info=True)


if __name__ == "__main__":
    print(torch.__version__)
    print(has_mps)
    torch_mps_gc()
