from functools import lru_cache
import sys
import torch
from modules import config

import logging

logger = logging.getLogger(__name__)

if sys.platform == "darwin":
    from modules.devices import mac_devices


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_devices.has_mps


def get_cuda_device_id():
    return (
        int(config.runtime_env_vars.device_id)
        if config.runtime_env_vars.device_id is not None
        and config.runtime_env_vars.device_id.isdigit()
        else 0
    ) or torch.cuda.current_device()


def get_cuda_device_string():
    if config.runtime_env_vars.device_id is not None:
        return f"cuda:{config.runtime_env_vars.device_id}"

    return "cuda"


def get_available_gpus() -> list[tuple[int, int]]:
    """
    Get the list of available GPUs and their free memory.

    :return: A list of tuples where each tuple contains (GPU index, free memory in bytes).
    """
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_memory = props.total_memory - torch.cuda.memory_reserved(i)
        available_gpus.append((i, free_memory))
    return available_gpus


def get_memory_available_gpus(min_memory=2048):
    available_gpus = get_available_gpus()
    memory_available_gpus = [
        gpu for gpu, free_memory in available_gpus if free_memory > min_memory
    ]
    return memory_available_gpus


def get_target_device_id_or_memory_available_gpu():
    memory_available_gpus = get_memory_available_gpus()
    device_id = get_cuda_device_id()
    if device_id not in memory_available_gpus:
        if len(memory_available_gpus) != 0:
            logger.warning(
                f"Device {device_id} is not available or does not have enough memory. will try to use {memory_available_gpus}"
            )
            config.runtime_env_vars.device_id = str(memory_available_gpus[0])
        else:
            logger.warning(
                f"Device {device_id} is not available or does not have enough memory. Using CPU instead."
            )
            return "cpu"
    return get_cuda_device_string()


def get_optimal_device_name():
    if config.runtime_env_vars.use_cpu == "all":
        return "cpu"

    if torch.cuda.is_available():
        return get_target_device_id_or_memory_available_gpu()

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    if task in config.cmd_opts.use_cpu or "all" in config.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()


def torch_gc():
    try:
        if torch.cuda.is_available():
            with torch.cuda.device(get_cuda_device_string()):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        if has_mps():
            mac_devices.torch_mps_gc()
    except Exception as e:
        logger.error(f"Error in torch_gc", exc_info=True)


cpu: torch.device = torch.device("cpu")
device: torch.device = get_optimal_device()
dtype: torch.dtype = torch.float32
dtype_dvae: torch.dtype = torch.float32
dtype_vocos: torch.dtype = torch.float32
dtype_gpt: torch.dtype = torch.float32
dtype_decoder: torch.dtype = torch.float32


def reset_device():
    global device
    global dtype
    global dtype_dvae
    global dtype_vocos
    global dtype_gpt
    global dtype_decoder

    if config.runtime_env_vars.half:
        dtype = torch.float16
        dtype_dvae = torch.float16
        dtype_vocos = torch.float16
        dtype_gpt = torch.float16
        dtype_decoder = torch.float16

        logger.info("Using half precision: torch.float16")
    else:
        dtype = torch.float32
        dtype_dvae = torch.float32
        dtype_vocos = torch.float32
        dtype_gpt = torch.float32
        dtype_decoder = torch.float32

        logger.info("Using full precision: torch.float32")

    if config.runtime_env_vars.use_cpu == "all":
        device = cpu
    else:
        device = get_optimal_device()

    logger.info(f"Using device: {device}")


@lru_cache
def first_time_calculation():
    """
    just do any calculation with pytorch layers - the first time this is done it allocaltes about 700MB of memory and
    spends about 2.7 seconds doing that, at least wih NVidia.
    """

    x = torch.zeros((1, 1)).to(device, dtype)
    linear = torch.nn.Linear(1, 1).to(device, dtype)
    linear(x)

    x = torch.zeros((1, 1, 3, 3)).to(device, dtype)
    conv2d = torch.nn.Conv2d(1, 1, (3, 3)).to(device, dtype)
    conv2d(x)
