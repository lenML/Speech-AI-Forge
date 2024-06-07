import subprocess as sp
import os
import time
import numpy as np
import torch
from modules.generate_audio import generate_audio_batch
from modules.models import reload_chat_tts, unload_chat_tts
from modules import config
from modules.devices import devices
from modules.utils import audio
import csv

import logging
import tracemalloc

logger = logging.getLogger(__name__)

filename = "performance_results.csv"

# Disable garbage collection
config.auto_gc = False


def get_gpu_memory_usage():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_used_values = [int(x.split()[0]) for x in memory_used_info]
    return memory_used_values


def setup_env(
    half_precision: bool = False,
    compile_model: bool = False,
    use_cpu: bool = False,
):
    if half_precision:
        config.runtime_env_vars.half = True
    else:
        config.runtime_env_vars.half = False

    if compile_model:
        config.runtime_env_vars.compile = True
    else:
        config.runtime_env_vars.compile = False

    if use_cpu:
        config.runtime_env_vars.use_cpu = "all"
    else:
        del config.runtime_env_vars.use_cpu

    devices.reset_device()
    devices.first_time_calculation()


default_params = {
    "temperature": 1e-12,
    "top_K": 20,
    "top_P": 0.5,
    "spk": 42,
    "infer_seed": 42,
}


def test_tts(
    text="八百标兵奔北坡，炮兵并排北边跑。炮兵怕把标兵碰，标兵怕碰炮兵炮",
    batch_size=1,
    use_decoder=False,
    half_precision=False,
    compile_model=False,
    use_cpu=False,
):
    setup_env(
        half_precision=half_precision, compile_model=compile_model, use_cpu=use_cpu
    )

    texts = [text] * batch_size

    begin_mem_alloc = get_gpu_memory_usage()[0]

    print(f"begin_mem_alloc: {begin_mem_alloc}")

    tracemalloc.start()

    reload_chat_tts()
    # warmup
    logger.info("Warmup")
    generate_audio_batch(
        texts=texts,
        use_decoder=use_decoder,
        **default_params,
    )

    logger.info("+++++++++++ Benchmark Start +++++++++++")
    start_time = time.time()
    (sr, audio_data) = generate_audio_batch(
        texts=texts,
        use_decoder=use_decoder,
        **default_params,
    )[0]
    end_time = time.time()
    logger.info("+++++++++++ Benchmark End +++++++++++")

    end_mem_alloc = get_gpu_memory_usage()[0]
    print(f"end_mem_alloc: {end_mem_alloc}")

    gpu_mem = end_mem_alloc - begin_mem_alloc
    gpu_mem_gb = gpu_mem / 1024

    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_mem_gb = peak_mem / 1024 / 1024 / 1024

    unload_chat_tts()

    segment = audio.ndarray_to_segment(audio_data, sr)
    duration = segment.duration_seconds * batch_size

    pt = end_time - start_time
    rtf = pt / duration

    print(f"Batch size:      {batch_size}")
    print(f"Text:            {text}")
    print(f"use_decoder:     {use_decoder}")
    print(f"Half precision:  {half_precision}")
    print(f"Compile model:   {compile_model}")
    print(f"Use CPU:         {use_cpu}")
    print(f"GPU MEM:         {gpu_mem_gb:.2f} GB")
    print(f"CPU MEM:         {cpu_mem_gb:.2f} GB")
    print(f"Duration:        {duration} seconds")
    print(f"Processing time: {pt:.2f} seconds")
    print(f"RTF:             {rtf:.2f}")

    # 将结果写入 CSV 文件
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                batch_size,
                use_decoder,
                half_precision,
                compile_model,
                use_cpu,
                f"{gpu_mem_gb:.2f}" if not use_cpu else "N/A",
                f"{duration:.2f}",
                f"{rtf:.2f}",
            ]
        )


if not os.path.exists(filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Batch size",
                "Use decoder",
                "Half precision",
                "Compile model",
                "Use CPU",
                "GPU Memory",
                "Duration",
                "RTF",
            ]
        )


for batch_size in range(2, 9, 2):
    for use_decoder in [False, True]:
        for half_precision in [False, True]:
            for compile_model in [False, True]:
                for use_cpu in [False, True]:
                    try:
                        if use_cpu and half_precision:
                            raise Exception("Half precision is not supported on CPU")
                        test_tts(
                            batch_size=batch_size,
                            use_decoder=use_decoder,
                            half_precision=half_precision,
                            compile_model=compile_model,
                            use_cpu=use_cpu,
                        )
                    except Exception as e:
                        logger.error(e, exc_info=True)
                        with open(filename, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(
                                [
                                    batch_size,
                                    use_decoder,
                                    half_precision,
                                    compile_model,
                                    use_cpu,
                                    "N/A",
                                    "N/A",
                                    "N/A",
                                ]
                            )
