import argparse
import asyncio
import csv
import logging
import os
import subprocess as sp
import threading
import time
import tracemalloc
from typing import Coroutine

import numpy as np
import torch

from modules import config
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.models.zoo import model_zoo
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from modules.core.pipeline.processor import NP_AUDIO
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.devices import devices

# from modules.generate_audio import generate_audio_batch
# from modules.models import reload_chat_tts, unload_chat_tts
from modules.utils import audio_utils


def wait(coro: Coroutine):
    result = None

    def _run():
        nonlocal result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        loop.close()

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join()
    return result


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disable garbage collection
# NOTE: 这个好像没用了...暂时留着注释
# config.auto_gc = False


def reload_model(model_id="chat-tts"):
    logger.info(f"Reloading model: {model_id}")
    model_zoo.unload_all_models()
    model_zoo.load_model(model_id=model_id)
    logger.info("Reloaded model")


def unload_models():
    logger.info("Unloading models")
    model_zoo.unload_all_models()
    logger.info("Unloaded models")


default_params = {
    "temperature": 1e-12,
    "top_K": 20,
    "top_P": 0.5,
}


def generate_audio_batch(
    texts: list[str], model_id="chat-tts", *args, **kwargs
) -> NP_AUDIO:
    pipe0 = PipelineFactory.create_chattts_pipeline(
        ctx=TTSPipelineContext(
            texts=texts,
            tts_config=TTSConfig(
                mid=model_id,
                temperature=default_params["temperature"],
                top_k=default_params["top_K"],
                top_p=default_params["top_P"],
            ),
            spk=TTSSpeaker.empty(),
            infer_config=InferConfig(eos=" ", sync_gen=False, seed=42),
        ),
    )

    sr, y = wait(pipe0.generate())
    return sr, y


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


def run_tts_benchmark(
    text="八百标兵奔北坡，炮兵并排北边跑。炮兵怕把标兵碰，标兵怕碰炮兵炮",
    batch_size=1,
    use_decoder=False,
    half_precision=False,
    compile_model=False,
    use_cpu=False,
    model_id="chat-tts",
    filename="performance_results.csv",
):
    setup_env(
        half_precision=half_precision, compile_model=compile_model, use_cpu=use_cpu
    )

    texts = [text] * batch_size

    begin_mem_alloc = get_gpu_memory_usage()[0]

    print(f"begin_mem_alloc: {begin_mem_alloc}")

    tracemalloc.start()

    reload_model(model_id)
    # warmup with compile_model=True
    if compile_model:
        logger.info("Warmup for compile_model=True")
        generate_audio_batch(
            texts=texts,
        )

    logger.info("+++++++++++ Benchmark Start +++++++++++")
    start_time = time.time()
    sr, audio_data = generate_audio_batch(
        texts=texts,
    )
    end_time = time.time()
    logger.info("+++++++++++ Benchmark End +++++++++++")

    end_mem_alloc = get_gpu_memory_usage()[0]
    print(f"end_mem_alloc: {end_mem_alloc}")

    gpu_mem = end_mem_alloc - begin_mem_alloc
    gpu_mem_gb = gpu_mem / 1024

    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_mem_gb = peak_mem / 1024 / 1024 / 1024

    unload_models()

    segment = audio_utils.ndarray_to_segment(audio_data, sr)
    duration = segment.duration_seconds

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
    print(f"Audio Duration:  {duration} seconds")
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


def get_args():
    """
    可以选择 model_id 默认为 chat-tts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="chat-tts")
    parser.add_argument(
        "--text",
        type=str,
        default="八百标兵奔北坡，炮兵并排北边跑。炮兵怕把标兵碰，标兵怕碰炮兵炮",
    )

    return parser.parse_args()


def main_benchmark():
    args = get_args()
    model_id = args.model_id
    text = args.text
    filename = f"performance_results_{model_id}.csv"

    print("+++++++++++ Benchmark Running +++++++++++")
    print(f"model_id: {model_id}")
    print(f"text: {text}")
    print(f"filename: {filename}")

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
                                raise Exception(
                                    "Half precision is not supported on CPU"
                                )
                            run_tts_benchmark(
                                text=text,
                                batch_size=batch_size,
                                use_decoder=use_decoder,
                                half_precision=half_precision,
                                compile_model=compile_model,
                                use_cpu=use_cpu,
                                model_id=model_id,
                                filename=filename,
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


if __name__ == "__main__":
    main_benchmark()
