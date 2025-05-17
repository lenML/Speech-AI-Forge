import logging
import time
from pathlib import Path
from typing import Optional, Union

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from modules import config
from modules.devices import devices
from modules.repos_static.fish_speech.fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)
from modules.repos_static.fish_speech.fish_speech.models.vqgan.modules.firefly import (
    FireflyArchitecture,
)
from modules.repos_static.fish_speech.tools.llama.generate import (
    decode_one_token_ar,
    decode_one_token_naive,
    generate_long,
)

logger = logging.getLogger(__name__)


class FF14_llama:
    """
    封装 fishspeech llama
    """

    MODEL_PATH = Path("./models/fish-speech-1_4")

    def __init__(self) -> None:

        self.model: Union[NaiveTransformer, DualARTransformer] = None
        self.decode_one_token: callable = None

        self.config_name = "firefly_gan_vq"
        self.checkpoint_filename = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        self.device = devices.get_device_for("fish-speech")

        self.repo_path = Path("./modules/repos_static/fish_speech")

        model, decode_one_token = self.load_model()
        self.model = model
        self.decode_one_token = decode_one_token

    def unload(self):
        if self.model is None:
            return
        self.model.to("cpu")
        del self.model

    def load_model(self):
        checkpoint_path = str(self.MODEL_PATH)
        device = self.device
        precision = devices.dtype

        model: Union[NaiveTransformer, DualARTransformer] = (
            BaseTransformer.from_pretrained(checkpoint_path, load_weights=True)
        )

        model = model.to(device=device, dtype=precision)
        logger.info(f"Restored model from checkpoint")

        if isinstance(model, DualARTransformer):
            decode_one_token = decode_one_token_ar
            logger.info("Using DualARTransformer")
        else:
            decode_one_token = decode_one_token_naive
            logger.info("Using NaiveTransformer")

        if config.runtime_env_vars.compile:
            logger.info("Compiling function...")
            decode_one_token = torch.compile(
                decode_one_token,
                fullgraph=True,
                backend="inductor" if torch.cuda.is_available() else "aot_eager",
                mode="reduce-overhead" if torch.cuda.is_available() else None,
            )

        logger.info("Loading model ...")
        t0 = time.time()
        model = model.eval()

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

        return model, decode_one_token

    def generate(
        self,
        *,
        text: str,
        prompt_text: Optional[Union[str, list[str]]] = None,
        prompt_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        max_new_tokens: int = 0,
        top_p: int = 0.7,
        repetition_penalty: float = 1.5,
        temperature: float = 0.7,
        chunk_length: int = 150,
    ) -> np.ndarray:
        if (
            isinstance(prompt_text, list)
            and isinstance(prompt_tokens, list)
            and len(prompt_text) != len(prompt_tokens)
        ):
            raise ValueError(
                f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
            )

        model = self.model
        decode_one_token = self.decode_one_token
        device = self.device
        compile = config.runtime_env_vars.compile

        generator = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=compile,
            iterative_prompt=False,
            chunk_length=chunk_length,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
        )

        codes = []

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
            elif response.action == "next":
                break
            else:
                logger.error(f"Error: {response}")

        generated_codes = torch.cat(codes, dim=1).cpu().numpy()

        return generated_codes
