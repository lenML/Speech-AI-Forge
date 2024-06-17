import logging
import random
from typing import Any, Dict, List

import numpy as np
from lxml import etree

from modules.data import styles_mgr
from modules.speaker import speaker_mgr

logger = logging.getLogger(__name__)


def expand_spk(attrs: dict):
    input_spk = attrs.get("spk", "")
    if isinstance(input_spk, int):
        return
    if isinstance(input_spk, str) and input_spk.isdigit():
        attrs.update({"spk": int(input_spk)})
        return
    try:
        speaker = speaker_mgr.get_speaker(input_spk)
        attrs.update({"spk": speaker})
    except Exception as e:
        logger.error(f"apply style failed, {e}")


def expand_style(attrs: dict):
    if attrs.get("style", "") != "":
        try:
            params = styles_mgr.find_params_by_name(str(attrs["style"]))
            attrs.update(params)
        except Exception as e:
            logger.error(f"apply style failed, {e}")


def merge_prompt(attrs: dict, elem):

    def attr_num(attrs: Dict[str, Any], k: str, min_value: int, max_value: int):
        val = elem.get(k, attrs.get(k, ""))
        if val == "":
            return
        if val == "max":
            val = max_value
        if val == "min":
            val = min_value
        val = np.clip(int(val), min_value, max_value)
        if "prefix" not in attrs or attrs["prefix"] == None:
            attrs["prefix"] = ""
        attrs["prefix"] += " " + f"[{k}_{val}]"

    attr_num(attrs, "oral", 0, 9)
    attr_num(attrs, "speed", 0, 9)
    attr_num(attrs, "laugh", 0, 2)
    attr_num(attrs, "break", 0, 7)


def apply_random_seed(attrs: dict):
    seed = attrs.get("seed", "")
    if seed == "random" or seed == "rand":
        seed = random.randint(0, 2**32 - 1)
        attrs["seed"] = seed
        logger.info(f"random seed: {seed}")
