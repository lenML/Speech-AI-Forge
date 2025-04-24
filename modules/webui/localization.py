import json
import logging
import os

import gradio as gr

logger = logging.getLogger(__name__)

current_translation = {}
localization_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "language"
)


def localization_js(filename):
    global current_translation

    if isinstance(filename, str):
        if filename.find(":")>0:
            filename = filename.split(":")
            full_name = os.path.abspath(os.path.join(localization_root, filename[0] + ".json"))
        else:
            full_name = os.path.abspath(os.path.join(localization_root, filename + ".json"))
        if os.path.exists(full_name):
            try:
                with open(full_name, encoding="utf-8") as f:
                    current_translation = json.load(f)
                    assert isinstance(current_translation, dict)
                    for k, v in current_translation.items():
                        assert isinstance(k, str), f"Key is not a string, got {k}"
                        assert isinstance(v, str) or isinstance(
                            v, list
                        ), f"Value for key {k} is not a string or list"

                    logger.info(f"Loaded localization file {full_name}")
            except Exception as e:
                logger.warning(str(e))
                logger.warning(f"Failed to load localization file {full_name}")
        else:
            logger.warning(f"Localization file {full_name} does not exist")
    else:
        logger.warning(f"Localization file {filename} is not a string")

    # current_translation = {k: 'XXX' for k in current_translation.keys()}  # use this to see if all texts are covered

    return f"window.localization = {json.dumps(current_translation)}"


def dump_english_config(components):
    all_texts = []
    for c in components:
        if isinstance(c, gr.Markdown) and "no-translate" in c.elem_classes:
            continue
        if isinstance(c, gr.Dropdown):
            continue
        if isinstance(c, gr.HTML):
            continue
        if isinstance(c, gr.Textbox):
            continue

        label = getattr(c, "label", None)
        value = getattr(c, "value", None)
        choices = getattr(c, "choices", None)
        info = getattr(c, "info", None)

        if isinstance(label, str):
            all_texts.append(label)
        if isinstance(value, str):
            all_texts.append(value)
        if isinstance(info, str):
            all_texts.append(info)
        if isinstance(choices, list):
            for x in choices:
                if isinstance(x, str):
                    all_texts.append(x)
                if isinstance(x, tuple):
                    for y in x:
                        if isinstance(y, str):
                            all_texts.append(y)

    config_dict = {k: k for k in all_texts if k != "" and "progress-container" not in k}
    full_name = os.path.abspath(os.path.join(localization_root, "en.json"))

    with open(full_name, "w", encoding="utf-8") as json_file:
        json.dump(config_dict, json_file, indent=4, ensure_ascii=False)

    return
