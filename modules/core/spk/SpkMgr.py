import json
import logging
from typing import Union

import torch

from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.core.tools.FolderDatabase import FolderDatabase


class SpeakerManager(FolderDatabase):
    logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__("./data/speakers/")

    def is_valid_file(self, file_path: str) -> bool:
        return file_path.endswith(".spkv1.json") or file_path.endswith(".spkv1.png")

    def load_item(self, file_path: str) -> Union[TTSSpeaker, None]:
        if file_path.endswith(".spkv1.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    return TTSSpeaker.from_json(data)
                except json.JSONDecodeError:
                    logging.error("Invalid JSON file: " + file_path)
                    return None
        elif file_path.endswith(".spkv1.png"):
            raise NotImplementedError("Loading speaker images is not yet implemented")
        else:
            raise ValueError("Invalid file extension for speaker file: " + file_path)

    def save_item(self, item: TTSSpeaker, file_path: str):
        if file_path.endswith(".spkv1.json"):
            with open(file_path, "w", encoding="utf-8") as f:
                json_str = item.to_json_str()
                f.write(json_str)
        elif file_path.endswith(".spkv1.png"):
            raise NotImplementedError("Saving speaker images is not yet implemented")
        else:
            raise ValueError("Invalid file extension for speaker file: " + file_path)

    def get_speaker(self, name: str) -> Union[TTSSpeaker, None]:
        return self.get_item(lambda x: x.name == name)

    def get_speaker_by_id(self, id: str) -> Union[TTSSpeaker, None]:
        return self.get_item(lambda x: x.id == id)

    def list_speakers(self) -> list[TTSSpeaker]:
        return list(self.items.values())


spk_mgr = SpeakerManager()
