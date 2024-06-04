import os
import torch

from modules import models
from modules.utils.SeedContext import SeedContext

import uuid


def create_speaker_from_seed(seed):
    chat_tts = models.load_chat_tts()
    with SeedContext(seed):
        emb = chat_tts.sample_random_speaker()
    return emb


class Speaker:
    def __init__(self, seed, name="", gender="", describe=""):
        self.id = uuid.uuid4()
        self.seed = seed
        self.name = name
        self.gender = gender
        self.describe = describe
        self.emb = None

    def to_json(self):
        return {
            "id": str(self.id),
            "seed": self.seed,
            "name": self.name,
            "gender": self.gender,
            "describe": self.describe,
            # "emb": self.emb.tolist(),
        }

    def fix(self):
        is_update = False
        if "id" not in self.__dict__:
            setattr(self, "id", uuid.uuid4())
            is_update = True
        if "seed" not in self.__dict__:
            setattr(self, "seed", -2)
            is_update = True
        if "name" not in self.__dict__:
            setattr(self, "name", "")
            is_update = True
        if "gender" not in self.__dict__:
            setattr(self, "gender", "*")
            is_update = True
        if "describe" not in self.__dict__:
            setattr(self, "describe", "")
            is_update = True

        return is_update


# 每个speaker就是一个 emb 文件 .pt
# 管理 speaker 就是管理 ./data/speaker/ 下的所有 speaker
# 可以 用 seed 创建一个 speaker
# 可以 刷新列表 读取所有 speaker
# 可以列出所有 speaker
class SpeakerManager:
    def __init__(self):
        self.speakers = []
        self.speaker_dir = "./data/speakers/"
        self.refresh_speakers()

    def refresh_speakers(self):
        self.speakers = []
        for speaker in os.listdir(self.speaker_dir):
            if speaker.endswith(".pt"):
                speaker = torch.load(
                    self.speaker_dir + speaker, map_location=torch.device("cpu")
                )
                self.speakers.append(speaker)

                is_update = speaker.fix()
                if is_update:
                    torch.save(speaker, self.speaker_dir + speaker.name + ".pt")

    def list_speakers(self):
        return self.speakers

    def create_speaker(self, seed, name="", gender=""):
        if name == "":
            name = seed
        speaker = Speaker(seed, name=name, gender=gender)
        speaker.emb = create_speaker_from_seed(seed)
        torch.save(speaker, self.speaker_dir + name + ".pt")
        self.refresh_speakers()
        return speaker

    def create_speaker_from_tensor(
        self, tensor, filename="", name="", gender="", describe=""
    ):
        if name == "":
            name = filename
        speaker = Speaker(seed=-2, name=name, gender=gender, describe=describe)
        if isinstance(tensor, torch.Tensor):
            speaker.emb = tensor
        if isinstance(tensor, list):
            speaker.emb = torch.tensor(tensor)
        torch.save(speaker, self.speaker_dir + filename + ".pt")
        self.refresh_speakers()
        return speaker

    def get_speaker(self, name) -> Speaker | None:
        try:
            return torch.load(self.speaker_dir + name + ".pt")
        except FileNotFoundError:
            return None

    def __len__(self):
        return len(self.speakers)


speaker_mgr = SpeakerManager()
