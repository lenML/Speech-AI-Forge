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
    def __init__(self, seed, name):
        self.id = uuid.uuid4()
        self.seed = seed
        self.name = name
        self.emb = create_speaker_from_seed(seed)

    def to_json(self):
        return {
            "id": str(self.id),
            "seed": self.seed,
            "name": self.name,
            # "emb": self.emb.tolist(),
        }


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
                self.speakers.append(torch.load(self.speaker_dir + speaker))

    def list_speakers(self):
        return self.speakers

    def create_speaker(self, seed, name=""):
        if name == "":
            name = seed
        speaker = Speaker(seed, name)
        torch.save(speaker, self.speaker_dir + name + ".pt")
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
