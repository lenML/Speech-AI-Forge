import os
import uuid
from typing import Union

import torch
from box import Box

from modules import models
from modules.utils.SeedContext import SeedContext


def create_speaker_from_seed(seed):
    chat_tts = models.load_chat_tts()
    with SeedContext(seed, True):
        emb = chat_tts.sample_random_speaker()
    return emb


class Speaker:
    @staticmethod
    def from_file(file_like):
        speaker = torch.load(file_like, map_location=torch.device("cpu"))
        speaker.fix()
        return speaker

    @staticmethod
    def from_tensor(tensor):
        speaker = Speaker(seed_or_tensor=-2)
        speaker.emb = tensor
        return speaker

    @staticmethod
    def from_seed(seed: int):
        speaker = Speaker(seed_or_tensor=seed)
        speaker.emb = create_speaker_from_seed(seed)
        return speaker

    def __init__(
        self, seed_or_tensor: Union[int, torch.Tensor], name="", gender="", describe=""
    ):
        self.id = uuid.uuid4()
        self.seed = -2 if isinstance(seed_or_tensor, torch.Tensor) else seed_or_tensor
        self.name = name
        self.gender = gender
        self.describe = describe
        self.emb = None if isinstance(seed_or_tensor, int) else seed_or_tensor

        # TODO replace emb => tokens
        self.tokens = []

    def to_json(self, with_emb=False):
        return Box(
            **{
                "id": str(self.id),
                "seed": self.seed,
                "name": self.name,
                "gender": self.gender,
                "describe": self.describe,
                "emb": self.emb.tolist() if with_emb else None,
            }
        )

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

    def __hash__(self):
        return hash(str(self.id))

    def __eq__(self, other):
        if not isinstance(other, Speaker):
            return False
        return str(self.id) == str(other.id)


# 每个speaker就是一个 emb 文件 .pt
# 管理 speaker 就是管理 ./data/speaker/ 下的所有 speaker
# 可以 用 seed 创建一个 speaker
# 可以 刷新列表 读取所有 speaker
# 可以列出所有 speaker
class SpeakerManager:
    def __init__(self):
        self.speakers = {}
        self.speaker_dir = "./data/speakers/"
        self.refresh_speakers()

    def refresh_speakers(self):
        self.speakers = {}
        for speaker_file in os.listdir(self.speaker_dir):
            if speaker_file.endswith(".pt"):
                self.speakers[speaker_file] = Speaker.from_file(
                    self.speaker_dir + speaker_file
                )
        # 检查是否有被删除的，同步到 speakers
        for fname, spk in self.speakers.items():
            if not os.path.exists(self.speaker_dir + fname):
                del self.speakers[fname]

    def list_speakers(self) -> list[Speaker]:
        return list(self.speakers.values())

    def create_speaker_from_seed(self, seed, name="", gender="", describe=""):
        if name == "":
            name = seed
        filename = name + ".pt"
        speaker = Speaker(seed, name=name, gender=gender, describe=describe)
        speaker.emb = create_speaker_from_seed(seed)
        torch.save(speaker, self.speaker_dir + filename)
        self.refresh_speakers()
        return speaker

    def create_speaker_from_tensor(
        self, tensor, filename="", name="", gender="", describe=""
    ):
        if filename == "":
            filename = name
        speaker = Speaker(
            seed_or_tensor=-2, name=name, gender=gender, describe=describe
        )
        if isinstance(tensor, torch.Tensor):
            speaker.emb = tensor
        if isinstance(tensor, list):
            speaker.emb = torch.tensor(tensor)
        torch.save(speaker, self.speaker_dir + filename + ".pt")
        self.refresh_speakers()
        return speaker

    def get_speaker(self, name) -> Union[Speaker, None]:
        for speaker in self.speakers.values():
            if speaker.name == name:
                return speaker
        return None

    def get_speaker_by_id(self, id) -> Union[Speaker, None]:
        for speaker in self.speakers.values():
            if str(speaker.id) == str(id):
                return speaker
        return None

    def get_speaker_filename(self, id: str):
        filename = None
        for fname, spk in self.speakers.items():
            if str(spk.id) == str(id):
                filename = fname
                break
        return filename

    def update_speaker(self, speaker: Speaker):
        filename = None
        for fname, spk in self.speakers.items():
            if str(spk.id) == str(speaker.id):
                filename = fname
                break

        if filename:
            torch.save(speaker, self.speaker_dir + filename)
            self.refresh_speakers()
            return speaker
        else:
            raise ValueError("Speaker not found for update")

    def save_all(self):
        for speaker in self.speakers.values():
            filename = self.get_speaker_filename(speaker.id)
            torch.save(speaker, self.speaker_dir + filename)
        # self.refresh_speakers()

    def __len__(self):
        return len(self.speakers)


speaker_mgr = SpeakerManager()
