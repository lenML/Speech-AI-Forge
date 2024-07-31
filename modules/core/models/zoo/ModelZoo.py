from typing import Dict, Union

from modules.core.models.tts.ChatTtsModel import ChatTTSModel
from modules.core.models.tts.CosyVoiceModel import CosyVoiceTTSModel
from modules.core.models.tts.FishSpeechModel import FishSpeechModel
from modules.core.models.TTSModel import TTSModel
from modules.devices import devices


class ModelZoo:
    """
    管理控制 model load/unload/download/checker
    """

    models: Dict[str, TTSModel] = {
        "chat-tts": ChatTTSModel(),
        "fish-speech": FishSpeechModel(),
        "cosy-voice": CosyVoiceTTSModel(),
    }

    # 当mem不足时，是否自动卸载其他模型
    auto_unload = True

    def __init__(self) -> None:
        pass

    def get_model(self, model_id: str) -> Union[TTSModel, None]:
        return self.models[model_id]

    def unload_all_models(self, exclude=None):
        for model in self.models.values():
            if model == exclude:
                continue
            model.unload()

    def is_not_engouh_mem(self):
        usage = devices.get_memory_usage()
        return usage.free_mb < (2 * 1024)

    def load_model(self, model_id: str):
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        if self.auto_unload and self.is_not_engouh_mem():
            self.unload_all_models(model)
        model.load()
        return model

    def unload_model(self, model_id: str):
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        model.unload()
        return model


model_zoo = ModelZoo()
