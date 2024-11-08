try:
    from modules.repos_static.sys_paths import setup_repos_paths

    setup_repos_paths()
except:
    pass

from typing import Dict, Union

from modules.core.models.BaseZooModel import BaseZooModel
from modules.core.models.TTSModel import TTSModel
from modules.core.models.enhancer.ResembleEnhanceModel import ResembleEnhanceModel
from modules.core.models.stt.Whisper import WhisperModel
from modules.core.models.tts.ChatTtsModel import ChatTTSModel
from modules.core.models.tts.CosyVoiceModel import CosyVoiceTTSModel
from modules.core.models.tts.F5TtsModel import F5TtsModel
from modules.core.models.tts.FireRed.FireRedTTSModel import FireRedTTSModel
from modules.core.models.tts.FishSpeechModel import FishSpeechModel
from modules.core.models.tts.fishspeech.FishSpeech14Model import FishSpeech14Model
from modules.core.models.vc.OpenVoice import OpenVoiceModel
from modules.devices import devices


class ModelZoo:
    """
    管理控制 model load/unload/download/checker
    """

    models: Dict[str, BaseZooModel] = {
        # === tts ===
        "chat-tts": ChatTTSModel(),
        "fish-speech": FishSpeech14Model(),
        "cosy-voice": CosyVoiceTTSModel(),
        "fire-red-tts": FireRedTTSModel(),
        "f5-tts": F5TtsModel(),
        # === enhancer ===
        "resemble-enhance": ResembleEnhanceModel(),
        # === whisper ===
        "whisper": WhisperModel("whisper.large"),
        # "whisper.large": WhisperModel("whisper.large"),
        # "whisper.medium": WhisperModel("whisper.medium"),
        # "whisper.small": WhisperModel("whisper.small"),
        # "whisper.tiny": WhisperModel("whisper.tiny"),
        # === voice clone ===
        "open-voice": OpenVoiceModel(),
    }

    # 当mem不足时，是否自动卸载其他模型
    auto_unload = True

    def __init__(self) -> None:
        pass

    def get_model(self, model_id: str) -> Union[BaseZooModel, None]:
        return self.models[model_id]

    def get_tts_model_ids(self) -> list[str]:
        ids = []
        for id, model in self.models.items():
            if isinstance(model, TTSModel):
                ids.append(id)
        return ids

    def get_available_tts_model(self) -> list[TTSModel]:
        models = []
        for id, model in self.models.items():
            if isinstance(model, TTSModel) and model.is_downloaded():
                models.append(model)
        return models

    def get_model_ids(self) -> list[str]:
        ids = []
        for id, model in self.models.items():
            ids.append(id)
        return ids

    @devices.after_gc()
    def unload_all_models(self, exclude=None):
        for model in self.models.values():
            if model == exclude:
                continue
            model.unload()

        devices.do_gc()

    @devices.after_gc()
    def reload_all_models(self, exclude=None):
        loaded_models = [model for model in self.models.values() if model.is_loaded()]
        self.unload_all_models(exclude=exclude)

        for model in loaded_models:
            model.load()

        devices.do_gc()

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

    @devices.after_gc()
    def unload_model(self, model_id: str):
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        model.unload()
        return model

    # --------------- getters --------------

    def get_chat_tts(self) -> ChatTTSModel:
        return self.get_model("chat-tts")

    def get_cosy_voice(self) -> CosyVoiceTTSModel:
        return self.get_model("cosy-voice")

    def get_fish_speech(self) -> FishSpeech14Model:
        return self.get_model("fish-speech")

    def get_resemble_enhance(self) -> ResembleEnhanceModel:
        return self.get_model("resemble-enhance")

    def get_whisper(self) -> WhisperModel:
        return self.get_model("whisper")

    def get_open_voice(self) -> OpenVoiceModel:
        return self.get_model("open-voice")

    def get_fire_red_tts(self) -> FireRedTTSModel:
        return self.get_model("fire-red-tts")

    def get_f5_tts(self) -> F5TtsModel:
        return self.get_model("f5-tts")


model_zoo = ModelZoo()
