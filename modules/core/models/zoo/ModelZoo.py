try:
    from modules.repos_static.sys_paths import setup_repos_paths

    setup_repos_paths()
except:
    pass

import logging
from typing import Dict, Union

from modules.core.models.BaseZooModel import BaseZooModel
from modules.core.models.enhancer.ResembleEnhanceModel import ResembleEnhanceModel
from modules.core.models.stt.SenseVoice import SenseVoiceModel
from modules.core.models.stt.STTModel import STTModel
from modules.core.models.stt.Whisper import WhisperModel
from modules.core.models.tts.ChatTtsModel import ChatTTSModel
from modules.core.models.tts.CosyVoiceModel import CosyVoiceTTSModel
from modules.core.models.tts.F5TtsModel import F5TtsModel
from modules.core.models.tts.FireRed.FireRedTTSModel import FireRedTTSModel
from modules.core.models.tts.GptSoVits.GptSoVitsModel import GptSoVitsModel
from modules.core.models.tts.fishspeech.FishSpeech14Model import FishSpeech14Model
from modules.core.models.tts.IndexTTS.IndexTTSModel import IndexTTSModel
from modules.core.models.tts.SparkTTS.SparkTTSModel import SparkTTSModel
from modules.core.models.TTSModel import TTSModel
from modules.core.models.vc.OpenVoice import OpenVoiceModel
from modules.devices import devices

logger = logging.getLogger(__name__)

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
        "index-tts": IndexTTSModel(),
        "spark-tts": SparkTTSModel(),
        "gpt-sovits-v1": GptSoVitsModel("v1"),
        "gpt-sovits-v2": GptSoVitsModel("v2"),
        "gpt-sovits-v3": GptSoVitsModel("v3"),
        "gpt-sovits-v4": GptSoVitsModel("v4"),
        # === enhancer ===
        "resemble-enhance": ResembleEnhanceModel(),
        # === whisper ===
        "whisper.large": WhisperModel("whisper.large"),
        "whisper.turbo": WhisperModel("whisper.turbo"),
        # "whisper.large": WhisperModel("whisper.large"),
        # "whisper.medium": WhisperModel("whisper.medium"),
        # "whisper.small": WhisperModel("whisper.small"),
        # "whisper.tiny": WhisperModel("whisper.tiny"),
        "sensevoice": SenseVoiceModel(),
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

    def get_stt_model_ids(self) -> list[str]:
        ids = []
        for id, model in self.models.items():
            if isinstance(model, STTModel):
                ids.append(id)
        return ids

    def get_available_stt_model(self) -> list[STTModel]:
        models = []
        for id, model in self.models.items():
            if isinstance(model, STTModel) and model.is_downloaded():
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
            logger.info(f"Model {model.model_id} unloaded")

    @devices.after_gc()
    def reload_all_models(self, exclude=None):
        loaded_models = [model for model in self.models.values() if model.is_loaded()]
        self.unload_all_models(exclude=exclude)

        for model in loaded_models:
            model.load()

    def is_not_engouh_mem(self):
        usage = devices.get_memory_usage()
        # TODO: 这里其实应该至少基于评估模型文件大小来判断...
        return usage.free_mb < (2 * 1024)

    def load_model(self, model_id: str):
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        if self.auto_unload and self.is_not_engouh_mem():
            logger.info("Not enough memory, auto unload other models")
            self.unload_all_models(model)
        model.load()
        logger.info(f"Model {model_id} loaded")
        return model

    @devices.after_gc()
    def unload_model(self, model_id: str):
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        model.unload()
        logger.info(f"Model {model_id} unloaded")
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

    def get_whisper_turbo(self) -> WhisperModel:
        return self.get_model("whisper.turbo")

    def get_sensevoice(self) -> SenseVoiceModel:
        return self.get_model("sensevoice")

    def get_open_voice(self) -> OpenVoiceModel:
        return self.get_model("open-voice")

    def get_fire_red_tts(self) -> FireRedTTSModel:
        return self.get_model("fire-red-tts")

    def get_f5_tts(self) -> F5TtsModel:
        return self.get_model("f5-tts")

    def get_index_tts(self) -> IndexTTSModel:
        return self.get_model("index-tts")

    def get_spark_tts(self) -> SparkTTSModel:
        return self.get_model("spark-tts")

    def get_gpt_sovits_v1(self) -> GptSoVitsModel:
        return self.get_model("gpt-sovits-v1")

    def get_gpt_sovits_v2(self) -> GptSoVitsModel:
        return self.get_model("gpt-sovits-v2")

    def get_gpt_sovits_v3(self) -> GptSoVitsModel:
        return self.get_model("gpt-sovits-v3")

    def get_gpt_sovits_v4(self) -> GptSoVitsModel:
        return self.get_model("gpt-sovits-v4")


model_zoo = ModelZoo()
