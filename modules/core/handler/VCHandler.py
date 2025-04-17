from typing import AsyncGenerator, Generator

from modules.core.handler.AudioHandler import AudioHandler
from modules.core.handler.datacls.audio_model import EncoderConfig
from modules.core.handler.datacls.tts_model import InferConfig
from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.models.vc.VCModel import VCModel
from modules.core.models.zoo.ModelZoo import model_zoo
from modules.core.pipeline.processor import NP_AUDIO
from modules.core.spk.TTSSpeaker import TTSSpeaker


class VCHandler(AudioHandler):

    def __init__(
        self,
        input_audio: NP_AUDIO,
        ref_spk: TTSSpeaker,
        vc_config: VCConfig,
        encoder_config: EncoderConfig,
    ) -> None:
        super().__init__(encoder_config=encoder_config, infer_config=InferConfig())

        assert isinstance(vc_config, VCConfig), "vc_config must be VCConfig"
        assert isinstance(ref_spk, TTSSpeaker), "spk must be TTSSpeaker"

        self.ref_spk = ref_spk
        self.input_audio = input_audio
        self.vc_config = vc_config
        self.model: VCModel = self.get_model()

        if self.model is None:
            raise Exception(f"Model {self.vc_config.mid} is not supported")

    def get_model(self) -> VCModel:
        model_id = (
            self.vc_config.mid.lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .strip()
        )
        if model_id.startswith("openvoice"):
            return model_zoo.get_model(model_id="open-voice")

        raise Exception(f"Model {model_id} is not supported")

    async def enqueue(self) -> NP_AUDIO:
        result = self.model.convert(
            src_audio=self.input_audio, config=self.vc_config, ref_spk=self.ref_spk
        )
        return result

    async def enqueue_stream(self) -> AsyncGenerator[NP_AUDIO, None]:
        raise NotImplementedError(
            "Method 'enqueue_stream' not implemented in VCHandler"
        )

    def get_sample_rate(self):
        return self.model.get_sample_rate()
