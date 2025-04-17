from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.models.BaseZooModel import BaseZooModel
from modules.core.pipeline.processor import NP_AUDIO
from modules.core.spk.TTSSpeaker import TTSSpeaker


class VCModel(BaseZooModel):

    def convert(
        self, src_audio: NP_AUDIO, ref_spk: TTSSpeaker, config: VCConfig
    ) -> NP_AUDIO:
        raise NotImplementedError

    def get_sample_rate(self) -> int:
        raise NotImplementedError
