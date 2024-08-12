from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.models.BaseZooModel import BaseZooModel
from modules.core.pipeline.processor import NP_AUDIO


class VCModel(BaseZooModel):

    def convert(self, src_audio: NP_AUDIO, config: VCConfig) -> NP_AUDIO:
        raise NotImplementedError
