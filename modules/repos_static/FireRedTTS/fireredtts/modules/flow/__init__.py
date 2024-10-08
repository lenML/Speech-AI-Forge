from fireredtts.modules.flow.codec_embedding import HHGCodecEmbedding
from fireredtts.modules.flow.conformer import ConformerDecoderV2
from fireredtts.modules.flow.mel_encoder import MelReduceEncoder
from fireredtts.modules.flow.decoder import ConditionalCFM, ConditionalDecoder
from fireredtts.modules.flow.flow_model import InterpolateRegulator, CrossAttnFlowMatching
from fireredtts.modules.flow.mel_spectrogram import MelSpectrogramExtractor


def get_flow_frontend(flow_config):
    flow = CrossAttnFlowMatching(
        output_size=flow_config["output_size"],
        input_embedding=HHGCodecEmbedding(**flow_config["input_embedding"]),
        encoder=ConformerDecoderV2(**flow_config["encoder"]),
        length_regulator=InterpolateRegulator(**flow_config["length_regulator"]),
        mel_encoder=MelReduceEncoder(**flow_config["mel_encoder"]),
        decoder=ConditionalCFM(
            estimator=ConditionalDecoder(**flow_config["decoder"]["estimator"]),
            t_scheduler=flow_config["decoder"]["t_scheduler"],
            inference_cfg_rate=flow_config["decoder"]["inference_cfg_rate"]
        )
    )
    return flow


