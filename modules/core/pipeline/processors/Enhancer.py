from modules.core.models.zoo.ModelZoo import model_zoo
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.processor import NP_AUDIO, AudioProcessor


class EnhancerProcessor(AudioProcessor):
    def _process_array(self, audio: NP_AUDIO, context: TTSPipelineContext) -> NP_AUDIO:
        enhancer_config = context.enhancer_config

        if not enhancer_config.enabled:
            return audio
        nfe = enhancer_config.nfe
        solver = enhancer_config.solver
        lambd = enhancer_config.lambd
        tau = enhancer_config.tau

        sample_rate, audio_data = audio

        model = model_zoo.get_resemble_enhance()
        audio_data, sample_rate = model.apply_audio_enhance_full(
            audio_data=audio_data,
            sr=sample_rate,
            nfe=nfe,
            solver=solver,
            lambd=lambd,
            tau=tau,
        )

        return sample_rate, audio_data
