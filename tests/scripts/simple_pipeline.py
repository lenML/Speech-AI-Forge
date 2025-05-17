import soundfile as sf

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tn_model import TNConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.pipeline.factory import PipelineFactory
from modules.core.pipeline.processor import TTSPipelineContext
from modules.core.spk.SpkMgr import spk_mgr

context = TTSPipelineContext(
    ssml="""
    <speak version="0.1">
        <voice spk="Bob_ft10">
            hello world! 123456789
            hello world! 123456789
            hello world! 123456789
        </voice>
        <voice spk="Bob_ft10">
            你好，世界！ 123456789
            你好，世界！ 123456789
            你好，世界！ 123456789
        </voice>
    </speak>
    """,
    spk=spk_mgr.get_speaker("Bob_ft10"),
    tts_config=TTSConfig(),
    infer_config=InferConfig(no_cache=True),
    adjust_config=AdjustConfig(),
    enhancer_config=EnhancerConfig(),
    tn_config=TNConfig(),
)

pipeline = PipelineFactory.create(context)

with open("output_pipeline.wav", "wb") as f:
    sr, audio = pipeline.generate()
    sf.write(f, audio, sr)

with open("output_pipeline_stream.wav", "wb") as f:
    for sr, audio in pipeline.generate_stream():
        sf.write(f, audio, sr)
