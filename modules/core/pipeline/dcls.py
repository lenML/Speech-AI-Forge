from dataclasses import dataclass, field
from typing import Literal, Optional

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tn_model import TNConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.spk.TTSSpeaker import TTSSpeaker


@dataclass(repr=False, eq=False)
class TTSSegment:
    _type: Literal["break", "audio"]

    duration_ms: Optional[int] = None
    text: str = ""

    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 20
    infer_seed: int = 42
    prompt: str = ""
    prompt1: str = ""
    prompt2: str = ""
    prefix: str = ""

    emotion: str = ""

    spk: TTSSpeaker = None


@dataclass(repr=False, eq=False)
class TTSPipelineContext:
    text: Optional[str] = None
    ssml: Optional[str] = None

    spk: Optional[TTSSpeaker] = None
    tts_config: TTSConfig = field(default_factory=TTSConfig)
    infer_config: InferConfig = field(default_factory=InferConfig)
    adjust_config: AdjustConfig = field(default_factory=AdjustConfig)
    enhancer_config: EnhancerConfig = field(default_factory=EnhancerConfig)
    vc_config: VCConfig = field(default_factory=VCConfig)

    tn_config: TNConfig = field(default_factory=TNConfig)

    # 当调用 interrupt 时，此变量会被设置为 True
    stop: bool = False
