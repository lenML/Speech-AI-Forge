from dataclasses import dataclass
from typing import Literal, Optional

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.chattts_model import ChatTTSConfig, InferConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tn_model import TNConfig
from modules.core.spk.TTSSpeaker import TTSSpeaker


@dataclass(repr=False, eq=False)
class TTSSegment:
    _type: Literal["break", "audio"]
    duration_s: int = 0

    text: str = ""
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 20
    infer_seed: int = 42
    prompt: str = ""
    prompt1: str = ""
    prompt2: str = ""
    prefix: str = ""

    spk: TTSSpeaker = None


@dataclass(repr=False, eq=False)
class TTSPipelineContext:
    text: Optional[str] = None
    ssml: Optional[str] = None

    spk: Optional[TTSSpeaker] = None
    tts_config: ChatTTSConfig = ChatTTSConfig()
    infer_config: InferConfig = InferConfig()
    adjust_config: AdjustConfig = AdjustConfig()
    enhancer_config: EnhancerConfig = EnhancerConfig()

    tn_config: TNConfig = TNConfig()

    stop: bool = False
