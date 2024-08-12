from dataclasses import dataclass
from typing import Optional

from modules.core.spk.TTSSpeaker import TTSSpeaker


@dataclass(repr=False, eq=False)
class VCConfig:
    enabled: bool = False

    # model id
    mid: str = "open-voice"

    spk: Optional[TTSSpeaker] = None
    emotion: Optional[str] = None

    tau: float = 0.3
