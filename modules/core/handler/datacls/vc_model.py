from typing import Optional

from pydantic import BaseModel

from modules.core.spk.TTSSpeaker import TTSSpeaker


class VCConfig(BaseModel):
    enabled: bool = False

    # model id
    mid: str = "open-voice"
    emotion: Optional[str] = None

    tau: float = 0.3
