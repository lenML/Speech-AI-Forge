from typing import Optional

from pydantic import BaseModel


class SpeakerReference(BaseModel):
    wav_b64: str
    text: str


class SpeakerConfig(BaseModel):
    """
    任选其中一种形式指定 spk
    """

    from_spk_id: Optional[str] = None
    from_spk_name: Optional[str] = None
    from_ref: Optional[SpeakerReference] = None
