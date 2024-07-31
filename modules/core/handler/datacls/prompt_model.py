from typing import Optional

from pydantic import BaseModel


# refiner: stage 1
# text => emotion
# text => text
# text => instruction
class RefinerStageConfig(BaseModel):
    enabled: bool = False

    name: Optional[str] = None

    temperature: float = 0.75
    top_p: float = 1
