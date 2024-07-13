from typing import Optional

from pydantic import BaseModel


class TNConfig(BaseModel):
    enabled: Optional[list[str]] = None
    disabled: Optional[list[str]] = None
