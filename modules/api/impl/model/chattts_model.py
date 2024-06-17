from pydantic import BaseModel


class ChatTTSConfig(BaseModel):
    style: str = ""
    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20
    prompt1: str = ""
    prompt2: str = ""
    prefix: str = ""


class InferConfig(BaseModel):
    batch_size: int = 4
    spliter_threshold: int = 100
    # end_of_sentence
    eos: str = "[uv_break]"
    seed: int = 42
