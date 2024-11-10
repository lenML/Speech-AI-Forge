from pydantic import BaseModel


class TTSConfig(BaseModel):
    # model id
    mid: str = "chat-tts"

    style: str = ""
    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20
    prompt: str = ""
    prompt1: str = ""
    prompt2: str = ""
    prefix: str = ""
    emotion: str = ""

    # TODO 添加 dit 配置


class InferConfig(BaseModel):
    batch_size: int = 4
    spliter_threshold: int = 100
    # end_of_sentence
    eos: str = "。"
    seed: int = 42

    stream: bool = False
    stream_chunk_size: int = 64

    no_cache: bool = False

    # 开启同步生成 （主要是给gradio/pytest用）
    sync_gen: bool = False
