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
    # NOTE: batch_size * spliter_threshold = 预计最大vram面积 * 不同模型的系数
    # 大概 batch_sise=2 spliter_threshold=30 可以保证在8gb显存正常推理
    batch_size: int = 2
    spliter_threshold: int = 30

    # end_of_sentence
    eos: str = "。"
    seed: int = 42

    stream: bool = False
    stream_chunk_size: int = 64

    no_cache: bool = False

    # 开启同步生成 （主要是给gradio/pytest用）
    sync_gen: bool = False
    # 超时设置 默认为 60 * 15 十五分钟
    timeout: int = 60 * 15
