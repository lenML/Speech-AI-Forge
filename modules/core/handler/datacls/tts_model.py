from pydantic import BaseModel


class TTSConfig(BaseModel):
    # model id
    mid: str = "chat-tts"

    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20

    # 这几个可以理解都是prompt插槽，具体怎么用看推理代码实现
    # sytle: 用于指定我们内部系统中的 styles ，等于是预设，这个推理的时候就合并到 prompt 中了
    # TODO: 计划完全简化 sytle 不应该存在这个东西，和其他插槽重叠，功能也不是很完善

    # prompt: 此为特殊值一般是合成 prompt 来自 style 指定
    # prompt1: 正式的prompt插槽位置
    # prompt2: 正式的prompt插槽位置 第二个
    # prefix: 特殊插槽位置，约等于 context 上下文，类比 whisper 中的 prefix 设置
    style: str = ""
    prompt: str = ""
    prompt1: str = ""
    prompt2: str = ""
    prefix: str = ""

    # 指定使用哪个特定的子 reference
    emotion: str = ""
    # 设置 emotion prompt ，只有特定模型支持
    emotion_prompt: str = ""

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
