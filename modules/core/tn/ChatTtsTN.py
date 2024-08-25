from modules.core.models import zoo
from modules.core.tn.TNPipeline import GuessLang

from .base_tn import BaseTN

ChatTtsTN = BaseTN.clone()
ChatTtsTN.freeze_tokens = [
    "[Sasr]",
    "[Pasr]",
    "[Easr]",
    "[Stts]",
    "[Ptts]",
    "[Etts]",
    "[Sbreak]",
    "[Pbreak]",
    "[Ebreak]",
    "[uv_break]",
    "[v_break]",
    "[lbreak]",
    "[llbreak]",
    "[undefine]",
    "[laugh]",
    "[spk_emb]",
    "[empty_spk]",
    "[music]",
    "[pure]",
    "[break_0]",
    "[break_1]",
    "[break_2]",
    "[break_3]",
    "[break_4]",
    "[break_5]",
    "[break_6]",
    "[break_7]",
    "[laugh_0]",
    "[laugh_1]",
    "[laugh_2]",
    "[oral_0]",
    "[oral_1]",
    "[oral_2]",
    "[oral_3]",
    "[oral_4]",
    "[oral_5]",
    "[oral_6]",
    "[oral_7]",
    "[oral_8]",
    "[oral_9]",
    "[speed_0]",
    "[speed_1]",
    "[speed_2]",
    "[speed_3]",
    "[speed_4]",
    "[speed_5]",
    "[speed_6]",
    "[speed_7]",
    "[speed_8]",
    "[speed_9]",
]


@ChatTtsTN.block()
def replace_unk_tokens(text: str, guess_lang: GuessLang):
    """
    把不在字典里的字符替换为 " , "

    FIXME: 总感觉不太好...但是没有遇到问题的话暂时留着...
    """
    # NOTE: 太影响性能了，放弃
    return text
    # chat_tts = zoo.ChatTTS.load_chat_tts()
    # if chat_tts.tokenizer._tokenizer is None:
    #     # 这个地方只有在 huggingface spaces 中才会触发
    #     # 因为 hugggingface 自动处理模型卸载加载，所以如果拿不到就算了...
    #     return text
    # tokenizer = zoo.ChatTTS.get_tokenizer()
    # vocab = tokenizer.get_vocab()
    # vocab_set = set(vocab.keys())
    # # 添加所有英语字符
    # vocab_set.update(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    # vocab_set.update(set(" \n\r\t"))
    # replaced_chars = [char if char in vocab_set else " , " for char in text]
    # output_text = "".join(replaced_chars)
    # return output_text


if __name__ == "__main__":
    from modules.devices import devices

    ChatTtsTN.remove_block("replace_unk_tokens")

    devices.reset_device()
    test_cases = [
        "ChatTTS是专门为对话场景设计的文本转语音模型，例如LLM助手对话任务。它支持英文和中文两种语言。最大的模型使用了10万小时以上的中英文数据进行训练。在HuggingFace中开源的版本为4万小时训练且未SFT的版本.",
        " [oral_9] [laugh_0] [break_0] 电 [speed_0] 影 [speed_0] 中 梁朝伟 [speed_9] 扮演的陈永仁的编号27149",
        " 明天有62％的概率降雨",
        "大🍌，一条大🍌，嘿，你的感觉真的很奇妙  [lbreak]",
        "I like eating 🍏",
        """
# 你好，世界
```js
console.log('1')
```
**加粗**

*一条文本*
        """,
        """
在沙漠、岩石、雪地上行走了很长的时间以后，小王子终于发现了一条大路。所有的大路都是通往人住的地方的。
“你们好。”小王子说。
这是一个玫瑰盛开的花园。
“你好。”玫瑰花说道。
小王子瞅着这些花，它们全都和他的那朵花一样。
“你们是什么花？”小王子惊奇地问。
“我们是玫瑰花。”花儿们说道。
“啊！”小王子说……。
        """,
        """
State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.

🤗 Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

📝 Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
🖼️ Computer Vision: image classification, object detection, and segmentation.
🗣️ Audio: automatic speech recognition and audio classification.
🐙 Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.
        """,
        """
120米
有12%的概率会下雨
埃隆·马斯克
""",
    ]

    for i, test_case in enumerate(test_cases):
        print(f"case {i}:\n", {"x": ChatTtsTN.normalize(test_case)})
