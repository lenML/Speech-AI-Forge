import gradio as gr

from modules.webui import webui_config
from modules.webui.tts_tabs.tts_infer_ui import TTSInterface
from modules.webui.webui_utils import get_styles


# 没有 seed/style/examples 配置的ui
class NotSeedTTSInterface(TTSInterface):

    def __init__(self, model_id="chat-tts"):
        super().__init__(model_id)

        self.no_seed = True

    def create_tts_style_guide(self):
        pass

    def create_tts_text_guide(self):
        pass

    def create_examples_interface(self, text_input):
        return None

class CosyVoiceInterface(NotSeedTTSInterface):

    def __init__(self, model_id="cosy-voice-2"):
        super().__init__(model_id)
        self.refine_visible = False
        self.contorl_tokens = []
        self.spliter_eos = " 。 "

        # NOTE: 只使用 _p 结尾的 因为没有 prompt 在这个模型中没用
        styles: list[str] = [s.get("name") for s in get_styles()]
        self.styles = (
            ["*auto"]
            # NOTE: _p_en 在前面，因为对中文指令识别一般
            + [s for s in styles if s.endswith("_p_en")]
            + [s for s in styles if s.endswith("_p")]
        )


class FireRedTTSInterface(NotSeedTTSInterface):

    def __init__(self):
        super().__init__("fire-red-tts")
        self.refine_visible = False
        self.contorl_tokens = [
            # NOTE: 貌似还不支持，据github说十月份会开源支持的新版本模型...
            # ref: https://github.com/FireRedTeam/FireRedTTS/issues/12
            # "[elong]",
            # "[oralsii]",
            # "[tsk]",
            # "[breath]",
            # "[laugh]",
            # "(filled pause)",
            # "(confirmation)",
        ]
        self.spliter_eos = " 。 "

        # NOTE: 这个模型不需要参考文本
        self.need_ref_text = False
        self.show_style_dropdown = False

        self.default_temprature = 0.75
        self.default_top_p = 0.85
        self.default_top_k = 30


class F5TtsInterface(NotSeedTTSInterface):

    def __init__(self):
        super().__init__("f5-tts")
        self.refine_visible = False
        self.contorl_tokens = []
        self.spliter_eos = " 。 "

        # NOTE: 这个模型不支持 instruction
        self.show_style_dropdown = False
        # TODO: 其实有采样配置，但是和GPT模型不一样，得重新调整生成流程
        self.show_sampling = False


class FishSpeechInterface(NotSeedTTSInterface):

    def __init__(self):
        super().__init__("fish-speech")
        self.refine_visible = False
        self.contorl_tokens = []
        self.spliter_eos = " 。 "

        self.default_temprature = 0.7
        self.default_top_p = 0.7

        # NOTE: 这个模型不支持 instruction
        self.show_style_dropdown = False


class IndexTTSInterface(NotSeedTTSInterface):

    def __init__(self):
        super().__init__("index-tts")
        self.refine_visible = False
        self.contorl_tokens = []
        self.spliter_eos = " 。 "

        self.styles = ["*auto"]

        # TODO: 这些配置目前其实不支持，需要实现 index-tts infer
        self.default_temprature = 0.7
        self.default_top_p = 0.7

        # NOTE: 这个模型不支持 instruction
        self.show_style_dropdown = False

        # 这个模型几乎不需要 enhance, 或者说，自带 enhance
        self.default_enable_enhance = False


class IndexTTSV2Interface(NotSeedTTSInterface):

    def __init__(self):
        super().__init__("index-tts-v2")
        self.refine_visible = False
        self.contorl_tokens = []
        self.spliter_eos = " 。 "

        self.styles = ["*auto"]

        self.default_temprature = 0.8
        self.default_top_p = 0.8
        self.default_top_k = 30

        # NOTE: 这个模型不支持 instruction
        self.show_style_dropdown = False

        # 这个模型几乎不需要 enhance, 或者说，自带 enhance
        self.default_enable_enhance = False

        # NOTE: 这个用于实现特性 emotion prompt，传给 ttsconfig.emotion
        self.show_emotion_input_box = True


class SparkTTSInterface(NotSeedTTSInterface):

    def __init__(self):
        super().__init__("spark-tts")
        self.refine_visible = False
        self.contorl_tokens = []
        self.spliter_eos = " 。 "

        self.styles = ["*auto"]

        self.default_temprature = 0.8
        self.default_top_p = 0.95
        self.default_top_k = 50

        # NOTE: 这个模型不支持 instruction
        self.show_style_dropdown = False


class GptSoVitsInterface(NotSeedTTSInterface):

    def __init__(self):
        super().__init__("gpt-sovits-v4")
        self.refine_visible = False
        self.contorl_tokens = []
        self.spliter_eos = " 。 "

        self.styles = ["*auto"]

        self.default_temprature = 0.8
        self.default_top_p = 0.95
        self.default_top_k = 50

        # NOTE: 这个模型不支持 instruction
        self.show_style_dropdown = False


class Qwen3TTSInterface(NotSeedTTSInterface):

    def __init__(self, model_id="qwen3-tts-06cv"):
        super().__init__(model_id)
        self.refine_visible = False
        self.contorl_tokens = []
        self.spliter_eos = "\n"

        self.styles = ["*auto"]

        # 来自 modules/repos_static/Qwen3_TTS/qwen_tts/inference/qwen3_tts_model.py 复制出来方便看默认值
        # hard_defaults = dict(
        #     do_sample=True,
        #     top_k=50,
        #     top_p=1.0,
        #     temperature=0.9,
        #     repetition_penalty=1.05,
        #     subtalker_dosample=True,
        #     subtalker_top_k=50,
        #     subtalker_top_p=1.0,
        #     subtalker_temperature=0.9,
        #     max_new_tokens=2048,
        # )

        self.default_temprature = 0.9
        self.default_top_p = 1.0
        self.default_top_k = 50

        # NOTE: 这个模型不支持 instruction
        self.show_style_dropdown = False
        # 这个模型很强不太需要这个
        self.default_enable_enhance = False
        self.default_loudness_norm = False


class Qwen3TTSVoiceDesignInterface(NotSeedTTSInterface):

    def __init__(self, model_id="qwen3-tts-17vd"):
        super().__init__(model_id)
        self.refine_visible = False
        self.contorl_tokens = []
        self.spliter_eos = "\n"

        self.styles = ["*auto"]

        self.default_temprature = 0.9
        self.default_top_p = 1.0
        self.default_top_k = 50

        # NOTE: 这个模型使用的是 prompt 1 作为 voice design ，所以关闭 sytle
        self.show_style_dropdown = False
        # 音色设计模型不太需要这个
        self.support_speaker = False

        # 这个模型很强不太需要这个
        self.default_enable_enhance = False
        self.default_loudness_norm = False
        # 上下文越长，效果越好
        self.def_spliter_thr = 100

    def create_prompt_engineering_interface(self):
        with gr.Group(visible=True):
            gr.Markdown("🔧Prompt engineering")
            prompt1_input = gr.Textbox(
                label="Voice Design",
                lines=10,
                placeholder="Describe the voice characteristics, emotions, and other details you want to achieve.",
            )

            # 这几个是为了保证返回值正确
            prompt2_input = gr.Textbox(label="Prompt 2", visible=False)
            prefix_input = gr.Textbox(label="Prefix", visible=False)
            prompt_audio = gr.File(label="prompt_audio", visible=False)
        return prompt1_input, prompt2_input, prefix_input, prompt_audio


class ChatTTSInterface(TTSInterface):
    def __init__(self, model_id="chat-tts"):
        super().__init__(model_id)

        self.support_seed_speaker = True


def create_tts_interface():
    # ---- 1. 模型注册表（顺序 = 默认顺序）----
    MODELS: list[TTSInterface] = [
        Qwen3TTSInterface("qwen3-tts-17base"),
        Qwen3TTSVoiceDesignInterface(),
        Qwen3TTSInterface("qwen3-tts-17cv"),
        Qwen3TTSInterface("qwen3-tts-06base"),
        Qwen3TTSInterface("qwen3-tts-06cv"),
        IndexTTSV2Interface(),
        CosyVoiceInterface("cosy-voice-2"),
        CosyVoiceInterface("cosy-voice-3"),
        IndexTTSInterface(),
        F5TtsInterface(),
        GptSoVitsInterface(),
        FishSpeechInterface(),
        SparkTTSInterface(),
        FireRedTTSInterface(),
        ChatTTSInterface(),
    ]

    # ---- 2. 从 class 拿 model_id ----
    model_choices = [cls.model_id for cls in MODELS]
    default_model_id = model_choices[0]

    current_model_id = gr.State(default_model_id)

    model_selector = gr.Dropdown(
        choices=model_choices,
        value=default_model_id,
        label="TTS Model",
    )

    # ---- 3. 创建所有 Interface + Group ----
    interfaces = []
    groups = []

    for idx, cls in enumerate(MODELS):
        with gr.Row(visible=(idx == 0)) as group:
            iface = cls
            iface.create_tts_interface()

        interfaces.append(iface)
        groups.append(group)

    # ---- 4. model_id -> index 映射 ----
    model_index_map = {model_id: idx for idx, model_id in enumerate(model_choices)}

    # ---- 5. 切换逻辑 ----
    def switch_model(model_id):
        active_idx = model_index_map[model_id]

        updates = [gr.update(visible=(i == active_idx)) for i in range(len(groups))]

        return [model_id, *updates]

    # ---- 6. 绑定 Dropdown ----
    model_selector.change(
        fn=switch_model,
        inputs=model_selector,
        outputs=[current_model_id, *groups],
    )
