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

    def __init__(self):
        super().__init__("cosy-voice")
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


class ChatTTSInterface(TTSInterface):
    def __init__(self, model_id="chat-tts"):
        super().__init__(model_id)

        self.support_seed_speaker = True


def create_tts_interface():

    with gr.Tabs():
        with gr.TabItem("ChatTTS"):
            tts_interface = TTSInterface()
            tts_interface.create_tts_interface()
        with gr.TabItem("CosyVoice"):
            cosy_voice_interface = CosyVoiceInterface()
            cosy_voice_interface.create_tts_interface()
        with gr.TabItem("FireRedTTS"):
            fire_red_tts_interface = FireRedTTSInterface()
            fire_red_tts_interface.create_tts_interface()
        with gr.TabItem("F5TTS"):
            f5_tts_interface = F5TtsInterface()
            f5_tts_interface.create_tts_interface()
        with gr.TabItem("IndexTTS"):
            index_tts_interface = IndexTTSInterface()
            index_tts_interface.create_tts_interface()
        with gr.TabItem("SparkTTS"):
            spark_tts_interface = SparkTTSInterface()
            spark_tts_interface.create_tts_interface()
        with gr.TabItem("FishSpeech"):
            fishspeech_interface = FishSpeechInterface()
            fishspeech_interface.create_tts_interface()
        with gr.TabItem("GPT-SoVITS"):
            gpt_sovits_interface = GptSoVitsInterface()
            gpt_sovits_interface.create_tts_interface()
