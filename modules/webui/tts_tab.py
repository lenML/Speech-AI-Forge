import gradio as gr

from modules.webui import webui_config
from modules.webui.tts_tabs.tts_infer_ui import TTSInterface
from modules.webui.webui_utils import (
    get_styles,
)


# 没有 seed/style/examples 配置的ui
class NotSeedTTSInterface(TTSInterface):

    def create_tts_style_guide(self):
        pass

    def create_tts_text_guide(self):
        pass

    def create_examples_interface(self, text_input):
        return None

    def create_speaker_picker(self):
        spk_input_dropdown = gr.Dropdown(
            choices=self.speaker_names,
            interactive=True,
            value=self.default_selected_speaker,
            show_label=False,
        )
        spk_input_text = gr.Textbox(
            label="Speaker (Text or Seed)",
            value=self.default_speaker_name,
            show_label=False,
            visible=False,
        )
        spk_input_dropdown.change(
            fn=self.get_speaker_name_from_show_name,
            inputs=[spk_input_dropdown],
            outputs=[spk_input_text],
        )

        reload_button = gr.Button(value="🔄", variant="secondary")

        def reload_spks():
            self.reload_speakers()
            names = self.get_speaker_names()
            return gr.Dropdown(choices=names)

        reload_button.click(
            fn=reload_spks,
            inputs=[],
            outputs=[spk_input_dropdown],
        )

        return spk_input_text, spk_input_dropdown


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

        # NOTE: 只使用 _p 结尾的 因为没有 prompt 在这个模型中没用
        styles: list[str] = [s.get("name") for s in get_styles()]
        self.styles = (
            ["*auto"]
            # NOTE: _p_en 在前面，因为对中文指令识别一般
            + [s for s in styles if s.endswith("_p_en")]
            + [s for s in styles if s.endswith("_p")]
        )

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

        # NOTE: 只使用 _p 的 因为没有 prompt 在这个模型中没用
        styles: list[str] = [s.get("name") for s in get_styles()]
        self.styles = (
            ["*auto"]
            # NOTE: _p_en 在前面，因为对中文指令识别一般
            + [s for s in styles if s.endswith("_p_en")]
            + [s for s in styles if s.endswith("_p")]
        )

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

        # NOTE: 只使用 _p 的 因为没有 prompt 在这个模型中没用
        styles: list[str] = [s.get("name") for s in get_styles()]
        self.styles = (
            ["*auto"]
            # NOTE: _p_en 在前面，因为对中文指令识别一般
            + [s for s in styles if s.endswith("_p_en")]
            + [s for s in styles if s.endswith("_p")]
        )

        self.default_temprature = 0.7
        self.default_top_p = 0.7

        # NOTE: 这个模型不支持 instruction
        self.show_style_dropdown = False


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

        # NOTE: 现在没有SFT版本，效果很差
        with gr.TabItem("FishSpeech"):
            cosy_voice_interface = FishSpeechInterface()
            cosy_voice_interface.create_tts_interface()
