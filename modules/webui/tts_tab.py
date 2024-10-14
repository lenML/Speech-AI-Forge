import gradio as gr
import torch

from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.webui import webui_config
from modules.webui.webui_utils import (
    SPK_FILE_EXTS,
    get_speakers,
    get_styles,
    load_spk_info,
    refine_text,
    tts_generate,
)


class TTSInterface:
    def __init__(self, model_id="chat-tts"):
        def spk_filter(spk: TTSSpeaker):
            if spk.has_refs:
                return True
            if spk.get_token(model_id=model_id) is not None:
                return True
            return False

        self.speakers = get_speakers(spk_filter)
        self.speaker_names = self.get_speaker_names()
        self.styles = ["*auto"] + [s.get("name") for s in get_styles()]

        self.default_selected_speaker = (
            self.speaker_names[1] if len(self.speaker_names) > 1 else "*random"
        )
        self.default_speaker_name = self.get_speaker_name_from_show_name(
            self.default_selected_speaker
        )

        self.model_id = model_id
        self.refine_visible = True

        self.contorl_tokens = ["[laugh]", "[uv_break]", "[v_break]", "[lbreak]"]

        self.spliter_eos = "[uv_break]"

        self.default_content = webui_config.localization.DEFAULT_TTS_TEXT

        # 是否需要参考文本，某些模型不需要，直接填一个占位符然后隐藏
        self.need_ref_text = True

        self.default_temprature = 0.3
        self.default_top_p = 0.7
        self.default_top_k = 20

        self.show_style_dropdown = True

    def get_speaker_names(self):
        names = ["*random"] + [
            self.get_speaker_show_name(speaker) for speaker in self.speakers
        ]
        names.sort(key=lambda x: x.startswith("*") and "-1" or x)
        return names

    @staticmethod
    def get_speaker_show_name(spk: TTSSpeaker):
        if spk.gender == "*" or spk.gender == "":
            return spk.name
        return f"{spk.gender} : {spk.name}"

    @staticmethod
    def get_speaker_name_from_show_name(spk_show_name: str):
        if spk_show_name == "*random":
            return "-1"
        if ":" in spk_show_name:
            return spk_show_name.split(":")[-1].strip()
        return spk_show_name

    def tts_generate_with_history(
        self,
        audio_history: list,
        progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
        *args,
        **kwargs,
    ):
        audio = tts_generate(model_id=self.model_id, *args, **kwargs)
        mp3_1 = audio
        mp3_2 = audio_history[-1] if len(audio_history) > 0 else None
        mp3_3 = audio_history[-2] if len(audio_history) > 1 else None
        return mp3_1, mp3_2, mp3_3, [mp3_3, mp3_2, mp3_1]

    def create_speaker_picker(self):
        spk_input_text = gr.Textbox(
            label="Speaker (Text or Seed)",
            value=self.default_speaker_name,
            show_label=False,
        )
        spk_input_dropdown = gr.Dropdown(
            choices=self.speaker_names,
            interactive=True,
            value=self.default_selected_speaker,
            show_label=False,
        )
        spk_rand_button = gr.Button(value="🎲", variant="secondary")

        spk_input_dropdown.change(
            fn=self.get_speaker_name_from_show_name,
            inputs=[spk_input_dropdown],
            outputs=[spk_input_text],
        )
        spk_rand_button.click(
            lambda x: str(torch.randint(0, 2**32 - 1, (1,)).item()),
            inputs=[spk_input_text],
            outputs=[spk_input_text],
        )
        return spk_input_text, spk_input_dropdown, spk_rand_button

    def create_speaker_interface(self):
        with gr.Group():
            gr.Markdown("🗣️Speaker")
            with gr.Tabs():
                with gr.Tab(label="Pick"):
                    spk_input_text, spk_input_dropdown, spk_rand_button = (
                        self.create_speaker_picker()
                    )

                with gr.Tab(label="Upload"):
                    spk_file_upload = gr.File(
                        label="Speaker (Upload)", file_types=SPK_FILE_EXTS
                    )
                    gr.Markdown("📝Speaker info")
                    infos = gr.Markdown("empty", elem_classes=["no-translate"])
                    spk_file_upload.change(
                        fn=load_spk_info, inputs=[spk_file_upload], outputs=[infos]
                    )

                with gr.Tab(label="Refrence"):
                    # 使用参考音频
                    ref_audio_upload = gr.Audio(label="Refrence Audio")

                    text_visible = self.need_ref_text == True
                    ref_text_input = gr.Textbox(
                        label="Refrence Text",
                        placeholder="Text from refrence audio",
                        value="" if text_visible else "无意义文本",
                        show_label=False,
                        lines=1,
                        visible=text_visible,
                    )

        return (
            spk_input_text,
            spk_input_dropdown,
            spk_file_upload,
            ref_audio_upload,
            ref_text_input,
        )

    def create_tts_style_guide(self):
        gr.Markdown("TTS_STYLE_GUIDE")

    def create_style_interface(self):
        with gr.Group(visible=self.show_style_dropdown):
            gr.Markdown("🎭Style")
            self.create_tts_style_guide()
            style_input_dropdown = gr.Dropdown(
                choices=self.styles, interactive=True, show_label=False, value="*auto"
            )
        return style_input_dropdown

    def create_sampling_interface(self):
        with gr.Group():
            gr.Markdown("🎛️Sampling")
            temperature_input = gr.Slider(
                0.01, 2.0, value=self.default_temprature, step=0.01, label="Temperature"
            )
            top_p_input = gr.Slider(
                0.1, 1.0, value=self.default_top_p, step=0.1, label="Top P"
            )
            top_k_input = gr.Slider(
                1, 50, value=self.default_top_k, step=1, label="Top K"
            )
            batch_size_input = gr.Slider(
                1, webui_config.max_batch_size, value=4, step=1, label="Batch Size"
            )
        return temperature_input, top_p_input, top_k_input, batch_size_input

    def create_splitter_interface(self):
        with gr.Group():
            gr.Markdown("🎛️Spliter")
            eos_input = gr.Textbox(label="eos", value=self.spliter_eos)
            spliter_thr_input = gr.Slider(
                label="Spliter Threshold", value=100, minimum=50, maximum=1000, step=1
            )
        return eos_input, spliter_thr_input

    def create_tts_text_guide(self):
        gr.Markdown(f"TTS_TEXT_GUIDE")

    def create_text_input_interface(self):
        with gr.Group():
            input_title = gr.Markdown("📝Text Input", elem_id="input-title")
            self.create_tts_text_guide()
            text_input = gr.Textbox(
                show_label=False,
                label="Text to Speech",
                lines=10,
                placeholder="输入文本或选择示例",
                elem_id="text-input",
                value=self.default_content,
            )

            with gr.Row(visible=len(self.contorl_tokens) > 1):
                for tk in self.contorl_tokens:
                    t_btn = gr.Button(tk)
                    t_btn.click(
                        lambda text, tk=tk: text + " " + tk,
                        inputs=[text_input],
                        outputs=[text_input],
                    )

        return text_input

    def create_examples_interface(self, text_input):
        with gr.Group():
            gr.Markdown("🎄Examples")
            sample_dropdown = gr.Dropdown(
                choices=[
                    sample["text"] for sample in webui_config.localization.tts_examples
                ],
                show_label=False,
                value=None,
                interactive=True,
            )
            sample_dropdown.change(
                fn=lambda x: x, inputs=[sample_dropdown], outputs=[text_input]
            )
        return sample_dropdown

    def create_output_interface(self):
        with gr.Group():
            gr.Markdown("🎨Output")
            audio_history = gr.State([])
            tts_output1 = gr.Audio(label="Generated Audio", format="mp3")
            tts_output2 = gr.Audio(label="History -1", format="mp3")
            tts_output3 = gr.Audio(label="History -2", format="mp3")
        return audio_history, tts_output1, tts_output2, tts_output3

    def create_refiner_interface(self):
        with gr.Group(visible=self.refine_visible):
            gr.Markdown("🎶Refiner")
            rf_oral_input = gr.Slider(
                label="Oral", value=2, minimum=-1, maximum=9, step=1
            )
            rf_speed_input = gr.Slider(
                label="Speed", value=2, minimum=-1, maximum=9, step=1
            )
            rf_break_input = gr.Slider(
                label="Break", value=2, minimum=-1, maximum=7, step=1
            )
            rf_laugh_input = gr.Slider(
                label="Laugh", value=0, minimum=-1, maximum=2, step=1
            )
            refine_button = gr.Button("✍️Refine Text")
        return (
            rf_oral_input,
            rf_speed_input,
            rf_break_input,
            rf_laugh_input,
            refine_button,
        )

    def create_prompt_engineering_interface(self):
        with gr.Group(visible=webui_config.experimental):
            gr.Markdown("🔧Prompt engineering")
            prompt1_input = gr.Textbox(label="Prompt 1")
            prompt2_input = gr.Textbox(label="Prompt 2")
            prefix_input = gr.Textbox(label="Prefix")
            prompt_audio = gr.File(
                label="prompt_audio", visible=webui_config.experimental
            )
        return prompt1_input, prompt2_input, prefix_input, prompt_audio

    def create_adjuster_interface(self):
        with gr.Group():
            gr.Markdown("🎛️Adjuster")
            speed_input = gr.Slider(
                label="Speed", value=1.0, minimum=0.5, maximum=2.0, step=0.1
            )
            pitch_input = gr.Slider(
                label="Pitch", value=0, minimum=-12, maximum=12, step=0.1
            )
            volume_up_input = gr.Slider(
                label="Volume Gain", value=0, minimum=-20, maximum=6, step=0.1
            )
            enable_loudness_normalization = gr.Checkbox(
                value=True, label="Enable Loudness EQ"
            )
            headroom_input = gr.Slider(
                label="Headroom", value=1, minimum=0, maximum=12, step=0.1
            )
        return (
            speed_input,
            pitch_input,
            volume_up_input,
            enable_loudness_normalization,
            headroom_input,
        )

    def create_inference_seed_interface(self):
        with gr.Group():
            gr.Markdown("💃Inference Seed")
            infer_seed_input = gr.Number(
                value=42,
                label="Inference Seed",
                show_label=False,
                minimum=-1,
                maximum=2**32 - 1,
            )
            infer_seed_rand_button = gr.Button(value="🎲", variant="secondary")
            infer_seed_rand_button.click(
                lambda x: int(torch.randint(0, 2**32 - 1, (1,)).item()),
                inputs=[infer_seed_input],
                outputs=[infer_seed_input],
            )
        return infer_seed_input

    def create_generate_interface(self):
        with gr.Group():
            gr.Markdown("🔊Generate")
            disable_normalize_input = gr.Checkbox(
                value=False, label="Disable Normalize", visible=False
            )
            with gr.Group():
                enable_enhance = gr.Checkbox(value=True, label="Enable Enhance")
                enable_de_noise = gr.Checkbox(value=False, label="Enable De-noise")
            tts_button = gr.Button(
                "🔊Generate Audio", variant="primary", elem_classes="big-button"
            )
        return disable_normalize_input, enable_enhance, enable_de_noise, tts_button

    def create_tts_interface(self):
        with gr.Row():
            with gr.Column(scale=1):
                (
                    spk_input_text,
                    spk_input_dropdown,
                    spk_file_upload,
                    ref_audio_upload,
                    ref_text_input,
                ) = self.create_speaker_interface()
                style_input_dropdown = self.create_style_interface()
                temperature_input, top_p_input, top_k_input, batch_size_input = (
                    self.create_sampling_interface()
                )
                eos_input, spliter_thr_input = self.create_splitter_interface()
                use_decoder_input = gr.Checkbox(
                    value=True, label="Use Decoder", visible=False
                )

            with gr.Column(scale=4):
                text_input = self.create_text_input_interface()
                sample_dropdown = self.create_examples_interface(text_input=text_input)
                audio_history, tts_output1, tts_output2, tts_output3 = (
                    self.create_output_interface()
                )

            with gr.Column(scale=1):
                (
                    rf_oral_input,
                    rf_speed_input,
                    rf_break_input,
                    rf_laugh_input,
                    refine_button,
                ) = self.create_refiner_interface()
                prompt1_input, prompt2_input, prefix_input, prompt_audio = (
                    self.create_prompt_engineering_interface()
                )
                (
                    speed_input,
                    pitch_input,
                    volume_up_input,
                    enable_loudness_normalization,
                    headroom_input,
                ) = self.create_adjuster_interface()
                infer_seed_input = self.create_inference_seed_interface()
                disable_normalize_input, enable_enhance, enable_de_noise, tts_button = (
                    self.create_generate_interface()
                )

        refine_button.click(
            refine_text,
            inputs=[
                text_input,
                rf_oral_input,
                rf_speed_input,
                rf_break_input,
                rf_laugh_input,
            ],
            outputs=[text_input],
        )

        tts_button.click(
            self.tts_generate_with_history,
            inputs=[
                audio_history,
                text_input,
                temperature_input,
                top_p_input,
                top_k_input,
                spk_input_text,
                infer_seed_input,
                use_decoder_input,
                prompt1_input,
                prompt2_input,
                prefix_input,
                style_input_dropdown,
                disable_normalize_input,
                batch_size_input,
                enable_enhance,
                enable_de_noise,
                spk_file_upload,
                spliter_thr_input,
                eos_input,
                pitch_input,
                speed_input,
                volume_up_input,
                enable_loudness_normalization,
                headroom_input,
                ref_audio_upload,
                ref_text_input,
            ],
            outputs=[tts_output1, tts_output2, tts_output3, audio_history],
        )


class CosyVoiceInterface(TTSInterface):

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

        spk_rand_button = gr.Button(value="🎲", variant="secondary", visible=False)

        return spk_input_text, spk_input_dropdown, spk_rand_button


class FireRedTTSInterface(TTSInterface):

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

        spk_rand_button = gr.Button(value="🎲", variant="secondary", visible=False)

        return spk_input_text, spk_input_dropdown, spk_rand_button


class FishSpeechInterface(TTSInterface):

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

        spk_rand_button = gr.Button(value="🎲", variant="secondary", visible=False)

        return spk_input_text, spk_input_dropdown, spk_rand_button


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

        # NOTE: 现在没有SFT版本，效果很差
        with gr.TabItem("FishSpeech", visible=webui_config.experimental):
            cosy_voice_interface = FishSpeechInterface()
            cosy_voice_interface.create_tts_interface()
