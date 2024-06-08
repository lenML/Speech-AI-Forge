import io
import gradio as gr
import torch

from modules.hf import spaces
from modules.webui.webui_utils import get_speakers, tts_generate
from modules.speaker import speaker_mgr, Speaker

import tempfile


def spk_to_tensor(spk):
    spk = spk.split(" : ")[1].strip() if " : " in spk else spk
    if spk == "None" or spk == "":
        return None
    return speaker_mgr.get_speaker(spk).emb


def get_speaker_show_name(spk):
    if spk.gender == "*" or spk.gender == "":
        return spk.name
    return f"{spk.gender} : {spk.name}"


def merge_spk(
    spk_a,
    spk_a_w,
    spk_b,
    spk_b_w,
    spk_c,
    spk_c_w,
    spk_d,
    spk_d_w,
):
    tensor_a = spk_to_tensor(spk_a)
    tensor_b = spk_to_tensor(spk_b)
    tensor_c = spk_to_tensor(spk_c)
    tensor_d = spk_to_tensor(spk_d)

    assert (
        tensor_a is not None
        or tensor_b is not None
        or tensor_c is not None
        or tensor_d is not None
    ), "At least one speaker should be selected"

    merge_tensor = torch.zeros_like(
        tensor_a
        if tensor_a is not None
        else (
            tensor_b
            if tensor_b is not None
            else tensor_c if tensor_c is not None else tensor_d
        )
    )

    total_weight = 0
    if tensor_a is not None:
        merge_tensor += spk_a_w * tensor_a
        total_weight += spk_a_w
    if tensor_b is not None:
        merge_tensor += spk_b_w * tensor_b
        total_weight += spk_b_w
    if tensor_c is not None:
        merge_tensor += spk_c_w * tensor_c
        total_weight += spk_c_w
    if tensor_d is not None:
        merge_tensor += spk_d_w * tensor_d
        total_weight += spk_d_w

    if total_weight > 0:
        merge_tensor /= total_weight

    merged_spk = Speaker.from_tensor(merge_tensor)
    merged_spk.name = "<MIX>"

    return merged_spk


@torch.inference_mode()
@spaces.GPU
def merge_and_test_spk_voice(
    spk_a, spk_a_w, spk_b, spk_b_w, spk_c, spk_c_w, spk_d, spk_d_w, test_text
):
    merged_spk = merge_spk(
        spk_a,
        spk_a_w,
        spk_b,
        spk_b_w,
        spk_c,
        spk_c_w,
        spk_d,
        spk_d_w,
    )
    return tts_generate(
        spk=merged_spk,
        text=test_text,
    )


@torch.inference_mode()
@spaces.GPU
def merge_spk_to_file(
    spk_a,
    spk_a_w,
    spk_b,
    spk_b_w,
    spk_c,
    spk_c_w,
    spk_d,
    spk_d_w,
    speaker_name,
    speaker_gender,
    speaker_desc,
):
    merged_spk = merge_spk(
        spk_a, spk_a_w, spk_b, spk_b_w, spk_c, spk_c_w, spk_d, spk_d_w
    )
    merged_spk.name = speaker_name
    merged_spk.gender = speaker_gender
    merged_spk.desc = speaker_desc

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        torch.save(merged_spk, tmp_file)
        tmp_file_path = tmp_file.name

    return tmp_file_path


merge_desc = """
## Speaker Merger

在本面板中，您可以选择多个说话人并指定他们的权重，合成新的语音并进行测试。以下是各个功能的详细说明：

### 1. 选择说话人
您可以从下拉菜单中选择最多四个说话人（A、B、C、D），每个说话人都有一个对应的权重滑块，范围从0到10。权重决定了每个说话人在合成语音中的影响程度。

### 2. 合成语音
在选择好说话人和设置好权重后，您可以在“测试文本”框中输入要测试的文本，然后点击“测试语音”按钮来生成并播放合成的语音。

### 3. 保存说话人
您还可以在右侧的“说话人信息”部分填写新的说话人的名称、性别和描述，并点击“保存说话人”按钮来保存合成的说话人。保存后的说话人文件将显示在“合成说话人”栏中，供下载使用。
"""


# 显示 a b c d 四个选择框，选择一个或多个，然后可以试音，并导出
def create_speaker_panel():
    speakers = get_speakers()

    speaker_names = ["None"] + [get_speaker_show_name(speaker) for speaker in speakers]

    with gr.Tabs():
        with gr.TabItem("Merger"):
            gr.Markdown(merge_desc)

            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Group():
                            spk_a = gr.Dropdown(
                                choices=speaker_names, value="None", label="Speaker A"
                            )
                            spk_a_w = gr.Slider(
                                value=1,
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                label="Weight A",
                            )

                        with gr.Group():
                            spk_b = gr.Dropdown(
                                choices=speaker_names, value="None", label="Speaker B"
                            )
                            spk_b_w = gr.Slider(
                                value=1,
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                label="Weight B",
                            )

                        with gr.Group():
                            spk_c = gr.Dropdown(
                                choices=speaker_names, value="None", label="Speaker C"
                            )
                            spk_c_w = gr.Slider(
                                value=1,
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                label="Weight C",
                            )

                        with gr.Group():
                            spk_d = gr.Dropdown(
                                choices=speaker_names, value="None", label="Speaker D"
                            )
                            spk_d_w = gr.Slider(
                                value=1,
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                label="Weight D",
                            )

                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Group():
                                gr.Markdown("🎤Test voice")
                                with gr.Row():
                                    test_voice_btn = gr.Button(
                                        "Test Voice", variant="secondary"
                                    )

                                    with gr.Column(scale=4):
                                        test_text = gr.Textbox(
                                            label="Test Text",
                                            placeholder="Please input test text",
                                            value="说话人合并测试 123456789 [uv_break] ok, test done [lbreak]",
                                        )

                                        output_audio = gr.Audio(label="Output Audio")

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("🗃️Save to file")

                        speaker_name = gr.Textbox(
                            label="Name", value="forge_speaker_merged"
                        )
                        speaker_gender = gr.Textbox(label="Gender", value="*")
                        speaker_desc = gr.Textbox(
                            label="Description", value="merged speaker"
                        )

                        save_btn = gr.Button("Save Speaker", variant="primary")

                        merged_spker = gr.File(
                            label="Merged Speaker", interactive=False, type="binary"
                        )

            test_voice_btn.click(
                merge_and_test_spk_voice,
                inputs=[
                    spk_a,
                    spk_a_w,
                    spk_b,
                    spk_b_w,
                    spk_c,
                    spk_c_w,
                    spk_d,
                    spk_d_w,
                    test_text,
                ],
                outputs=[output_audio],
            )

            save_btn.click(
                merge_spk_to_file,
                inputs=[
                    spk_a,
                    spk_a_w,
                    spk_b,
                    spk_b_w,
                    spk_c,
                    spk_c_w,
                    spk_d,
                    spk_d_w,
                    speaker_name,
                    speaker_gender,
                    speaker_desc,
                ],
                outputs=[merged_spker],
            )
