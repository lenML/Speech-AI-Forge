import gradio as gr
import torch
from modules.normalization import text_normalize
from modules.webui.webui_utils import (
    get_speakers,
    get_styles,
    split_long_text,
)
from modules.hf import spaces


# NOTE: 因为 text_normalize 需要使用 tokenizer
@torch.inference_mode()
@spaces.GPU
def merge_dataframe_to_ssml(dataframe, spk, style, seed):
    if style == "*auto":
        style = None
    if spk == "-1" or spk == -1:
        spk = None
    if seed == -1 or seed == "-1":
        seed = None

    ssml = ""
    indent = " " * 2

    for i, row in dataframe.iterrows():
        ssml += f"{indent}<voice"
        if spk:
            ssml += f' spk="{spk}"'
        if style:
            ssml += f' style="{style}"'
        if seed:
            ssml += f' seed="{seed}"'
        ssml += ">\n"
        ssml += f"{indent}{indent}{text_normalize(row.iloc[1])}\n"
        ssml += f"{indent}</voice>\n"
    return f"<speak version='0.1'>\n{ssml}</speak>"


# 长文本处理
# 可以输入长文本，并选择切割方法，切割之后可以将拼接的SSML发送到SSML tab
# 根据 。 句号切割，切割之后显示到 data table
def create_spliter_tab(ssml_input, tabs):
    speakers = get_speakers()

    def get_speaker_show_name(spk):
        if spk.gender == "*" or spk.gender == "":
            return spk.name
        return f"{spk.gender} : {spk.name}"

    speaker_names = ["*random"] + [
        get_speaker_show_name(speaker) for speaker in speakers
    ]

    styles = ["*auto"] + [s.get("name") for s in get_styles()]

    with gr.Row():
        with gr.Column(scale=1):
            # 选择说话人 选择风格 选择seed
            with gr.Group():
                gr.Markdown("🗣️Speaker")
                spk_input_text = gr.Textbox(
                    label="Speaker (Text or Seed)",
                    value="female2",
                    show_label=False,
                )
                spk_input_dropdown = gr.Dropdown(
                    choices=speaker_names,
                    interactive=True,
                    value="female : female2",
                    show_label=False,
                )
                spk_rand_button = gr.Button(
                    value="🎲",
                    variant="secondary",
                )
            with gr.Group():
                gr.Markdown("🎭Style")
                style_input_dropdown = gr.Dropdown(
                    choices=styles,
                    interactive=True,
                    show_label=False,
                    value="*auto",
                )
            with gr.Group():
                gr.Markdown("🗣️Seed")
                infer_seed_input = gr.Number(
                    value=42,
                    label="Inference Seed",
                    show_label=False,
                    minimum=-1,
                    maximum=2**32 - 1,
                )
                infer_seed_rand_button = gr.Button(
                    value="🎲",
                    variant="secondary",
                )

            send_btn = gr.Button("📩Send to SSML", variant="primary")

        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("📝Long Text Input")
                gr.Markdown("- 此页面用于处理超长文本")
                gr.Markdown("- 切割后，可以选择说话人、风格、seed，然后发送到SSML")
                long_text_input = gr.Textbox(
                    label="Long Text Input",
                    lines=10,
                    placeholder="输入长文本",
                    elem_id="long-text-input",
                    show_label=False,
                )
                long_text_split_button = gr.Button("🔪Split Text")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("🎨Output")
                long_text_output = gr.DataFrame(
                    headers=["index", "text", "length"],
                    datatype=["number", "str", "number"],
                    elem_id="long-text-output",
                    interactive=False,
                    wrap=True,
                    value=[],
                )

    spk_input_dropdown.change(
        fn=lambda x: x.startswith("*") and "-1" or x.split(":")[-1].strip(),
        inputs=[spk_input_dropdown],
        outputs=[spk_input_text],
    )
    spk_rand_button.click(
        lambda x: int(torch.randint(0, 2**32 - 1, (1,)).item()),
        inputs=[spk_input_text],
        outputs=[spk_input_text],
    )
    infer_seed_rand_button.click(
        lambda x: int(torch.randint(0, 2**32 - 1, (1,)).item()),
        inputs=[infer_seed_input],
        outputs=[infer_seed_input],
    )
    long_text_split_button.click(
        split_long_text,
        inputs=[long_text_input],
        outputs=[long_text_output],
    )

    infer_seed_rand_button.click(
        lambda x: int(torch.randint(0, 2**32 - 1, (1,)).item()),
        inputs=[infer_seed_input],
        outputs=[infer_seed_input],
    )

    send_btn.click(
        merge_dataframe_to_ssml,
        inputs=[
            long_text_output,
            spk_input_text,
            style_input_dropdown,
            infer_seed_input,
        ],
        outputs=[ssml_input],
    )

    def change_tab():
        return gr.Tabs(selected="ssml")

    send_btn.click(change_tab, inputs=[], outputs=[tabs])
