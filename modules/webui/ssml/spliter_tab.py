import gradio as gr
import torch

from modules.normalization import text_normalize
from modules.utils.hf import spaces
from modules.webui import webui_utils
from modules.webui.webui_utils import get_speakers, get_styles, split_long_text


# NOTE: 因为 text_normalize 需要使用 tokenizer
@torch.inference_mode()
@spaces.GPU(duration=120)
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
        text = row.iloc[1]
        text = text_normalize(text)

        if text.strip() == "":
            continue

        ssml += f"{indent}<voice"
        if spk:
            ssml += f' spk="{spk}"'
        if style:
            ssml += f' style="{style}"'
        if seed:
            ssml += f' seed="{seed}"'
        ssml += ">\n"
        ssml += f"{indent}{indent}{text}\n"
        ssml += f"{indent}</voice>\n"
    # 原封不动输出回去是为了触发 loadding 效果
    return dataframe, spk, style, seed, f"<speak version='0.1'>\n{ssml}</speak>"


# 长文本处理
# 可以输入长文本，并选择切割方法，切割之后可以将拼接的SSML发送到SSML tab
# 根据 。 句号切割，切割之后显示到 data table
def create_spliter_tab(ssml_input, tabs1, tabs2):
    speakers, speaker_names = webui_utils.get_speaker_names()
    speaker_names = ["*random"] + speaker_names

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
                gr.Markdown("💃Inference Seed")
                infer_seed_input = gr.Number(
                    value=42,
                    label="Inference Seed",
                    show_label=False,
                    minimum=-1,
                    maximum=2**32 - 1,
                )
                infer_seed_rand_button = gr.Button(
                    value="🎲",
                    # tooltip="Random Seed",
                    variant="secondary",
                )

            with gr.Group():
                gr.Markdown("🎛️Spliter")
                eos_input = gr.Textbox(
                    label="eos",
                    value="[uv_break]",
                )
                spliter_thr_input = gr.Slider(
                    label="Spliter Threshold",
                    value=100,
                    minimum=50,
                    maximum=1000,
                    step=1,
                )

        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("📝Long Text Input")
                gr.Markdown("SSML_SPLITER_GUIDE")
                long_text_input = gr.Textbox(
                    label="Long Text Input",
                    lines=10,
                    placeholder="输入长文本",
                    elem_id="long-text-input",
                    show_label=False,
                )
                long_text_split_button = gr.Button("🔪Split Text")

            with gr.Group():
                gr.Markdown("🎨Output")
                long_text_output = gr.DataFrame(
                    headers=["index", "text", "length"],
                    datatype=["number", "str", "number"],
                    elem_id="long-text-output",
                    interactive=True,
                    wrap=True,
                    value=[],
                    row_count=(0, "dynamic"),
                    col_count=(3, "fixed"),
                )

                send_btn = gr.Button("📩Send to SSML", variant="primary")

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
        inputs=[
            long_text_input,
            spliter_thr_input,
            eos_input,
        ],
        outputs=[
            long_text_output,
        ],
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
        outputs=[
            long_text_output,
            spk_input_text,
            style_input_dropdown,
            infer_seed_input,
            ssml_input,
        ],
    )

    def change_tab():
        return gr.Tabs(selected="ssml"), gr.Tabs(selected="ssml.editor")

    send_btn.click(change_tab, inputs=[], outputs=[tabs1, tabs2])
