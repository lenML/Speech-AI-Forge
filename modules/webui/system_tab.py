import os
import time

import gradio as gr

from modules.core.models.zoo import model_zoo
from modules.devices import devices
from modules.webui import webui_config

is_huggingface_space = os.environ.get("IS_HF_SPACE", None) is not None


def get_system_status():
    cpu_mem = devices.get_cpu_memory()
    gpu_mem = devices.get_gpu_memory()

    status = (
        f"**CPU Memory**:\n"
        f"- Total: {cpu_mem.total_mb:.2f} MB\n"
        f"- Used: {cpu_mem.used_mb:.2f} MB\n"
        f"- Free: {cpu_mem.free_mb:.2f} MB\n\n"
        f"**GPU Memory**:\n"
        f"- Total: {gpu_mem.total_mb:.2f} MB\n"
        f"- Used: {gpu_mem.used_mb:.2f} MB\n"
        f"- Free: {gpu_mem.free_mb:.2f} MB"
    )
    return status

# NOTE: 这样写会抢占输出队列...
# def update_markdown():
#     while True:
#         yield get_system_status()
#         time.sleep(5)


def create_system_tab(demo: gr.Blocks):
    with gr.Row() as r:
        with gr.Column(scale=1):
            gr.Markdown(f"info")

            with gr.Group():
                gr.Markdown("🏗 System Monitor")
                status_box = gr.Markdown(value="")

                # 每秒更新一次状态
                demo.load(fn=get_system_status, inputs=[], outputs=status_box, every=5)

        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("🚩 Features")
                toggle_experimental = gr.Checkbox(
                    label="Enable Experimental Features",
                    value=webui_config.experimental,
                    interactive=False,
                )

            with gr.Group():
                gr.Markdown("🦁Model zoo")
                with gr.Row():
                    with gr.Column():
                        unload_models_btn = gr.Button(
                            "Unload All Models",
                            interactive=is_huggingface_space is False,
                        )
                    with gr.Column():
                        reload_models_btn = gr.Button(
                            "Reload All Models",
                            interactive=is_huggingface_space is False,
                        )

                def reload_models():
                    model_zoo.reload_all_models()

                def unload_models():
                    model_zoo.unload_all_models()

                unload_models_btn.click(fn=unload_models, inputs=None, outputs=None)
                reload_models_btn.click(fn=reload_models, inputs=None, outputs=None)
