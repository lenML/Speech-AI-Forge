import gradio as gr

from modules.core.models import zoo
from modules.core.spk import spk_mgr
from modules.webui import webui_config
from modules.webui.webui_utils import get_speaker_names

from .ft_ui_utils import get_datasets_listfile
from .ProcessMonitor import ProcessMonitor


class SpeakerFt:
    def __init__(self):
        self.process_monitor = ProcessMonitor()
        self.status_str = "idle"

    def unload_main_thread_models(self):
        zoo.model_zoo.unload_all_models()

    def run(
        self,
        batch_size: int,
        epochs: int,
        lr: str,
        train_text: bool,
        data_path: str,
        select_speaker: str = "",
    ):
        if self.process_monitor.process:
            return
        self.unload_main_thread_models()
        spk_path = None
        if select_speaker != "" and select_speaker != "none":
            select_speaker = select_speaker.split(" : ")[1].strip()
            spk = spk_mgr.get_speaker(select_speaker)
            if spk is None:
                return ["Speaker not found"]
            spk_filename = spk_mgr.get_item_path(lambda x: x.id == spk.id)
            spk_path = f"./data/speakers/{spk_filename}"

        command = ["python3", "-m", "modules.finetune.train_speaker"]
        command += [
            f"--batch_size={batch_size}",
            f"--epochs={epochs}",
            f"--data_path={data_path}",
        ]
        if train_text:
            command.append("--train_text")
        if spk_path:
            command.append(f"--init_speaker={spk_path}")

        self.status("Training process starting")

        self.process_monitor.start_process(command)

        self.status("Training started")

    def status(self, text: str):
        self.status_str = text

    def flush(self):
        stdout, stderr = self.process_monitor.get_output()
        return f"{self.status_str}\n{stdout}\n{stderr}"

    def clear(self):
        self.process_monitor.stdout = ""
        self.process_monitor.stderr = ""
        self.status("Logs cleared")

    def stop(self):
        self.process_monitor.stop_process()
        self.status("Training stopped")


def create_speaker_ft_tab(demo: gr.Blocks):
    spk_ft = SpeakerFt()
    speakers, speaker_names = get_speaker_names()
    speaker_names = ["none"] + speaker_names

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("🎛️hparams")
                dataset_input = gr.Dropdown(
                    label="Dataset", choices=get_datasets_listfile()
                )
                lr_input = gr.Textbox(label="Learning Rate", value="1e-2")
                epochs_input = gr.Slider(
                    label="Epochs", value=10, minimum=1, maximum=100, step=1
                )
                batch_size_input = gr.Slider(
                    label="Batch Size", value=4, minimum=1, maximum=64, step=1
                )
                train_text_checkbox = gr.Checkbox(label="Train text_loss", value=True)
                init_spk_dropdown = gr.Dropdown(
                    label="Initial Speaker",
                    choices=speaker_names,
                    value="none",
                )

            with gr.Group():
                start_train_btn = gr.Button("Start Training")
                stop_train_btn = gr.Button("Stop Training")
                clear_train_btn = gr.Button("Clear logs")
        with gr.Column(scale=5):
            with gr.Group():
                # log
                gr.Markdown("📜logs")
                log_output = gr.Textbox(
                    show_label=False, label="Log", value="", lines=20, interactive=True
                )

    start_train_btn.click(
        spk_ft.run,
        inputs=[
            batch_size_input,
            epochs_input,
            lr_input,
            train_text_checkbox,
            dataset_input,
            init_spk_dropdown,
        ],
        outputs=[],
    )
    stop_train_btn.click(spk_ft.stop)
    clear_train_btn.click(spk_ft.clear)

    if webui_config.experimental:
        demo.load(spk_ft.flush, every=1, outputs=[log_output])
