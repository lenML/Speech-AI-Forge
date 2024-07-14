import os
import subprocess


def get_datasets_dir():
    """
    列出 ./datasets/data_* 文件夹
    """
    dataset_path = "./datasets"
    dataset_list = os.listdir(dataset_path)
    dataset_list = [
        d for d in dataset_list if os.path.isdir(os.path.join(dataset_path, d))
    ]
    dataset_list = [d for d in dataset_list if d.startswith("data_")]
    return dataset_list


def get_datasets_listfile():
    datasets = get_datasets_dir()
    listfiles = []
    for d in datasets:
        dir_path = os.path.join("./datasets", d)
        files = os.listdir(dir_path)
        for f in files:
            if f.endswith(".list"):
                listfiles.append(os.path.join(dir_path, f))
    return listfiles


def run_speaker_ft(
    batch_size: int, epochs: int, train_text: bool, data_path: str, init_speaker: str
):
    command = ["python3", "-m", "modules.finetune.train_speaker"]
    command += [
        f"--batch_size={batch_size}",
        f"--epochs={epochs}",
        f"--data_path={data_path}",
    ]
    if train_text:
        command.append("--train_text")
    if init_speaker:
        command.append(f"--init_speaker={init_speaker}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )

    return process
