import torch

from modules.repos_static.ChatTTS.ChatTTS.core import Chat

from .dataset import XzListTar
from .train import TrainModule, train_gpt


def train_speaker_embeddings(
    chat: Chat,
    dataset: XzListTar,
    batch_size=16,
    epochs=10,
    train_text=False,
    speaker_embeds=None,
) -> dict[str, torch.Tensor]:
    return train_gpt(
        chat=chat,
        dataset=dataset,
        train_module=TrainModule.SPEAKER,
        batch_size=batch_size,
        epochs=epochs,
        train_text=train_text,
        speaker_embeds=speaker_embeds,
    )


if __name__ == "__main__":
    import argparse
    import os
    import pathlib

    import numpy as np

    from modules import config
    from modules.core.models.tts.ChatTTS.ChatTTS import load_chat_tts
    from modules.core.spk.TTSSpeaker import TTSSpeaker
    from modules.devices import devices

    config.runtime_env_vars.no_half = True
    config.runtime_env_vars.use_cpu = []
    devices.reset_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default="./")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_text", action="store_true", help="train text loss")
    # 初始化 speaker
    parser.add_argument("--init_speaker", type=str)
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/data_speaker_a/speaker_a.list",
        help="the data_path to json/list file",
    )
    parser.add_argument("--tar_path", type=str, help="the tarball path with wavs")
    parser.add_argument(
        "--tar_in_memory", action="store_true", help="load tarball in memory"
    )

    args = parser.parse_args()

    data_path: str = args.data_path
    tar_path: str | None = args.tar_path
    tar_in_memory: bool = args.tar_in_memory
    train_text: bool = args.train_text
    # gpt_lora: bool = args.gpt_lora
    # gpt_kbit: int = args.gpt_kbit
    save_folder: str = args.save_folder
    batch_size: int = args.batch_size
    epochs: int = args.epochs
    init_speaker: str = args.init_speaker

    speaker_embeds_save_path = os.path.join(save_folder, "speaker_embeds.npz")

    chat: Chat = load_chat_tts()
    dataset = XzListTar(
        root=data_path,
        tokenizer=chat.tokenizer._tokenizer,
        tar_path=tar_path,
    )

    print("len(dataset)", len(dataset))

    speaker_embeds = None
    if init_speaker:
        spk: TTSSpeaker = TTSSpeaker.from_file(init_speaker)
        if spk.get_token("chat-tts") is None:
            raise ValueError("this init_speaker is not ChatTTS speaker")
        token: torch.Tensor = spk.get_token("chat-tts").tokens[0]
        if not isinstance(token, torch.Tensor):
            raise ValueError("cant get ChatTTS token")
        speaker_embeds = {speaker: token.clone() for speaker in dataset.speakers}

    speaker_embeds = train_speaker_embeddings(
        chat,
        dataset,
        batch_size=batch_size,
        epochs=epochs,
        train_text=train_text,
        speaker_embeds=speaker_embeds,
    )

    def create_spk(token, name: str):
        spk = TTSSpeaker.from_token(
            model_id="chat-tts",
            tokens=[token],
        )
        spk.set_name(name)
        return spk

    speaker_outs = {
        speaker: create_spk(
            token=speaker_embed.detach().cpu(), name=f"ep{epochs}_{speaker}"
        )
        for speaker, speaker_embed in speaker_embeds.items()
    }
    time_str = np.datetime_as_string(np.datetime64("now", "s"))
    time_str = time_str.replace(":", "_").replace(" ", "_").replace("-", "_")
    for speaker, speaker_out in speaker_outs.items():
        filepath = (
            pathlib.Path(save_folder)
            / f"spk_{speaker}_{time_str}_ep{epochs}.spkv1.json"
        )
        json_str = speaker_out.to_json_str()
        with open(filepath, "+w", encoding="utf-8") as f:
            f.write(json_str)

# example
"""
python -m modules.finetune.train_speaker \
    --data_path datasets/data_speaker_a/speaker_a.list \
    --save_folder ./data \
    --init_speaker ./data/speakers/Bob.spkv1.json \
    --epochs 100 \
    --batch_size 6
    

python -m modules.finetune.train_speaker \
    --data_path datasets/data_lian/lian.list \
    --save_folder ./data \
    --init_speaker ./data/speakers/Bob.spkv1.json \
    --epochs 100 \
    --batch_size 20 \
    --train_text
"""
