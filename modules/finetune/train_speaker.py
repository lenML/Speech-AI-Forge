import torch
import torch.nn.functional as F
import transformers

from modules.finetune.model.encoder import DVAEEncoder, get_encoder_config
from modules.finetune.utils.output import ansi, get_ansi_len, output_iter

from .utils.dataset import AudioCollator, XzListTar
from .utils.logger import MetricLogger
from .utils.model import quantize

IGNORE_TOKEN_ID = transformers.trainer_pt_utils.LabelSmoother.ignore_index


def train_speaker_embeddings(
    chat,
    dataset,
    gpt,
    batch_size=16,
    epochs=10,
    train_text=False,
    train_mse=False,
    speaker_embeds=None,
):
    # NOTE: 新的实现需要改下面的代码 因为 pretrain_models 已经移除
    raise NotImplementedError("train_speaker_embeddings")

    tokenizer = chat.pretrain_models["tokenizer"]

    decoder_decoder = chat.pretrain_models["decoder"]
    decoder_decoder.eval().requires_grad_(False)
    decoder_encoder = DVAEEncoder(**get_encoder_config(decoder_decoder.decoder)).to(
        device=dataset.device
    )
    decoder_encoder.eval().requires_grad_(False)

    dvae_decoder = chat.pretrain_models["dvae"]
    dvae_decoder.eval().requires_grad_(False)
    dvae_encoder = DVAEEncoder(**get_encoder_config(dvae_decoder.decoder)).to(
        device=dataset.device
    )
    dvae_encoder.eval().requires_grad_(False)

    if speaker_embeds is None:
        speaker_embeds = {
            speaker: torch.randn(
                768,
                device=dataset.device,
                requires_grad=True,
            )
            for speaker in dataset.speakers
        }

    for speaker_embed in speaker_embeds.values():
        std, mean = chat.pretrain_models["spk_stat"].chunk(2)
        speaker_embed.data = speaker_embed.data * std + mean

    SPEAKER_TOKEN_ID = tokenizer.convert_tokens_to_ids("[spk_emb]")
    AUDIO_EOS_TOKEN_ID = 0
    AUDIO_PAD_TOKEN_ID = AUDIO_EOS_TOKEN_ID

    optimizer = torch.optim.Adam(
        speaker_embeds.values(), lr=1e-2, weight_decay=0, betas=[0.9, 0.95], eps=1e-5
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-7)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=AudioCollator(text_pad=tokenizer.pad_token_id),
    )
    logger = MetricLogger()
    logger.create_meters(loss=None, mse_loss=None, audio_loss=None, text_loss=None)

    for _epoch in range(epochs):
        _epoch += 1
        logger.reset()
        header = "{blue_light}{0}: {1}{reset}".format(
            "Epoch", output_iter(_epoch, epochs), **ansi
        )
        header = header.ljust(max(len("Epoch"), 30) + get_ansi_len(header))
        iterator = logger.log_every(loader, header=header, tqdm_header="Batch")

        for batch in iterator:
            speakers = batch["speaker"]
            text_input_ids = batch["text_input_ids"]
            text_attention_mask = batch["text_attention_mask"]
            audio_mel_specs = batch["audio_mel_specs"]
            audio_attention_mask = batch["audio_attention_mask"]

            batch_size, text_len = text_attention_mask.size()

            dvae_audio_latents = dvae_encoder(audio_mel_specs, audio_attention_mask)
            _, dvae_audio_input_ids = quantize(
                dvae_decoder.vq_layer.quantizer, dvae_audio_latents
            )
            dvae_audio_input_ids[~audio_attention_mask.bool()] = AUDIO_PAD_TOKEN_ID

            extended_audio_attention_mask = torch.cat(
                [
                    audio_attention_mask,
                    torch.zeros(
                        (batch_size, 1),
                        dtype=audio_attention_mask.dtype,
                        device=audio_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            extended_audio_input_ids = torch.cat(
                [
                    dvae_audio_input_ids,
                    AUDIO_PAD_TOKEN_ID
                    * torch.ones(
                        (batch_size, 1, gpt.num_vq),
                        dtype=dvae_audio_input_ids.dtype,
                        device=dvae_audio_input_ids.device,
                    ),
                ],
                dim=1,
            )
            indices = audio_attention_mask.int().sum(dim=1)
            for i in range(batch_size):
                extended_audio_attention_mask[i, indices[i]] = 1
                extended_audio_input_ids[i, indices[i]] = AUDIO_EOS_TOKEN_ID

            input_ids = torch.cat(
                [
                    text_input_ids.unsqueeze(-1).repeat(1, 1, gpt.num_vq),
                    extended_audio_input_ids,
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [text_attention_mask, extended_audio_attention_mask], dim=1
            )
            text_mask = torch.cat(
                [
                    torch.ones_like(text_attention_mask, dtype=bool),
                    torch.zeros_like(extended_audio_attention_mask, dtype=bool),
                ],
                dim=1,
            )

            labels = input_ids.clone()
            labels[~attention_mask.bool()] = IGNORE_TOKEN_ID

            inputs_embeds = gpt.get_emb(input_ids=input_ids, text_mask=text_mask)

            indices = torch.all(input_ids == SPEAKER_TOKEN_ID, dim=-1)
            for i, speaker in enumerate(speakers):
                inputs_embeds[i, indices[i]] = F.normalize(
                    speaker_embeds[speaker].to(dtype=inputs_embeds.dtype),
                    p=2.0,
                    dim=-1,
                    eps=1e-12,
                ).unsqueeze(0)
            outputs = gpt.gpt.forward(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state
            text_hidden_states = hidden_states[:, : text_len - 1]
            audio_hidden_states = hidden_states[:, text_len - 1 : -1]

            audio_logits = torch.stack(
                [gpt.head_code[i](audio_hidden_states) for i in range(gpt.num_vq)],
                dim=2,
            )
            audio_loss = loss_fn(
                audio_logits.flatten(0, 2), labels[:, text_len:].flatten(0, 2)
            )
            loss = audio_loss

            text_logits = gpt.head_text(text_hidden_states)
            text_loss = loss_fn(
                text_logits.flatten(0, 1), labels[:, 1:text_len, 0].flatten(0, 1)
            )
            loss += text_loss
            logger.meters["text_loss"].update(text_loss.item(), n=batch_size)

            gpt_gen_mel_specs = decoder_decoder(
                audio_hidden_states[:, :-1].transpose(1, 2)
            ).transpose(1, 2)
            mse_loss = torch.nn.functional.mse_loss(gpt_gen_mel_specs, audio_mel_specs)

            optimizer.zero_grad()

            if train_mse:
                loss += 0.01 * mse_loss

            if train_text:
                loss += 0.01 * text_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(speaker_embeds.values(), 1.0)
            optimizer.step()
            logger.meters["loss"].update(loss.item(), n=batch_size)
            logger.meters["mse_loss"].update(mse_loss.item(), n=batch_size)
            logger.meters["audio_loss"].update(audio_loss.item(), n=batch_size)
        lr_scheduler.step()
    optimizer.zero_grad()
    return speaker_embeds


if __name__ == "__main__":
    import argparse
    import os
    import pathlib

    import numpy as np

    from modules import config
    from modules.core.speaker import Speaker
    from modules.devices import devices
    from modules.models import load_chat_tts

    config.runtime_env_vars.no_half = True
    config.runtime_env_vars.use_cpu = []
    devices.reset_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default="./")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_text", action="store_true", help="train text loss")
    parser.add_argument("--train_mse", action="store_true", help="train mse loss")
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
    train_mse: bool = args.train_mse
    # gpt_lora: bool = args.gpt_lora
    # gpt_kbit: int = args.gpt_kbit
    save_folder: str = args.save_folder
    batch_size: int = args.batch_size
    epochs: int = args.epochs
    init_speaker: str = args.init_speaker

    speaker_embeds_save_path = os.path.join(save_folder, "speaker_embeds.npz")

    chat = load_chat_tts()
    dataset = XzListTar(
        root=data_path,
        tokenizer=chat.pretrain_models["tokenizer"],
        vocos_model=chat.pretrain_models["vocos"],
        tar_path=tar_path,
        tar_in_memory=tar_in_memory,
        device=devices.get_device_for("trainer"),
        # speakers=None,  # set(['speaker_A', 'speaker_B'])
    )

    print("len(dataset)", len(dataset))

    speaker_embeds = None
    if init_speaker:
        spk: Speaker = Speaker.from_file(init_speaker)
        speaker_embeds = {
            speaker: torch.tensor(
                spk.emb,
                device=devices.get_device_for("trainer"),
                requires_grad=True,
            )
            for speaker in dataset.speakers
        }

    speaker_embeds = train_speaker_embeddings(
        chat,
        dataset,
        chat.pretrain_models["gpt"],
        batch_size=batch_size,
        epochs=epochs,
        train_text=train_text,
        speaker_embeds=speaker_embeds,
    )
    speaker_outs = {
        speaker: Speaker(speaker_embed.detach().cpu(), f"ep{epochs}_{speaker}")
        for speaker, speaker_embed in speaker_embeds.items()
    }
    time_str = np.datetime_as_string(np.datetime64("now", "s"))
    time_str = time_str.replace(":", "_").replace(" ", "_").replace("-", "_")
    for speaker, speaker_out in speaker_outs.items():
        torch.save(
            speaker_out,
            pathlib.Path(save_folder) / f"spk_{speaker}_{time_str}_ep{epochs}.pt",
        )

# example
"""
python -m modules.finetune.train_speaker \
    --data_path datasets/data_speaker_a/speaker_a.list \
    --save_folder ./data \
    --init_speaker ./data/speakers/Bob.pt \
    --epochs 100 \
    --batch_size 6
"""
